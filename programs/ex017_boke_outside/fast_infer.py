"""小さな VLM の分類器（ex016 model.py．InternVL3.5 と Qwen3-VL）の推論を，お題ごとのプレフィックスのキャッシュで高速化する．

通常の推論（ex016 train.predict）は，ボケ1件ごとに「お題画像（256トークン）＋お題の文＋ボケ」の全体（約300トークン）を計算する．
同じお題のボケは「ボケ：「」までのプレフィックス（画像を含む）が共通であるため，次のように計算を分ける．
1. お題ごとに，プレフィックス（画像の符号化を含む）を1回だけ計算し，言語モデルの KV キャッシュを得る．
2. キャッシュをそのお題のボケの数だけ複製し（reorder_cache），ボケの文と閉じ括弧以降（約20〜60トークン）だけを計算する．
3. 各ボケの最後のトークンの隠れ状態から，発話の型と画像外の知識の確率を求める．
InternVL3.5 は画像のトークン数が一定（256）のため，プレフィックスの長さが同じお題をまとめて計算する．
Qwen3-VL は画像ごとにトークン数が異なり，位置が3次元（M-RoPE）であるため，プレフィックスを左詰めでまとめて計算し，
ボケの部分の位置は get_rope_index で求めたプレフィックスの最大の位置＋1から続ける．
プレフィックスとボケの部分は別々に文字列を分けてトークン化するため，境界のトークン化が通常の推論とわずかに異なる場合がある．
そのため，検証データで通常の推論との一致を確かめる（--check）．

速度（--bench）：BoketeDBv4 から無作為に選んだお題の全てのボケ（実際の1お題あたりの件数の分布を保つ）を，
通常の推論（バッチ32・64）と高速化した推論で処理し，画像の読み込み・前処理を含む件数/秒を測る．

出力：outputs/ex017_boke_outside/fast/{check,bench}.json

実行例:
    .venv_annotate/bin/python programs/ex017_boke_outside/fast_infer.py --weights <best_model.pt> --model iv2 --check --bench
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX16_DIR, OUTPUT_DIR, VOICES, D16, _load, image_path, load_eval  # noqa: E402

# ex016 の model.py と train.py は `from data import ...` で ex016 の data を読むため，読み込む間だけ差し替える
_own = sys.modules["data"]
sys.modules["data"] = D16
M16 = _load("ex016_model_for_v5", os.path.join(EX16_DIR, "model.py"))
sys.modules["model"] = M16
T16 = _load("ex016_train_for_v5", os.path.join(EX16_DIR, "train.py"))
sys.modules["data"] = _own
SPLIT_MARK = "ボケ：「"


class GroupDataset(torch.utils.data.Dataset):
    """お題ごとに，プレフィックスのトークン・画像とボケの部分のトークンを作るデータセット．"""

    def __init__(self, groups: list, key: str):
        """
        Args:
            groups (list[tuple]): (お題ID, 形式, [(識別子, ボケの文), ...]) のリスト．
            key (str): モデルのキー．
        """
        self.groups, self.key, self.proc = groups, key, None

    def __len__(self) -> int:
        """お題の数を返す．

        Returns:
            int: お題の数．
        """
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        """1お題分の入力を作る．

        Args:
            i (int): 番号．

        Returns:
            dict: prefix_ids (L,), pixel_values, suffix_ids（リスト），ids（識別子のリスト）．
        """
        if self.proc is None:
            self.proc = M16.load_processor(self.key)
        fid, fmt, items = self.groups[i]
        img = M16.small_image(image_path(fid))
        texts = [self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                                                             {"type": "text", "text": M16.build_prompt(fmt, t)}]}],
                                               add_generation_prompt=True, tokenize=False) for _, t in items]
        cut = texts[0].index(SPLIT_MARK) + len(SPLIT_MARK)
        extra = {"crop_to_patches": False} if self.key.startswith("iv") else {}
        pre = self.proc(text=[texts[0][:cut]], images=[[img]], return_tensors="pt", **extra)
        suf = [self.proc.tokenizer(t[cut:], add_special_tokens=False)["input_ids"] for t in texts]
        out = {"prefix_ids": pre["input_ids"][0], "pixel_values": pre["pixel_values"], "suffix_ids": suf, "ids": [k for k, _ in items]}
        if "image_grid_thw" in pre:
            out["image_grid_thw"], out["mm_token_type_ids"] = pre["image_grid_thw"], pre["mm_token_type_ids"][0]
        return out


def collate(batch: list) -> list:
    """お題のリストをそのまま返す（推論の関数でプレフィックスの長さごとにまとめる）．

    Args:
        batch (list[dict]): GroupDataset の要素．

    Returns:
        list[dict]: 同じリスト．
    """
    return batch


def suffix_batch(sufs: list, device) -> tuple:
    """ボケの部分のトークンを右詰めでまとめる．

    Args:
        sufs (list[list[int]]): ボケごとのトークン．
        device (torch.device): GPU．

    Returns:
        tuple[torch.Tensor, torch.Tensor]: トークン (B, S) とマスク (B, S)．
    """
    slen = max(len(s) for s in sufs)
    suf = torch.zeros((len(sufs), slen), dtype=torch.long, device=device)
    smask = torch.zeros_like(suf)
    for i, s in enumerate(sufs):
        suf[i, :len(s)] = torch.tensor(s, device=device)
        smask[i, :len(s)] = 1
    return suf, smask


def heads(model, h: torch.Tensor) -> tuple:
    """隠れ状態から確率を求める．

    Args:
        model (VLMClassifier): 分類器．
        h (torch.Tensor): 形状 (B, H) の隠れ状態．

    Returns:
        tuple[np.ndarray, np.ndarray]: 発話の型の確率と画像外の知識の確率．
    """
    h = h.float()
    return torch.softmax(model.voice_head(h), 1).cpu().numpy(), torch.sigmoid(model.outside_head(h)[:, 0]).cpu().numpy()


@torch.no_grad()
def predict_groups_qwen(model, groups: list, device) -> tuple:
    """Qwen3-VL について，プレフィックスを左詰めでまとめてキャッシュを作り，ボケの部分を計算する．

    Args:
        model (VLMClassifier): 分類器．
        groups (list[dict]): GroupDataset の要素のリスト．
        device (torch.device): GPU．

    Returns:
        tuple[list, np.ndarray, np.ndarray]: 識別子，発話の型の確率，画像外の知識の確率．
    """
    from transformers import DynamicCache

    plen = max(len(g["prefix_ids"]) for g in groups)
    n = len(groups)
    pre = torch.zeros((n, plen), dtype=torch.long, device=device)
    pmask = torch.zeros_like(pre)
    mm = torch.zeros_like(pre)
    for i, g in enumerate(groups):
        L = len(g["prefix_ids"])
        pre[i, plen - L:] = g["prefix_ids"].to(device)
        pmask[i, plen - L:] = 1
        mm[i, plen - L:] = g["mm_token_type_ids"].to(device)
    pix = torch.cat([g["pixel_values"] for g in groups]).to(device, torch.bfloat16)
    thw = torch.cat([g["image_grid_thw"] for g in groups]).to(device)
    qm = model.backbone.base_model.model
    qm.rope_deltas = None
    pos, _ = qm.get_rope_index(pre, image_grid_thw=thw, attention_mask=pmask, mm_token_type_ids=mm)
    start = pos.max(dim=2).values.max(dim=0).values + 1
    cache = DynamicCache()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.backbone(input_ids=pre, attention_mask=pmask, pixel_values=pix, image_grid_thw=thw, mm_token_type_ids=mm,
                       position_ids=pos, past_key_values=cache, use_cache=True)
    rep = torch.tensor([i for i, g in enumerate(groups) for _ in g["suffix_ids"]], device=device)
    cache.reorder_cache(rep)
    sufs = [s for g in groups for s in g["suffix_ids"]]
    suf, smask = suffix_batch(sufs, device)
    spos = (start[rep][:, None] + torch.arange(suf.shape[1], device=device)[None, :]).unsqueeze(0).expand(3, -1, -1)
    mask = torch.cat([pmask[rep], smask], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.backbone(input_ids=suf, attention_mask=mask, position_ids=spos, past_key_values=cache, use_cache=True)
    last = torch.tensor([len(s) - 1 for s in sufs], device=device)
    pv, po = heads(model, out.last_hidden_state[torch.arange(len(sufs), device=device), last])
    qm.rope_deltas = None
    return [k for g in groups for k in g["ids"]], pv, po


@torch.no_grad()
def predict_groups(model, groups: list, device) -> tuple:
    """お題のまとまりについて，プレフィックスのキャッシュを用いて確率を求める．

    Args:
        model (VLMClassifier): 分類器（ex016 model.py）．
        groups (list[dict]): GroupDataset の要素のリスト．
        device (torch.device): GPU．

    Returns:
        tuple[list, np.ndarray, np.ndarray]: 識別子，発話の型の確率 (N, 6)，画像外の知識の確率 (N,)．
    """
    from transformers import DynamicCache

    if "image_grid_thw" in groups[0]:
        return predict_groups_qwen(model, groups, device)
    ids, pvs, pos = [], [], []
    by_len = {}
    for g in groups:
        by_len.setdefault(len(g["prefix_ids"]), []).append(g)
    for plen, gs in by_len.items():
        pre_ids = torch.stack([g["prefix_ids"] for g in gs]).to(device)
        pix = torch.cat([g["pixel_values"] for g in gs]).to(device, torch.bfloat16)
        cache = DynamicCache()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.backbone(input_ids=pre_ids, pixel_values=pix, attention_mask=torch.ones_like(pre_ids), past_key_values=cache, use_cache=True)
        rep = torch.tensor([i for i, g in enumerate(gs) for _ in g["suffix_ids"]], device=device)
        cache.reorder_cache(rep)
        sufs = [s for g in gs for s in g["suffix_ids"]]
        slen = max(len(s) for s in sufs)
        suf = torch.zeros((len(sufs), slen), dtype=torch.long, device=device)
        smask = torch.zeros_like(suf)
        for i, s in enumerate(sufs):
            suf[i, :len(s)] = torch.tensor(s, device=device)
            smask[i, :len(s)] = 1
        mask = torch.cat([torch.ones((len(sufs), plen), dtype=torch.long, device=device), smask], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.backbone(input_ids=suf, attention_mask=mask, past_key_values=cache, use_cache=True)
        last = torch.tensor([len(s) - 1 for s in sufs], device=device)
        h = out.last_hidden_state[torch.arange(len(sufs), device=device), last].float()
        pvs.append(torch.softmax(model.voice_head(h), 1).cpu().numpy())
        pos.append(torch.sigmoid(model.outside_head(h)[:, 0]).cpu().numpy())
        ids += [k for g in gs for k in g["ids"]]
    return ids, np.concatenate(pvs), np.concatenate(pos)


def run_fast(model, key: str, groups: list, bokes_per_batch: int = 512, num_workers: int = 8) -> tuple:
    """お題のリスト全体を高速化した推論で処理する．

    Args:
        model (VLMClassifier): 分類器．
        key (str): モデルのキー．
        groups (list[tuple]): (お題ID, 形式, [(識別子, ボケ), ...]) のリスト．
        bokes_per_batch (int): 1回の推論でまとめるボケの数の目安．
        num_workers (int): 前処理のプロセス数．

    Returns:
        tuple[list, np.ndarray, np.ndarray]: 識別子，発話の型の確率，画像外の知識の確率．
    """
    loader = torch.utils.data.DataLoader(GroupDataset(groups, key), batch_size=max(1, bokes_per_batch // 14), num_workers=num_workers,
                                         collate_fn=collate, prefetch_factor=4)
    ids, pvs, pos = [], [], []
    for batch in tqdm(loader, leave=False):
        i, v, o = predict_groups(model, batch, "cuda")
        ids += i
        pvs.append(v)
        pos.append(o)
    return ids, np.concatenate(pvs), np.concatenate(pos)


def check(model, key: str, ev: pl.DataFrame, ref_path: str) -> dict:
    """検証データで，高速化した推論と通常の推論（保存済みの eval.csv）の一致を確かめる．

    Args:
        model (VLMClassifier): 分類器．
        key (str): モデルのキー．
        ev (pl.DataFrame): 検証データ．
        ref_path (str): 通常の推論の予測（eval.csv）．

    Returns:
        dict: 確率の差の最大値・平均値，判定の一致率．
    """
    groups = [(int(k[0]), k[1], list(zip(g["no"].to_list(), g["text"].to_list())))
              for k, g in ev.group_by(["odaiId", "format"], maintain_order=True)]
    ids, pv, po = run_fast(model, key, groups)
    fast = pl.DataFrame({"no": ids, "fo": po, "fv": np.array(VOICES)[pv.argmax(1)]})
    ref = pl.read_csv(ref_path).join(fast, on="no")
    d = np.abs(ref["p_outside"].to_numpy() - ref["fo"].to_numpy())
    return {"n": ref.height, "p_outside_max_abs_diff": float(d.max()), "p_outside_mean_abs_diff": float(d.mean()),
            "outside_agree": float(((ref["p_outside"] > 0.5) == (ref["fo"] > 0.5)).mean()), "voice_agree": float((ref["voice"] == ref["fv"]).mean())}


def bench(model, key: str, n_odai: int = 600, skip_standard: bool = False) -> dict:
    """実データのお題の全てのボケで，通常の推論と高速化した推論の速度を測る．

    Args:
        model (VLMClassifier): 分類器．
        key (str): モデルのキー．
        n_odai (int): お題の数．
        skip_standard (bool): 通常の推論の測定を省くか．

    Returns:
        dict: 方法ごとの件数/秒と，全件（9,250万件）の推定日数．
    """
    lab = D16.load_odai_labels().select("odaiId", "format").filter(pl.col("format").is_not_null())
    ids = lab.sample(n=n_odai * 3, seed=0)["odaiId"].to_list()
    b = D16.D15.load_bokes(ids, ("odaiId", "text")).filter(pl.col("text").fill_null("") != "").join(lab, on="odaiId")
    keep = b.group_by("odaiId").len().filter(pl.col("len") <= 200).sample(n=n_odai, seed=0)["odaiId"]
    b = b.filter(pl.col("odaiId").is_in(keep)).with_row_index("i")
    res = {"n_odai": n_odai, "n_boke": b.height, "bokes_per_odai": b.height / n_odai}
    df = b.with_columns(pl.lit("S").alias("voice"), pl.lit(False).alias("outside"))
    for bs in [32, 64]:
        if skip_standard:
            break
        loader = T16.make_loader(df, key, bs, False, 0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        T16.predict(model, loader)
        torch.cuda.synchronize()
        res[f"standard_bs{bs}_per_sec"] = b.height / (time.perf_counter() - start)
    groups = [(int(k[0]), k[1], list(zip(g["i"].to_list(), g["text"].to_list()))) for k, g in b.group_by(["odaiId", "format"])]
    for nb, nw in [(256, 8), (512, 8), (1024, 8), (256, 16), (512, 16)]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_fast(model, key, groups, nb, nw)
        torch.cuda.synchronize()
        res[f"fast_{nb}_w{nw}_per_sec"] = b.height / (time.perf_counter() - start)
    res["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    for k in [k for k in res if k.endswith("_per_sec")]:
        res[k.replace("_per_sec", "_days_for_all")] = 92_500_000 / res[k] / 86400
    return res


def main() -> None:
    """一致の確認と速度の測定を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="iv2")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--ref", default=None, help="通常の推論の eval.csv（--check 用．既定は weights と同じディレクトリ）")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    model = M16.load_model(args.model, args.weights, seed=0, gradient_checkpointing=False).cuda().eval()
    out = os.path.join(OUTPUT_DIR, "fast")
    os.makedirs(out, exist_ok=True)
    if args.check:
        r = check(model, args.model, load_eval(), args.ref or os.path.join(os.path.dirname(args.weights), "eval.csv"))
        json.dump(r, open(os.path.join(out, f"check{args.tag}.json"), "w"), indent=1)
        print("check", r, flush=True)
    if args.bench:
        r = bench(model, args.model)
        json.dump(r, open(os.path.join(out, f"bench{args.tag}.json"), "w"), indent=1)
        print("bench", r, flush=True)


if __name__ == "__main__":
    main()
