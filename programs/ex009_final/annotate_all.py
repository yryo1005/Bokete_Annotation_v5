"""BoketeDBv5 のアノテーション（最終版）：BoketeDBv4 の全てのお題画像に，定義書 v4 の形式・属性・媒体を付ける．

最終の構成は outputs/ex009_final/final_config.json に記録する（compare.py の比較に基づいて決めた）．処理は次の段階からなり，
各段階は途中の結果をファイルに保存するため，中断しても同じコマンドで再開できる．

1. list：BoketeDBv4 の image の表から全ての画像を列挙する．sha256 が同じ画像は1回だけ予測し（代表のお題ID），結果を共有する．
2. predict：代表の画像の形式・属性・媒体の確率を，final_config.json の models の確率の平均として求める．
   - mlp：SigLIP2-so400m の埋め込み（学習時と同じ前処理，float16，L2 正規化）に対する MLP（複数の seed の確率を平均）
   - finetune：微調整した SigLIP2-B
   画像は1回だけ読み込み，両方のモデルの入力を作る．SHARD 枚ごとに predict/part_XXXX.parquet に保存し，
   埋め込みも embeddings/part_XXXX.npy（float16）に保存する（後で生徒を学習し直す場合に再計算を省くため）．
3. veto：形式をカルタと予測したお題について，ボケの書き出しのかなの統計を求める（veto.parquet）．
   最終版の後処理（verify_noveto）ではカルタの取り消しを用いないため，この段階は省略される．
4. verify：お題文の確認の質問（final_config.json の verify_component，Qwen3-VL-8B）の対象となる代表の画像に質問を行う（verify/part_XXXX.parquet）．
   対象は postprocess.verify_targets で決める（お題文と予測した画像と，通常と予測しお題文の確率がしきい値以上の画像）．
5. finalize：後処理（postprocess.decide_format）を適用し，お題ごとのラベルを annotation.parquet に保存する．
   ラベルの分布と処理の記録を summary.json に保存する．

出力の列（annotation.parquet）：
- odaiId，sha256，rep_odaiId（予測に用いた代表のお題ID）
- format（karuta / anaume / odaibun / normal），has_blank，has_karuta_mark，has_text，
  photo，anime，game，manga，text_graphic，painting，illustration（bool）
- p_format_{形式}，p_{属性}，p_{媒体}（モデルの確率），karuta_vetoed，verify_asked，verify_answer（確認の質問の記録）

実行例:
    # 少量での動作確認（無作為の N 枚．出力先を分ける）
    .venv_annotate/bin/python programs/ex009_final/annotate_all.py --sample 3000 --out_dir outputs/ex009_final/annotation_test
    # 全データ
    .venv_annotate/bin/python programs/ex009_final/annotate_all.py --out_dir datasets/ex009_final/BoketeDBv5_annotation
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR  # noqa: E402
from data import ATTRIBUTES, EX5_DIR, FORMATS, MEDIA, PROJECT_DIR, _load  # noqa: E402
import postprocess as PP  # noqa: E402

DB_DIR = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4"
CONFIG_PATH = os.path.join(OUTPUT_DIR, "final_config.json")
SHARD = 200_000
VERIFY_CHUNK = 20_000


def list_images(out_dir: str, sample: int = None) -> pl.DataFrame:
    """全ての画像を列挙し，sha256 ごとの代表のお題IDを決める．

    Args:
        out_dir (str): 出力先のディレクトリ．
        sample (int): 動作確認用に無作為に抽出する枚数（None で全て）．

    Returns:
        pl.DataFrame: 列 odaiId, sha256, rep_odaiId を持つ表．
    """
    path = os.path.join(out_dir, "images.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    img = pl.scan_parquet(os.path.join(DB_DIR, "image", "*.parquet")).select("odaiId", "sha256", "readable").collect()
    img = img.filter(pl.col("readable")).drop("readable")
    if sample:
        img = img.sample(n=sample, seed=0)
    img = img.with_columns(pl.col("odaiId").min().over("sha256").alias("rep_odaiId")).sort("odaiId")
    img.write_parquet(path)
    return img


class _TwoInputDataset(torch.utils.data.Dataset):
    """画像を1回だけ読み込み，埋め込みのモデルの入力と微調整したモデルの入力を返すデータセット．"""

    def __init__(self, ids: list, processor, transform):
        """
        Args:
            ids (list[int]): お題IDのリスト．
            processor: SigLIP2-so400m の画像の前処理器（学習時の埋め込みと同じもの）．
            transform (callable): 微調整したモデルの予測用の変換．
        """
        self.ids, self.processor, self.transform = ids, processor, transform

    def __len__(self) -> int:
        """データ数を返す．

        Returns:
            int: 画像の数．
        """
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple:
        """i 番目の画像の2つの入力を返す．

        Args:
            i (int): インデックス．

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 埋め込みのモデルの入力（float16，(3, 384, 384)）と，微調整したモデルの入力．
        """
        from PIL import Image

        from data import image_path

        with Image.open(image_path(self.ids[i])) as im:
            im = im.convert("RGB")
            return (self.processor(images=im, return_tensors="pt")["pixel_values"][0].half(), self.transform(im))


def load_models(cfg: dict) -> dict:
    """final_config.json の models を読み込む．

    Args:
        cfg (dict): 最終の構成．

    Returns:
        dict: キー embed（SigLIP2-so400m のモデルと前処理器），mlps（MLP のリスト），finetunes（微調整したモデルのリスト），
        transform（微調整したモデルの予測用の変換）．
    """
    from transformers import AutoModel, AutoProcessor

    TR = _load("ex009_train_for_annotate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
    out = {"mlps": [], "finetunes": [], "transform": None, "embed": None}
    for spec in cfg["models"]:
        if spec["type"] == "mlp":
            name = spec["embedding"]
            out["embed"] = (AutoModel.from_pretrained(name, dtype=torch.float16).cuda().eval(),
                            AutoProcessor.from_pretrained(name).image_processor)
            for path in spec["paths"]:
                m = TR.ST.ThreeHeadMLP(spec["dim"]).cuda().eval()
                m.load_state_dict(torch.load(os.path.join(PROJECT_DIR, path), map_location="cuda"))
                out["mlps"].append(m)
        else:
            for path in spec["paths"]:
                m = TR.load_model(spec["model"], os.path.join(PROJECT_DIR, path)).cuda().eval()
                out["finetunes"].append(m.to(memory_format=torch.channels_last))
                out["transform"] = m.transforms()[1]
    return out


@torch.no_grad()
def predict_shard(models: dict, ids: list, num_workers: int) -> tuple:
    """1つのシャードの画像の確率と埋め込みを求める．

    Args:
        models (dict): load_models の戻り値．
        ids (list[int]): お題IDのリスト．
        num_workers (int): 画像の読み込みのプロセス数．

    Returns:
        tuple[dict, np.ndarray]: 出力層ごとの確率（モデルの種類ごとに平均してから平均）と，形状 (N, 1152) の埋め込み．
    """
    TR = _load("ex009_train_for_annotate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
    loader = torch.utils.data.DataLoader(_TwoInputDataset(ids, models["embed"][1], models["transform"]), batch_size=256,
                                         num_workers=num_workers, pin_memory=True, prefetch_factor=4)
    heads = ["format", "attr", "media"]
    acc = {"mlp": {h: [] for h in heads}, "ft": {h: [] for h in heads}}
    embs = []
    for x_emb, x_ft in tqdm(loader, desc="batch", leave=False):
        f = models["embed"][0].get_image_features(pixel_values=x_emb.cuda(non_blocking=True))
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        e = torch.nn.functional.normalize(f.float(), dim=-1).half()
        embs.append(e.cpu().numpy())
        ps = [TR.ST.probs_of(m(e.float())) for m in models["mlps"]]
        for h in heads:
            acc["mlp"][h].append(np.mean([p[h] for p in ps], axis=0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = [m(x_ft.cuda(non_blocking=True).to(memory_format=torch.channels_last)) for m in models["finetunes"]]
        ft = {"format": np.mean([torch.softmax(o["format"].float(), 1).cpu().numpy() for o in outs], axis=0)}
        for h in ["attr", "media"]:
            ft[h] = np.mean([torch.sigmoid(o[h].float()).cpu().numpy() for o in outs], axis=0)
        for h in heads:
            acc["ft"][h].append(ft[h])
    probs = {h: (np.concatenate(acc["mlp"][h]) + np.concatenate(acc["ft"][h])) / 2 for h in heads}
    return probs, np.concatenate(embs)


def predict(out_dir: str, reps: list, cfg: dict) -> None:
    """代表の画像の確率と埋め込みを求め，SHARD 枚ごとに保存する．

    Args:
        out_dir (str): 出力先のディレクトリ．
        reps (list[int]): 代表のお題IDのリスト（昇順）．
        cfg (dict): 最終の構成．

    Returns:
        None
    """
    d, de = os.path.join(out_dir, "predict"), os.path.join(out_dir, "embeddings")
    os.makedirs(d, exist_ok=True)
    os.makedirs(de, exist_ok=True)
    todo = [k for k in range(0, len(reps), SHARD) if not os.path.exists(os.path.join(d, f"part_{k // SHARD:04d}.parquet"))]
    if not todo:
        return
    models = load_models(cfg)
    for k in tqdm(todo, desc="predict", leave=False):
        ids = reps[k:k + SHARD]
        start = time.perf_counter()
        p, emb = predict_shard(models, ids, cfg.get("num_workers", 16))
        cols = {"rep_odaiId": ids, **{f"p_format_{c}": p["format"][:, i] for i, c in enumerate(FORMATS)},
                **{f"p_{c}": p["attr"][:, i] for i, c in enumerate(ATTRIBUTES)},
                **{f"p_{c}": p["media"][:, i] for i, c in enumerate(MEDIA)}}
        np.save(os.path.join(de, f"part_{k // SHARD:04d}.npy"), emb)
        pl.DataFrame(cols).write_parquet(os.path.join(d, f"part_{k // SHARD:04d}.parquet"))
        log_speed(out_dir, "predict", len(ids), time.perf_counter() - start)


def log_speed(out_dir: str, stage: str, n: int, sec: float) -> None:
    """段階ごとの処理枚数と秒数を記録する．

    Args:
        out_dir (str): 出力先のディレクトリ．
        stage (str): 段階の名前．
        n (int): 処理した枚数．
        sec (float): 秒数．

    Returns:
        None
    """
    path = os.path.join(out_dir, "speed.json")
    log = json.load(open(path)) if os.path.exists(path) else {}
    s = log.setdefault(stage, {"n": 0, "sec": 0.0})
    s["n"] += n
    s["sec"] += sec
    s["images_per_sec"] = s["n"] / s["sec"]
    json.dump(log, open(path, "w"), indent=2)


def load_predictions(out_dir: str) -> pl.DataFrame:
    """predict の結果を全て読み込む．

    Args:
        out_dir (str): 出力先のディレクトリ．

    Returns:
        pl.DataFrame: 列 rep_odaiId と確率を持つ表．
    """
    return pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(out_dir, "predict", "part_*.parquet")))])


def format_probs(df: pl.DataFrame) -> np.ndarray:
    """表から形式の確率の配列を取り出す．

    Args:
        df (pl.DataFrame): 列 p_format_{形式} を持つ表．

    Returns:
        np.ndarray: 形状 (N, 4) の確率．
    """
    return df.select([f"p_format_{c}" for c in FORMATS]).to_numpy()


def kana_stats(odai_ids: list) -> pl.DataFrame:
    """お題ごとに，かなで始まるボケの書き出しの数え上げを求める（ボケの表をシャードごとに読む）．

    Args:
        odai_ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 odaiId, n_kana, kana_counts を持つ表．
    """
    RULES = _load("ex004_rules_for_annotate", os.path.join(os.path.dirname(EX5_DIR), "ex004_four_axis_distill", "rules.py"))
    ids = pl.DataFrame({"odaiId": odai_ids}).with_columns((pl.col("odaiId") // 10000 + 1).alias("shard"))
    rows = []
    for (shard,), g in tqdm(ids.group_by("shard"), desc="veto", leave=False):
        path = os.path.join(DB_DIR, "boke", f"boke_{shard}.parquet")
        if not os.path.exists(path):
            continue
        b = pl.scan_parquet(path).filter(pl.col("odaiId").is_in(g["odaiId"].to_list())).select("odaiId", "text").collect()
        b = b.with_columns(pl.col("text").fill_null("").map_elements(RULES.first_char, return_dtype=pl.String).alias("c"))
        b = b.filter((pl.col("c") >= "ぁ") & (pl.col("c") <= "ゖ"))
        cnt = b.group_by("odaiId", "c").len()
        agg = cnt.group_by("odaiId").agg(pl.col("len").sum().alias("n_kana"),
                                          pl.concat_str([pl.col("c"), pl.lit(":"), pl.col("len").cast(pl.String)]).str.join(";")
                                          .alias("kana_counts"))
        rows.append(agg)
    out = pl.concat(rows) if rows else pl.DataFrame(schema={"odaiId": pl.Int64, "n_kana": pl.UInt32, "kana_counts": pl.String})
    return pl.DataFrame({"odaiId": odai_ids}).join(out, on="odaiId", how="left", maintain_order="left").with_columns(
        pl.col("n_kana").fill_null(0).cast(pl.Int64), pl.col("kana_counts").fill_null(""))


def veto(out_dir: str, img: pl.DataFrame) -> None:
    """形式をカルタと予測したお題のボケの書き出しの統計を保存する．

    Args:
        out_dir (str): 出力先のディレクトリ．
        img (pl.DataFrame): list_images の戻り値．

    Returns:
        None
    """
    path = os.path.join(out_dir, "veto.parquet")
    if os.path.exists(path):
        return
    pred = load_predictions(out_dir)
    k_reps = pred.filter(pl.Series(format_probs(pred).argmax(1) == FORMATS.index("karuta")))["rep_odaiId"]
    odai = img.filter(pl.col("rep_odaiId").is_in(k_reps.to_list()))["odaiId"].to_list()
    start = time.perf_counter()
    kana_stats(odai).write_parquet(path)
    log_speed(out_dir, "veto", len(odai), time.perf_counter() - start)


def verify(out_dir: str, img: pl.DataFrame, cfg: dict) -> None:
    """確認の質問の対象となる代表の画像に Qwen の確認の質問を行う．

    カルタの取り消しはお題ごとに異なりうるため，対象は代表の画像の予測（取り消し前）で決め，取り消しで形式が
    お題文に変わる画像も対象に含める（取り消し後にお題文となる可能性がある画像は，お題文の確率が2番目に高い画像）．

    Args:
        out_dir (str): 出力先のディレクトリ．
        img (pl.DataFrame): list_images の戻り値．
        cfg (dict): 最終の構成．

    Returns:
        None
    """
    d = os.path.join(out_dir, "verify")
    os.makedirs(d, exist_ok=True)
    pred = load_predictions(out_dir)
    pf = format_probs(pred)
    fmt = pf.argmax(1)
    alt = pf.copy()
    alt[:, FORMATS.index("karuta")] = -1
    targets = PP.verify_targets(pf, fmt, cfg["post"]) | (
        (fmt == FORMATS.index("karuta")) & (alt.argmax(1) == FORMATS.index("odaibun")))
    ids = pred.filter(pl.Series(targets))["rep_odaiId"].to_list()
    todo = [k for k in range(0, len(ids), VERIFY_CHUNK) if not os.path.exists(os.path.join(d, f"part_{k // VERIFY_CHUNK:04d}.parquet"))]
    if not todo:
        return
    T5 = _load("ex005_teacher_for_annotate", os.path.join(EX5_DIR, "teacher.py"))
    comp = cfg["verify_component"]
    llm = T5.build_llm(T5.COMPONENTS[comp]["max_pixels"])
    for k in tqdm(todo, desc="verify", leave=False):
        df, sec = T5.run_component(llm, comp, ids[k:k + VERIFY_CHUNK], comp)
        df.select(pl.col("fileId").alias("rep_odaiId"), pl.col("is_poster_prompt").alias("verify_answer")).write_parquet(
            os.path.join(d, f"part_{k // VERIFY_CHUNK:04d}.parquet"))
        log_speed(out_dir, "verify", len(df), sec)


def finalize(out_dir: str, img: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """後処理を適用し，お題ごとのラベルを保存する．

    Args:
        out_dir (str): 出力先のディレクトリ．
        img (pl.DataFrame): list_images の戻り値．
        cfg (dict): 最終の構成．

    Returns:
        pl.DataFrame: お題ごとのラベルの表．
    """
    TR = _load("ex009_train_for_annotate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
    pred = load_predictions(out_dir)
    ver_paths = sorted(glob.glob(os.path.join(out_dir, "verify", "part_*.parquet")))
    ver = pl.concat([pl.read_parquet(p) for p in ver_paths]) if ver_paths else pl.DataFrame(
        schema={"rep_odaiId": pl.Int64, "verify_answer": pl.Boolean})
    vt_path = os.path.join(out_dir, "veto.parquet")
    vt = pl.read_parquet(vt_path) if os.path.exists(vt_path) else pl.DataFrame(
        schema={"odaiId": pl.Int64, "n_kana": pl.Int64, "kana_counts": pl.String})
    df = img.join(pred, on="rep_odaiId", how="left").join(ver.unique("rep_odaiId"), on="rep_odaiId", how="left").join(
        vt, on="odaiId", how="left")
    pf = format_probs(df)
    max_rate = PP.kana_max_rate(df["kana_counts"].fill_null("").to_list(), df["n_kana"].fill_null(0).to_list())
    answer = df["verify_answer"].fill_null(True).to_numpy().astype(bool)
    fmt, vetoed, asked = PP.decide_format(pf, df["n_kana"].fill_null(0).to_numpy(), max_rate, answer, cfg["post"])
    missing = asked & df["verify_answer"].is_null().to_numpy()
    if missing.any():
        raise RuntimeError(f"確認の質問の回答がない画像が {int(missing.sum())} 枚ある．verify の段階を実行すること")
    onehot = np.zeros_like(pf)
    onehot[np.arange(len(fmt)), fmt] = 1.0
    probs = {"format": onehot, "attr": df.select([f"p_{c}" for c in ATTRIBUTES]).to_numpy(),
             "media": df.select([f"p_{c}" for c in MEDIA]).to_numpy()}
    labels = TR.to_frame(probs)
    out = pl.concat([df.select("odaiId", "sha256", "rep_odaiId"), labels,
                     df.select([c for c in df.columns if c.startswith("p_")]),
                     pl.DataFrame({"karuta_vetoed": vetoed, "verify_asked": asked}),
                     df.select(pl.when(pl.Series(asked)).then(pl.col("verify_answer")).alias("verify_answer"))],
                    how="horizontal")
    out.write_parquet(os.path.join(out_dir, "annotation.parquet"))
    summary = {"n_odai": out.height, "n_unique_images": int(img["rep_odaiId"].n_unique()), "config": cfg,
               "format_rate": {f: float((out["format"] == f).mean()) for f in FORMATS},
               "item_rate": {c: float(out[c].mean()) for c in ATTRIBUTES + MEDIA},
               "media_multi_rate": float((out.select(MEDIA).sum_horizontal() >= 2).mean()),
               "verify_asked_rate": float(out["verify_asked"].mean()), "karuta_vetoed": int(out["karuta_vetoed"].sum()),
               "speed": json.load(open(os.path.join(out_dir, "speed.json")))}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, ensure_ascii=False)
    return out


def main() -> None:
    """指定した段階を順に実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample", type=int, default=None, help="動作確認用に無作為に抽出する枚数")
    parser.add_argument("--stages", nargs="+", default=["list", "predict", "veto", "verify", "finalize"])
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_DIR, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(CONFIG_PATH))
    img = list_images(out_dir, args.sample)
    reps = sorted(img["rep_odaiId"].unique().to_list())
    if "predict" in args.stages:
        predict(out_dir, reps, cfg)
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    if "veto" in args.stages and not cfg["post"].endswith("noveto"):
        veto(out_dir, img)
    if "verify" in args.stages:
        verify(out_dir, img, cfg)
    if "finalize" in args.stages:
        out = finalize(out_dir, img, cfg)
        print(json.dumps(json.load(open(os.path.join(out_dir, "summary.json"))), ensure_ascii=False, indent=1)[:2000])
        print(out.head())


if __name__ == "__main__":
    main()
