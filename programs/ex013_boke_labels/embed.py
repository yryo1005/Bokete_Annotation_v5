"""ボケの文の埋め込みと，お題画像の埋め込み（v5.0 の保存済みのもの．読み込みのみ）を作る．

- 文の埋め込み：cl-nagoya/ruri-v3-310m（日本語），intfloat/multilingual-e5-base（多言語）．平均プーリングと L2 正規化．
  ruri は接頭辞なし，e5 は "query: " を付ける．処理速度も測る．
- お題画像の埋め込み：Bokete_Annotation_programs_v5.0 の embeddings/part_XXXX.npy（SigLIP2-so400m）を，
  rep_odaiId で引いて読む．

出力：datasets/ex013_boke_labels/emb/{集合}_{モデル}.npy，{集合}_image.npy，speed.json

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/embed.py
"""
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
from data import ANN_DIR, DATASET_DIR, load_eval, load_odai_labels  # noqa: E402

TEXT_MODELS = {"ruri310": ("cl-nagoya/ruri-v3-310m", ""), "e5base": ("intfloat/multilingual-e5-base", "query: ")}
EMB_DIR = os.path.join(DATASET_DIR, "emb")


@torch.no_grad()
def embed_texts(name: str, texts: list, batch_size: int = 512) -> tuple:
    """文のリストを埋め込む．

    Args:
        name (str): TEXT_MODELS のキー．
        texts (list[str]): 文のリスト．
        batch_size (int): バッチサイズ．

    Returns:
        tuple[np.ndarray, float]: 形状 (N, D) の float16 の埋め込みと，秒数．
    """
    from transformers import AutoModel, AutoTokenizer

    hf, prefix = TEXT_MODELS[name]
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf, dtype=torch.float16).cuda().eval()
    out = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for s in tqdm(range(0, len(texts), batch_size), desc=name, leave=False):
        batch = [prefix + (t or "").replace("\n", " ")[:200] for t in texts[s:s + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)
        out.append(e.half().cpu().numpy())
    torch.cuda.synchronize()
    sec = time.perf_counter() - start
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out), sec


def image_embeddings(odai_ids: list) -> np.ndarray:
    """お題画像の SigLIP2-so400m の埋め込みを，v5.0 の保存済みの部分から読む．

    Args:
        odai_ids (list[int]): お題IDのリスト（重複可）．

    Returns:
        np.ndarray: 形状 (N, 1152) の float16 の埋め込み（odai_ids の順）．
    """
    rep = dict(load_odai_labels(sorted(set(odai_ids))).select("odaiId", "rep_odaiId").iter_rows())
    want = {rep[i] for i in odai_ids}
    found = {}
    for p in sorted(glob.glob(os.path.join(ANN_DIR, "predict", "part_*.parquet"))):
        ids = pl.read_parquet(p, columns=["rep_odaiId"])["rep_odaiId"].to_numpy()
        idx = np.where(np.isin(ids, list(want)))[0]
        if len(idx):
            emb = np.load(p.replace("predict", "embeddings").replace(".parquet", ".npy"), mmap_mode="r")
            for i in idx:
                found[int(ids[i])] = np.asarray(emb[i])
    return np.stack([found[rep[i]] for i in odai_ids])


def main() -> None:
    """検証データと学習用のボケの埋め込みを作る．

    Returns:
        None
    """
    os.makedirs(EMB_DIR, exist_ok=True)
    sets = {"eval": load_eval().select("odaiId", "text"),
            "train": pl.read_parquet(os.path.join(DATASET_DIR, "train_pool.parquet")).select("odaiId", "text")}
    speed = {}
    for sname, df in sets.items():
        path = os.path.join(EMB_DIR, f"{sname}_image.npy")
        if not os.path.exists(path):
            np.save(path, image_embeddings(df["odaiId"].to_list()))
        for m in TEXT_MODELS:
            path = os.path.join(EMB_DIR, f"{sname}_{m}.npy")
            if os.path.exists(path):
                continue
            e, sec = embed_texts(m, df["text"].to_list())
            np.save(path, e)
            speed[f"{sname}_{m}"] = {"n": df.height, "sec": sec, "per_sec": df.height / sec}
            print(sname, m, e.shape, f"{df.height / sec:.0f} 件/秒", flush=True)
    json.dump(speed, open(os.path.join(EMB_DIR, "speed.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
