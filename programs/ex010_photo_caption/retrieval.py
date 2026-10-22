"""近傍の検索によるキャプション（VLM を全件に適用しない高速な方法）．

Bokete_Annotation_programs_v5.0 の全データのアノテーションで保存した SigLIP2-so400m の埋め込み（読み込みのみ）を用いる．
キャプションを付けておく画像の集合（pool）を写真から無作為に抽出し，VLM（q8_448_ja）でキャプションを付けておく．
評価用データの各画像に，埋め込みのコサイン類似度が最大の pool の画像のキャプションを付ける．

- --make_pool：pool（評価用・速度の測定用以外の写真から無作為に N_POOL 枚）を datasets/.../pool20k.csv に保存する
- --predict：pool の大きさを変えて（1,000〜20,000枚）評価用データのキャプションを作り，
  outputs/ex010_photo_caption/retrieval_{枚数}/captions_eval.csv に保存する．検索の速度も測る

実行例:
    .venv_annotate/bin/python programs/ex010_photo_caption/retrieval.py --make_pool
    .venv_annotate/bin/python programs/ex010_photo_caption/generate.py --method q8_448_ja --no_speed --extra_set pool20k
    .venv_annotate/bin/python programs/ex010_photo_caption/retrieval.py --predict
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import throughput  # noqa: E402
from data import ANN_DIR, DATASET_DIR, OUTPUT_DIR, eval_ids, load_photo_ids, make_splits  # noqa: E402

N_POOL = 20_000
POOL_SIZES = [1_000, 5_000, 20_000]


def load_embeddings(ids: list) -> np.ndarray:
    """代表の画像の埋め込みを読み込む（predict/part_XXXX.parquet の行と embeddings/part_XXXX.npy の行が対応する）．

    Args:
        ids (list[int]): 代表の画像のお題IDのリスト．

    Returns:
        np.ndarray: 形状 (N, 1152) の float32 の埋め込み（ids の順）．
    """
    want = set(ids)
    rows = {}
    for p in sorted(glob.glob(os.path.join(ANN_DIR, "predict", "part_*.parquet"))):
        rep = pl.read_parquet(p, columns=["rep_odaiId"])["rep_odaiId"].to_numpy()
        idx = np.where(np.isin(rep, list(want)))[0]
        if len(idx):
            emb = np.load(p.replace("predict", "embeddings").replace(".parquet", ".npy"), mmap_mode="r")
            for i in idx:
                rows[int(rep[i])] = np.asarray(emb[i], dtype=np.float32)
    return np.stack([rows[i] for i in ids])


def make_pool(seed: int = 1) -> None:
    """pool の画像を抽出して保存する．

    Args:
        seed (int): 乱数のシード．

    Returns:
        None
    """
    used = set(make_splits()["fileId"].to_list())
    ids = np.array([i for i in load_photo_ids() if i not in used])
    pick = np.random.default_rng(seed).choice(ids, N_POOL, replace=False)
    pl.DataFrame({"fileId": pick.tolist()}).write_csv(os.path.join(DATASET_DIR, "pool20k.csv"))


def predict() -> None:
    """pool の大きさごとに，評価用データに最近傍の画像のキャプションを付ける．

    Returns:
        None
    """
    pool = pl.read_csv(os.path.join(DATASET_DIR, "pool20k.csv"))["fileId"].to_list()
    caps = pl.read_csv(os.path.join(OUTPUT_DIR, "q8_448_ja", "captions_pool20k.csv"))
    cap_of = dict(zip(caps["fileId"].to_list(), caps["caption"].to_list()))
    ev = eval_ids()
    E_pool, E_ev = load_embeddings(pool), load_embeddings(ev)
    for n in POOL_SIZES:
        P = E_pool[:n]
        start = time.perf_counter()
        sim = E_ev @ P.T
        nn = sim.argmax(1)
        sec = time.perf_counter() - start
        out_dir = os.path.join(OUTPUT_DIR, f"retrieval_{n // 1000}k")
        os.makedirs(out_dir, exist_ok=True)
        pl.DataFrame({"fileId": ev, "caption": [cap_of[pool[j]] for j in nn], "neighbor": [pool[j] for j in nn],
                      "similarity": sim.max(1)}).write_csv(os.path.join(out_dir, "captions_eval.csv"))
        # 全写真での速度：埋め込みは保存済みのため，検索のみの時間（行列積）で見積もる
        meta = {"method": f"retrieval_{n // 1000}k", "pool_size": n, "speed_eval": throughput(len(ev), sec),
                "note": "埋め込みの計算（約180枚/秒，v5.0 のアノテーションで計算済み）と pool のキャプション付けの時間は含まない"}
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2, ensure_ascii=False)
        print(n, f"平均類似度 {sim.max(1).mean():.3f}")


def main() -> None:
    """コマンドライン引数に従って実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_pool", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    if args.make_pool:
        make_pool()
    if args.predict:
        predict()


if __name__ == "__main__":
    main()
