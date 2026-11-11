"""教師ラベルを付ける学習用のボケ（train_pool.parquet）を抽出する．

ex012 と同じ7層ごとに，検証データ以外のお題から，お題1件につき最大 PER_ODAI 件のボケを無作為に取り，層ごとに N_PER_STRATUM 件とする．
空のボケは除く．お題の形式（BoketeDBv5）を付けて保存する．

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/make_pool.py
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, PROJECT_DIR, load_bokes, load_eval  # noqa: E402

spec = importlib.util.spec_from_file_location("ex012_sample", os.path.join(PROJECT_DIR, "programs", "ex012_boke_scheme", "sample.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
N_PER_STRATUM, PER_ODAI = 5000, 3


def main() -> None:
    """抽出して保存する．

    Returns:
        None
    """
    used = set(load_eval()["odaiId"].to_list())
    ann = pl.scan_parquet(S.ANN).filter(pl.col("odaiId") == pl.col("rep_odaiId")).collect()
    rng = np.random.default_rng(2)
    parts = []
    for st, cond in S.strata().items():
        cand = [int(i) for i in rng.permutation(ann.filter(cond)["odaiId"].to_numpy())[:4000] if int(i) not in used]
        b = load_bokes(cand).filter(pl.col("text").fill_null("").str.strip_chars() != "")
        b = b.with_columns(pl.int_range(pl.len()).shuffle(seed=3).over("odaiId").alias("r")).filter(pl.col("r") < PER_ODAI)
        b = b.sample(n=min(N_PER_STRATUM, b.height), seed=4).with_columns(pl.lit(st).alias("stratum"))
        parts.append(b.drop("r"))
    pool = pl.concat(parts)
    pool = pool.join(ann.select("odaiId", "format"), on="odaiId", how="left")
    pool.write_parquet(os.path.join(DATASET_DIR, "train_pool.parquet"))
    print(pool.height, pool.group_by("stratum").len().rows())


if __name__ == "__main__":
    main()
