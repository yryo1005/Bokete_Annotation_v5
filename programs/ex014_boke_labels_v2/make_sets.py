"""教師ラベル用・検証用の集合を作る．

1. swap_pairs.csv：検証データの穴埋め・お題文のボケを，同じ形式の別のお題（検証データのお題）に付け替えた組（制約の違反の代わり）．
   列 no（元のボケの番号），odaiId（付け替え先），orig_odaiId，text，format．元の組（正）は検証データそのものを用いる．
2. aug_pool.csv：発話の型の水増し用．ex013 の学習用のボケ3.5万件（追加の検証データのお題は除く）から，
   無作為の N_RANDOM 件と，形式が通常かカルタで規則か教師（30B-A3B 第2版）が心の声・ツッコミ・状況の説明・名付け・見立てと予測した
   候補 N_RARE 件（型ごとに均等）．列 pool_index，odaiId，text，format，source（random / rare）．
3. swap_train.csv：制約の順守の分類器の学習用．ex013 の学習用のボケのうち穴埋め・お題文のもの（正）と，それを同じ形式の別のお題に
   付け替えたもの（負）．列 odaiId，text，format，label（1＝元のお題，0＝付け替え）．

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/make_sets.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, EX13_DATASET, EX13_DIR, EX13_OUTPUT, _load, load_eval  # noqa: E402

R13 = _load("ex013_rules_for_v2", os.path.join(EX13_DIR, "rules.py"))
N_RANDOM, N_RARE = 5000, 5000


def swap(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """各ボケのお題を，同じ形式の別のお題に付け替える．

    Args:
        df (pl.DataFrame): 列 odaiId, format を持つ表．
        rng (np.random.Generator): 乱数．

    Returns:
        pl.DataFrame: 列 odaiId を付け替え先にし，列 orig_odaiId を加えた表．
    """
    new = []
    for fid, fmt in df.select("odaiId", "format").rows():
        cand = df.filter((pl.col("format") == fmt) & (pl.col("odaiId") != fid))["odaiId"].unique().to_numpy()
        new.append(int(rng.choice(cand)))
    return df.rename({"odaiId": "orig_odaiId"}).with_columns(pl.Series("odaiId", new))


def main() -> None:
    """3つの集合を作って保存する．

    Returns:
        None
    """
    rng = np.random.default_rng(7)
    ev = load_eval().filter(pl.col("format").is_in(["anaume", "odaibun"])).select("no", "odaiId", "text", "format")
    swap(ev, rng).select("no", "odaiId", "orig_odaiId", "text", "format").write_csv(os.path.join(DATASET_DIR, "swap_pairs.csv"))

    pool = pl.read_parquet(os.path.join(EX13_DATASET, "train_pool.parquet")).with_row_index("pool_index")
    t = pl.read_csv(os.path.join(EX13_OUTPUT, "teacher_q30a3_v2", "train.csv"))
    pool = pl.concat([pool, t.select(pl.col("voice").alias("t_voice"))], how="horizontal")
    held = set(pl.read_csv(os.path.join(DATASET_DIR, "eval2_bokes.csv"))["odaiId"].to_list())
    pool = pool.filter(~pl.col("odaiId").is_in(list(held)))
    pool = pool.with_columns(pl.Series("r_voice", [R13.rule_voice(x, f) for x, f in zip(pool["text"], pool["format"])]))
    rand = pool.sample(n=N_RANDOM, seed=8)
    rest = pool.filter(~pl.col("pool_index").is_in(rand["pool_index"].to_list()) & pl.col("format").is_in(["normal", "karuta"]))
    parts = []
    for v in ["M", "T", "N", "L"]:
        c = rest.filter((pl.col("t_voice") == v) | (pl.col("r_voice") == v))
        c = c.filter(~pl.col("pool_index").is_in([i for p in parts for i in p["pool_index"].to_list()]))
        parts.append(c.sample(n=min(N_RARE // 4, c.height), seed=9))
    aug = pl.concat([rand.with_columns(pl.lit("random").alias("source")), pl.concat(parts).with_columns(pl.lit("rare").alias("source"))])
    aug.select("pool_index", "odaiId", "text", "format", "source").write_csv(os.path.join(DATASET_DIR, "aug_pool.csv"))

    qa = pool.filter(pl.col("format").is_in(["anaume", "odaibun"])).select("odaiId", "text", "format")
    neg = swap(qa, rng).select("odaiId", "text", "format")
    pl.concat([qa.with_columns(pl.lit(1).alias("label")), neg.with_columns(pl.lit(0).alias("label"))]).write_csv(
        os.path.join(DATASET_DIR, "swap_train.csv"))
    print("swap_pairs", ev.height, "aug", aug.height, aug.group_by("source").len().rows(), "swap_train", qa.height * 2)


if __name__ == "__main__":
    main()
