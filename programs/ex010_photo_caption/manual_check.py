"""採点者（Qwen3-VL-32B）の妥当性を確かめるため，Claude Code が一部のキャプションを目視で採点する．

最終評価用データの先頭40枚について，4つの方法のキャプションを，方法が分からないよう画像ごとに無作為な順で A〜D として
画像の下に並べた一覧を作る（--make）．目視の採点は manual_scores.txt に「番号 A B C D」（各1〜5）で記録する．
--analyze で方法ごとの目視の点と，採点者の点との一致（相関，±1以内の一致率）を求める．

実行例:
    .venv_annotate/bin/python programs/ex010_photo_caption/manual_check.py --make
    .venv_annotate/bin/python programs/ex010_photo_caption/manual_check.py --analyze
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import plot_image_grid  # noqa: E402
from data import DATASET_DIR, OUTPUT_DIR, THIS_DIR, eval_ids  # noqa: E402

METHODS = ["q8_448_ja", "q8_448_short", "q8_448_en", "q2_448_ja"]
N = 40
SHEET_DIR = os.path.join(DATASET_DIR, "manual_sheets")
KEY = os.path.join(DATASET_DIR, "manual_key.json")
SCORES = os.path.join(THIS_DIR, "manual_scores.txt")


def captions(method: str) -> dict:
    """方法の評価用データのキャプションを読み込む．

    Args:
        method (str): 方法の名前．

    Returns:
        dict[int, str]: お題ID → キャプション．
    """
    df = pl.read_csv(os.path.join(OUTPUT_DIR, method, "captions_eval.csv"))
    return dict(zip(df["fileId"].to_list(), df["caption"].fill_null("").to_list()))


def make() -> None:
    """目視の採点用の一覧と，記号と方法の対応（manual_key.json）を保存する．

    Returns:
        None
    """
    ids = eval_ids("test")[:N]
    caps = {m: captions(m) for m in METHODS}
    rng = np.random.default_rng(0)
    key, titles = [], []
    for k, i in enumerate(ids):
        order = rng.permutation(len(METHODS)).tolist()
        key.append([METHODS[j] for j in order])
        titles.append(f"{k:03d}\n" + "\n".join(f"{c}: {caps[METHODS[j]][i]}" for c, j in zip("ABCD", order)))
    os.makedirs(SHEET_DIR, exist_ok=True)
    json.dump({"ids": ids, "key": key}, open(KEY, "w"), ensure_ascii=False)
    for s in range(0, N, 8):
        plot_image_grid(ids[s:s + 8], titles[s:s + 8], f"目視の採点 {s // 8}", os.path.join(SHEET_DIR, f"sheet_{s // 8:02d}.png"),
                        ncols=4, size=400, wrap=34)


def analyze() -> dict:
    """目視の点を方法ごとに集計し，採点者の点と比べる．

    Returns:
        dict: 方法ごとの目視の平均点と，採点者との一致の指標．
    """
    k = json.load(open(KEY))
    rows = []
    for line in open(SCORES, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            n, *sc = line.split()
            for m, s in zip(k["key"][int(n)], sc):
                rows.append({"fileId": k["ids"][int(n)], "method": m, "manual": int(s)})
    df = pl.DataFrame(rows)
    judge = pl.concat([pl.read_csv(os.path.join(OUTPUT_DIR, m, "judge_eval.csv")).with_columns(pl.lit(m).alias("method"))
                       for m in METHODS]).select("fileId", "method", pl.col("score").alias("judge"))
    df = df.join(judge, on=["fileId", "method"])
    out = {"per_method": {m: {"manual": float(g["manual"].mean()), "judge": float(g["judge"].mean()),
                              "manual_ok": float((g["manual"] >= 4).mean())}
                          for (m,), g in df.group_by("method")},
           "pearson": float(np.corrcoef(df["manual"], df["judge"])[0, 1]),
           "within1": float(((df["manual"] - df["judge"]).abs() <= 1).mean()),
           "ok_agreement": float(((df["manual"] >= 4) == (df["judge"] >= 4)).mean()), "n": df.height}
    json.dump(out, open(os.path.join(OUTPUT_DIR, "manual_check.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main() -> None:
    """コマンドライン引数に従って実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.make:
        make()
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()
