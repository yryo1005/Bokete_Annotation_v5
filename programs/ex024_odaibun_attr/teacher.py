"""生徒を学習し直すための，お題文ありの教師ラベルを Qwen で作る．

対象は，現在の生徒の教師ラベル（T_v11）を付けた4.5万枚（無作為抽出4万枚と少ない区分の約5千枚）である．
パッケージの生徒の文字ありの確率が TEXT_GATE 未満の画像は，お題文なし（質問しない）とする．
2,048枚ずつ質問して保存し，中断しても続きから再開できる．

出力：outputs/ex024_odaibun_attr/teacher/{モデル}__{画素}__{指示}/part_XXXX.csv（列 fileId, asked, answer）

実行例:
    .venv_annotate/bin/python programs/ex024_odaibun_attr/teacher.py --model Qwen/Qwen3-VL-32B-Instruct-FP8 --prompt P3
"""
import argparse
import os
import sys

import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ANNOTATION, OUTPUT_DIR, PROJECT_DIR  # noqa: E402
from qwen_odaibun import run, verify  # noqa: E402

TEACHER_DIR = os.path.join(PROJECT_DIR, "datasets", "ex005_rare_weighted_distill", "teacher_labels")
TEXT_GATE = 0.3
CHUNK = 2048


def teacher_pool() -> pl.DataFrame:
    """教師ラベルを付ける画像と，質問するかを返す．

    Returns:
        pl.DataFrame: 列 fileId, src（pool / rare）, ask（bool）を持つ表．
    """
    ids = pl.concat([pl.read_csv(os.path.join(TEACHER_DIR, f"T_v11_{k}.csv")).select("fileId", pl.lit(k).alias("src"))
                     for k in ["pool", "rare"]])
    a = pl.scan_parquet(ANNOTATION).select(pl.col("odaiId").alias("fileId"), "p_has_text").collect()
    d = ids.join(a, on="fileId", how="left")
    return d.select("fileId", "src", (pl.col("p_has_text") >= TEXT_GATE).alias("ask")).sort("fileId")


def main() -> None:
    """教師ラベルを作って保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct-FP8")
    parser.add_argument("--prompt", default="P3")
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()
    out = os.path.join(OUTPUT_DIR, "teacher", f"{args.model.split('/')[-1]}__{args.max_pixels}__{args.prompt}")
    os.makedirs(out, exist_ok=True)
    pool = teacher_pool()
    ask = pool.filter(pl.col("ask"))["fileId"].to_list()
    chunks = [(k // CHUNK, ask[k:k + CHUNK]) for k in range(0, len(ask), CHUNK)]
    todo = [(i, c) for i, c in chunks if not os.path.exists(os.path.join(out, f"part_{i:04d}.csv"))]
    pool.filter(~pl.col("ask")).select("fileId", pl.lit(False).alias("asked"), pl.lit(False).alias("answer")).write_csv(
        os.path.join(out, "not_asked.csv"))
    if not todo:
        print("skip: all done")
        return
    llm = verify.build_llm(args.model, args.max_pixels)
    for i, c in tqdm(todo, desc="teacher", leave=False):
        df, sec, _ = run(llm, args.prompt, c, args.max_pixels, args.max_tokens, "Thinking" in args.model)
        df.select("fileId", pl.lit(True).alias("asked"), pl.col("answer").fill_null(False)).write_csv(
            os.path.join(out, f"part_{i:04d}.csv"))
        print(f"part {i}: {len(c)} images, {sec:.0f} sec, yes={int(df['answer'].fill_null(False).sum())}", flush=True)


if __name__ == "__main__":
    main()
