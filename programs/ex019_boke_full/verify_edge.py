"""パッケージのボケの段階が，例外的な入力で止まらずに処理できるかを確かめる．

小さな一覧（お題画像のファイルがないお題，本文が空のボケ，ボケが最も多い（max_group を超え分割される）お題，形式がないお題）を作り，
boke_predict → boke_compliance → finalize_shard を通して，出力の行数と各列の値を確かめる．

出力：outputs/ex019_boke_full/verify_edge.json

実行例:
    .venv_annotate/bin/python programs/ex019_boke_full/verify_edge.py
"""
import json
import os
import shutil
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex019_boke_full")
sys.path.insert(0, PKG)
from bokete_annotation.boke_pipeline import finalize_shard, stage_boke_compliance, stage_boke_predict  # noqa: E402


def main() -> None:
    """例外的な入力を処理し，結果を確かめる．

    Returns:
        None
    """
    d = os.path.join(OUTPUT_DIR, "edge")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "boke", "list"))
    src = pl.read_parquet(os.path.join(PKG, "outputs", "test_boke", "boke", "list", "part_0003.parquet"))
    big = src.group_by("rep_odaiId").len().sort("len", descending=True)["rep_odaiId"][0]
    a = src.filter(pl.col("rep_odaiId") == big)
    small = src.filter(pl.col("rep_odaiId") != big).head(5)
    missing = small.head(3).with_columns(pl.lit(999999999, dtype=src.schema["rep_odaiId"]).alias("rep_odaiId"), pl.lit("no/such/file.jpg").alias("path"),
                                         (pl.col("id") + 10**12).alias("id"))
    empty = small.tail(2).with_columns(pl.lit("").alias("text"), (pl.col("id") + 2 * 10**12).alias("id"))
    nofmt = small.head(1).with_columns(pl.lit(None, dtype=pl.String).alias("format"), pl.lit(None, dtype=src.schema["rep_odaiId"]).alias("rep_odaiId"),
                                       (pl.col("id") + 3 * 10**12).alias("id"))
    lst = pl.concat([a, missing, empty, nofmt]).sort(["rep_odaiId", "format", "id"], nulls_last=True)
    lst.write_parquet(os.path.join(d, "boke", "list", "part_9999.parquet"))
    cfg = json.load(open(os.path.join(PKG, "config.json"), encoding="utf-8"))
    stage_boke_predict(d, ["part_9999"], "/mnt/d/Bokete_Scraping_v3.2/datasets/images", cfg, PKG)
    stage_boke_compliance(d, ["part_9999"], cfg)
    out = finalize_shard(d, "part_9999")
    res = {"n_in": lst.height, "n_out": out.height,
           "big_group_predicted": int(out.filter(pl.col("id").is_in(a["id"].to_list()))["predicted"].sum()),
           "missing_image_predicted": out.filter(pl.col("id").is_in(missing["id"].to_list()))["predicted"].to_list(),
           "empty_text": out.filter(pl.col("id").is_in(empty["id"].to_list())).select("predicted", "empty_text").rows(),
           "no_format": out.filter(pl.col("id").is_in(nofmt["id"].to_list())).select("predicted", "format", "voice6").rows()}
    json.dump(res, open(os.path.join(OUTPUT_DIR, "verify_edge.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
