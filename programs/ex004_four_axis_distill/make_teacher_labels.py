"""教師ラベル用の画像（40,000枚）の VLM の出力に後処理の規則を適用し，生徒の学習に用いる教師ラベルを作る．

入力：datasets/ex004_four_axis_distill/teacher_labels/Qwen3-VL-8B-Instruct_v5.csv
出力：同じディレクトリの Qwen3-VL-8B-Instruct_v5_rules.csv（作品不明の規則とボケによるカルタの照合を適用したもの）

実行例:
    .venv_annotate/bin/python programs/ex004_four_axis_distill/make_teacher_labels.py
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR  # noqa: E402
from rules import apply_rules  # noqa: E402


def main() -> None:
    """規則を適用した教師ラベルを保存し，適用による変化を表示する．

    Returns:
        None
    """
    src = os.path.join(DATASET_DIR, "teacher_labels", "Qwen3-VL-8B-Instruct_v5.csv")
    raw = pl.read_csv(src)
    out = apply_rules(raw.drop("fileId"), raw["fileId"].to_list())
    out = pl.concat([raw.select("fileId"), out], how="horizontal")
    out.write_csv(src.replace("_v5.csv", "_v5_rules.csv"))
    print("形式の分布（規則の適用前）:", raw["format"].value_counts(sort=True).to_dicts())
    print("形式の分布（規則の適用後）:", out["format"].value_counts(sort=True).to_dicts())
    print("作品不明:", int(out["work_unknown"].sum()), "枚")


if __name__ == "__main__":
    main()
