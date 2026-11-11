"""目視のラベルの再現性を確かめるため，対象の1,620枚から90枚を無作為に選び，番号を付け直した一覧を作る．

出力：datasets/ex026_text_attr_v6/repeat_targets.csv（列 rno, no），sheets_repeat/sheet_{k:03}.png

実行例:
    .venv_annotate/bin/python programs/ex026_text_attr_v6/make_repeat.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, TARGETS  # noqa: E402
from make_sheets import PER_SHEET, draw  # noqa: E402

N_REPEAT = 90
SEED_REPEAT = 1


def main() -> None:
    """90枚を選び，番号を付け直して一覧を保存する．

    Returns:
        None
    """
    out = os.path.join(DATASET_DIR, "repeat_targets.csv")
    t = pl.read_csv(TARGETS)
    if not os.path.exists(out):
        idx = np.random.default_rng(SEED_REPEAT).choice(t.height, N_REPEAT, replace=False)
        t[idx].select("no").with_row_index("rno").write_csv(out)
    r = pl.read_csv(out).join(t.select("no", "rep_odaiId"), on="no")
    os.makedirs(os.path.join(DATASET_DIR, "sheets_repeat"), exist_ok=True)
    rows = r.select(pl.col("rno").alias("no"), "rep_odaiId").sort("no").to_dicts()
    for k in range(0, len(rows), PER_SHEET):
        draw(rows[k:k + PER_SHEET], os.path.join(DATASET_DIR, "sheets_repeat", f"sheet_{k // PER_SHEET:03d}.png"))


if __name__ == "__main__":
    main()
