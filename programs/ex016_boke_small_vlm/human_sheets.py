"""小さな VLM の微調整に用いる，目視（Claude）のラベルの学習データを作るための抽出と一覧の作成．

教師のラベルを付けた学習データ（data.load_train，34,634件．検証データのお題は含まない）から960件を選ぶ．
半数は無作為に，残りの半数は教師のラベルで少ない発話の型（心の声・ツッコミ・状況の説明・名付け）と画像外の知識が「はい」の項目を多めに選ぶ．
お題1件につき1件までとする．一覧には教師のラベルを表示せず，定義書 第3版で目視のラベルを付ける（human_labels.txt）．

出力：datasets/ex016_boke_small_vlm/human_items.csv（列 hid, odaiId, text, format, stratum），human_sheets/sheet_XXX.png

実行例:
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/human_sheets.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ceiling_sheets import plot_sheet  # noqa: E402
from data import DATASET_DIR, load_train  # noqa: E402

N_RANDOM = 480
# 層 → 件数（教師のラベルによる）
STRATA = {"M": 160, "T": 80, "N": 80, "L": 80, "outside": 80}


def main() -> None:
    """抽出と一覧の作成を行う．

    Returns:
        None
    """
    rng = np.random.default_rng(789)
    df = load_train().with_row_index("row")
    df = df[rng.permutation(df.height).tolist()].unique("odaiId", keep="first", maintain_order=True)
    picked, used = [], set()
    for name, n in STRATA.items():
        cand = df.filter(pl.col("outside") & (pl.col("voice") != "A")) if name == "outside" else df.filter(pl.col("voice") == name)
        cand = cand.filter(~pl.col("row").is_in(list(used))).head(n)
        used |= set(cand["row"].to_list())
        picked.append(cand.with_columns(pl.lit(name).alias("stratum")))
    rand = df.filter(~pl.col("row").is_in(list(used))).head(N_RANDOM).with_columns(pl.lit("random").alias("stratum"))
    out = pl.concat(picked + [rand])
    out = out[rng.permutation(out.height).tolist()].with_row_index("hid").select(
        pl.col("hid").cast(pl.Int64), "odaiId", "text", "format", "stratum")
    os.makedirs(os.path.join(DATASET_DIR, "human_sheets"), exist_ok=True)
    out.write_csv(os.path.join(DATASET_DIR, "human_items.csv"))
    rows = [(f"h{h:04d}", fid, fmt, text) for h, fid, fmt, text in out.select("hid", "odaiId", "format", "text").rows()]
    for k in range(0, len(rows), 8):
        plot_sheet(rows[k:k + 8], os.path.join(DATASET_DIR, "human_sheets", f"sheet_{k // 8:03d}.png"))
    print(out.group_by("stratum").len())


if __name__ == "__main__":
    main()
