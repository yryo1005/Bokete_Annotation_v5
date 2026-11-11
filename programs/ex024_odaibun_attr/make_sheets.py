"""お題文ありの目視のラベルを付けるための，画像の一覧（番号付き）を作る．

対象は，目視のラベル（定義書 v4，1,020枚）のうち形式が穴埋めまたはカルタの画像である．
形式がお題文の画像はお題文あり，通常の画像はお題文なしとみなせるため，目視で付け直す必要はない．

出力：datasets/ex024_odaibun_attr/sheets/sheet_{k:03}.png，datasets/ex024_odaibun_attr/relabel_targets.csv（列 no, fileId, set, format）

実行例:
    .venv_annotate/bin/python programs/ex024_odaibun_attr/make_sheets.py
"""
import os
import sys

import matplotlib
import polars as pl
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, image_path, load_truth_v4  # noqa: E402

PER_SHEET = 9
TILE = 520


def main() -> None:
    """対象の一覧と画像の一覧を保存する．

    Returns:
        None
    """
    t = load_truth_v4().filter(pl.col("format").is_in(["anaume", "karuta"])).sort("fileId")
    t = t.with_row_index("no").select("no", "fileId", "set", "format")
    os.makedirs(os.path.join(DATASET_DIR, "sheets"), exist_ok=True)
    t.write_csv(os.path.join(DATASET_DIR, "relabel_targets.csv"))
    rows = t.to_dicts()
    for k in tqdm(range(0, len(rows), PER_SHEET), desc="sheets", leave=False):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), rows[k:k + PER_SHEET]):
            im = Image.open(image_path(r["fileId"])).convert("RGB")
            im.thumbnail((TILE, TILE))
            ax.imshow(im)
            ax.set_title(f"{r['no']}", fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(DATASET_DIR, "sheets", f"sheet_{k // PER_SHEET:03d}.png"), dpi=60)
        plt.close(fig)
    print(t.group_by("format").len())


if __name__ == "__main__":
    main()
