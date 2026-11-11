"""対象の1,620枚を並べ替えて番号を付け，目視で v6 の属性を付けるための画像の一覧を作る．

予測や v5 のラベルの影響を避けるため，一覧には番号だけを表示する．

出力：datasets/ex026_text_attr_v6/targets.csv（列 no と data.build_targets の列），sheets/sheet_{k:03}.png

実行例:
    .venv_annotate/bin/python programs/ex026_text_attr_v6/make_sheets.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import polars as pl
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, SEED, TARGETS, build_targets, image_path  # noqa: E402

PER_SHEET = 9
TILE = 640


def draw(rows: list, out: str) -> None:
    """画像の一覧を1枚の図にする．

    Args:
        rows (list[dict]): 列 no, rep_odaiId を持つ行のリスト（最大 PER_SHEET 行）．
        out (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        im = Image.open(image_path(r["rep_odaiId"])).convert("RGB")
        im.thumbnail((TILE, TILE))
        ax.imshow(im)
        ax.set_title(f"{r['no']}", fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main() -> None:
    """対象を並べ替えて一覧を保存する（targets.csv があれば並べ替え直さない）．

    Returns:
        None
    """
    os.makedirs(os.path.join(DATASET_DIR, "sheets"), exist_ok=True)
    if not os.path.exists(TARGETS):
        t = build_targets()
        t = t[np.random.default_rng(SEED).permutation(t.height)].with_row_index("no")
        t.write_csv(TARGETS)
    rows = pl.read_csv(TARGETS).select("no", "rep_odaiId").to_dicts()
    jobs = [(rows[k:k + PER_SHEET], os.path.join(DATASET_DIR, "sheets", f"sheet_{k // PER_SHEET:03d}.png")) for k in range(0, len(rows), PER_SHEET)]
    with ProcessPoolExecutor(8) as ex:
        list(tqdm(ex.map(draw, *zip(*jobs)), total=len(jobs), desc="sheets", leave=False))
    print(pl.read_csv(TARGETS).group_by("source").len())


if __name__ == "__main__":
    main()
