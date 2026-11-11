"""目視のラベルの揺れ（再現性）を測るため，検証データ620件から160件を無作為に選び，ラベルを伏せた一覧を作る．

元の番号と異なる番号（r000〜）を付け，元のラベルを見ずに定義書 第3版で付け直す（ceiling_labels.txt）．
付け直したラベルと元のラベルの一致（6区分のマクロ F1，画像外の知識の F1）を，自動の判定の精度の上限の目安とする．

出力：datasets/ex016_boke_small_vlm/ceiling_items.csv（列 rid, no），ceiling_sheets/sheet_XX.png

残りの460件も同じ方法で付け直す（--rest．rest_labels.txt）．

実行例:
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/ceiling_sheets.py
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/ceiling_sheets.py --rest
"""
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, image_path, load_eval  # noqa: E402

N = 160


def plot_sheet(rows: list, path: str) -> None:
    """画像とボケを2列×4行に並べた一覧を保存する（ex014 sample_rare.py と同じ体裁）．

    Args:
        rows (list[tuple]): (番号, お題ID, 形式, ボケ) のリスト．
        path (str): 保存先．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={"width_ratios": [1, 1.1, 1, 1.1]})
    for ax in axes.ravel():
        ax.axis("off")
    for k, (no, fid, fmt, text) in enumerate(rows):
        r, c = k // 2, (k % 2) * 2
        im = Image.open(image_path(fid)).convert("RGB")
        im.thumbnail((400, 400))
        axes[r, c].imshow(im)
        axes[r, c].set_title(f"{no}（{fmt}）", fontsize=9)
        axes[r, c + 1].text(0, 0.95, "\n".join(textwrap.wrap((text or "").replace("\n", " "), 18)), va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def rest_sheets() -> None:
    """残りの460件（ceiling_items.csv にない項目）にも，番号 s000〜 を付けた一覧を作る（全件の2回目のラベル付け用）．

    出力：datasets/ex016_boke_small_vlm/rest_items.csv（列 sid, no），rest_sheets/sheet_XX.png

    Returns:
        None
    """
    ev = load_eval()
    done = set(pl.read_csv(os.path.join(DATASET_DIR, "ceiling_items.csv"))["no"].to_list())
    rest = ev.filter(~pl.col("no").is_in(list(done)))
    idx = np.random.default_rng(456).permutation(rest.height)
    sub = rest[idx.tolist()].with_row_index("sid")
    os.makedirs(os.path.join(DATASET_DIR, "rest_sheets"), exist_ok=True)
    sub.select(pl.col("sid").cast(pl.Int64), "no").write_csv(os.path.join(DATASET_DIR, "rest_items.csv"))
    rows = [(f"s{r:03d}", fid, fmt, text) for r, fid, fmt, text in sub.select("sid", "odaiId", "format", "text").rows()]
    for k in range(0, len(rows), 8):
        plot_sheet(rows[k:k + 8], os.path.join(DATASET_DIR, "rest_sheets", f"sheet_{k // 8:02d}.png"))


def main() -> None:
    """抽出と一覧の作成を行う．

    Returns:
        None
    """
    ev = load_eval()
    idx = np.random.default_rng(123).permutation(ev.height)[:N]
    sub = ev[idx.tolist()].with_row_index("rid")
    os.makedirs(os.path.join(DATASET_DIR, "ceiling_sheets"), exist_ok=True)
    sub.select(pl.col("rid").cast(pl.Int64), "no").write_csv(os.path.join(DATASET_DIR, "ceiling_items.csv"))
    rows = [(f"r{r:03d}", fid, fmt, text) for r, fid, fmt, text in sub.select("rid", "odaiId", "format", "text").rows()]
    for k in range(0, len(rows), 8):
        plot_sheet(rows[k:k + 8], os.path.join(DATASET_DIR, "ceiling_sheets", f"sheet_{k // 8:02d}.png"))


if __name__ == "__main__":
    if "--rest" in sys.argv:
        rest_sheets()
    else:
        main()
