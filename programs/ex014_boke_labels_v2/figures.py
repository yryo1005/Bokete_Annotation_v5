"""report_021 の図（代表的な方法の区分ごとのマクロ F1）を作る．

出力：outputs/ex014_boke_labels_v2/figures/fig_schemes.png

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/figures.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
# 分類色（dataviz の既定の分類色の1〜5番．検証済み．3:1 未満の色があるため棒に値を添える）
METHODS = [("rules", "規則", "#2a78d6"), ("teacher_q32_v3", "教師 32B（第3版）", "#eb6834"),
           ("student/base__ruri_fmt", "生徒（30B-A3B のラベル）", "#1baf7a"),
           ("student/aug_all_w__ruri_fmt", "生徒（水増し＋重み）", "#eda100"),
           ("student/aug_all_w__ruri_fmt__stack", "生徒（水増し＋重み）＋目視で補正", "#e87ba4")]
ITEMS = [("voice6_f1", "発話の型\n6区分"), ("voice3_f1", "発話の型\n3区分"), ("know4_f1", "知識\n4区分"),
         ("know3_0G_f1", "知識 3区分\n（0とGを統合）"), ("know2_O_f1", "知識 2区分\n（画像外か）")]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main() -> None:
    """図を作る．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    d = pl.read_csv(os.path.join(FIG, "summary.csv")).filter(pl.col("split") == "test")
    g = d.with_columns(pl.col("method").str.replace(r"/\d+$", "").alias("m")).group_by("m").agg([pl.col(k).mean() for k, _ in ITEMS])
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    w, x = 0.16, np.arange(len(ITEMS))
    for i, (m, name, color) in enumerate(METHODS):
        vals = [float(g.filter(pl.col("m") == m)[k][0]) for k, _ in ITEMS]
        bars = ax.bar(x + (i - 2) * w, vals, w - 0.015, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, [n for _, n in ITEMS], color=INK2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("最終評価用データ（312件）のマクロ F1", color=INK2)
    ax.grid(axis="y", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_schemes.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
