"""report_020 の図（代表的な方法の最終評価用データでの項目ごとの F1）を作る．

出力：outputs/ex013_boke_labels/figures/fig_methods.png

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/figures.py
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
# 分類色（dataviz の既定の分類色の1〜6番．検証済み．3:1 未満の色があるため，棒に値を添える）
METHODS = [("rules", "規則", "#2a78d6"), ("teacher_q30a3_v2", "教師 30B-A3B（第2版）", "#eb6834"),
           ("teacher_q32_v2", "教師 32B（第2版）", "#1baf7a"), ("student/mlp_ruri310_text_fmt_img", "生徒 MLP（文＋形式＋画像）", "#eda100"),
           ("student/ft_ruri", "生徒 微調整（ruri-v3）", "#e87ba4"), ("student/stack", "生徒＋目視ラベルで補正", "#008300")]
ITEMS = [("voice_f1", "発話の型\n（マクロ F1）"), ("know_f1", "必要な知識\n（マクロ F1）"), ("uses_text_f1", "画像中の文字\nへの依存（F1）"),
         ("inappropriate_f1", "性的・不適切\n（F1）")]
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
    d = d.with_columns(pl.col("method").str.replace(r"/\d+$", "").alias("m"))
    g = d.group_by("m").agg([pl.col(k).mean() for k, _ in ITEMS])
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    w = 0.13
    x = np.arange(len(ITEMS))
    for i, (m, name, color) in enumerate(METHODS):
        row = g.filter(pl.col("m") == m)
        vals = [float(row[k][0]) for k, _ in ITEMS]
        bars = ax.bar(x + (i - 2.5) * w, vals, w - 0.015, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, [n for _, n in ITEMS], color=INK2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("最終評価用データ（210件）の F1", color=INK2)
    ax.grid(axis="y", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_methods.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
