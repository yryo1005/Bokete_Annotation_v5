"""ex015 の結果の図を作る（evaluate.py の summary.csv と per_class.json から）．

- fig_teacher_variants.png：教師の指示・入力形式ごとの指標（調整用）
- fig_final.png：主要な方法の最終評価用の指標
- fig_voice6_per_class.png：主要な方法の6区分のクラスごとの F1（最終評価用）

出力：outputs/ex015_boke_labels_v3/figures/

実行例:
    .venv_annotate/bin/python programs/ex015_boke_labels_v3/figures.py
"""
import json
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METRICS = [("voice6_f1", "発話の型 6区分"), ("voice3_f1", "発話の型 3区分"), ("outside_f1", "画像外の知識")]
plt.rcParams["font.family"] = ["IPAexGothic", "Noto Sans CJK JP", "sans-serif"]


def mean_table(split: str) -> pl.DataFrame:
    """seed を平均した指標の表を返す．

    Args:
        split (str): dev / test．

    Returns:
        pl.DataFrame: 方法ごとの指標．
    """
    df = pl.read_csv(os.path.join(FIG, "summary.csv")).filter(pl.col("split") == split)
    df = df.with_columns(pl.col("method").str.replace(r"/\d+$", "").alias("m"))
    return df.group_by("m").agg([pl.col(c).mean() for c, _ in METRICS])


def bars(tab: pl.DataFrame, names: dict, path: str, title: str) -> None:
    """方法ごと・指標ごとの横棒グラフを描く．

    Args:
        tab (pl.DataFrame): mean_table の結果．
        names (dict): 方法 → 表示名（この順に並べる）．
        path (str): 保存先．
        title (str): 図の題．

    Returns:
        None
    """
    d = {r["m"]: r for r in tab.to_dicts()}
    keys = [k for k in names if k in d]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 0.45 * len(keys) + 1.4), sharey=True)
    y = np.arange(len(keys))
    for ax, (c, label), color in zip(axes, METRICS, COLORS):
        v = [d[k][c] if d[k][c] is not None else np.nan for k in keys]
        ax.barh(y, v, color=color, height=0.6)
        for yi, vi in zip(y, v):
            if not np.isnan(vi):
                ax.text(vi + 0.01, yi, f"{vi:.2f}", va="center", fontsize=8, color="#333333")
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(y, [names[k] for k in keys], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """図を作る．

    Returns:
        None
    """
    teachers = {"ex014/teacher_q32_v3": "前回の指示（32B）", "teacher_q32/one": "one：手順つき一括選択", "teacher_q32/one_cap": "one_cap：＋キャプション",
                "teacher_q32/one_kt_cap": "one_kt_cap：＋知識の種類", "teacher_q32/one_kt_cap_hr": "one_kt_cap_hr：＋高解像度",
                "teacher_q32/dec": "dec：分解した問い", "teacher_q32/dec2": "dec2：話し手の既定を画像内",
                "teacher_q32/dec_cap": "dec_cap", "teacher_q32/dec_fs": "dec_fs：＋例示", "teacher_q32/dec_fs_cap": "dec_fs_cap",
                "teacher_q30a3/one": "30B-A3B one", "teacher_q30a3/dec_fs": "30B-A3B dec_fs"}
    bars(mean_table("dev"), teachers, os.path.join(FIG, "fig_teacher_variants.png"), "教師の指示・入力形式の比較（調整用 308件）")
    final = json.load(open(os.path.join(FIG, "final_methods.json"), encoding="utf-8"))
    bars(mean_table("test"), final, os.path.join(FIG, "fig_final.png"), "主要な方法の比較（最終評価用 312件，生徒は3 seed の平均）")
    pc = json.load(open(os.path.join(FIG, "per_class.json"), encoding="utf-8"))
    voices = ["S", "M", "T", "N", "L", "A"]
    vname = ["セリフ", "心の声", "ツッコミ", "状況の説明", "名付け", "回答"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(final)
    for i, ((m, name), color) in enumerate(zip(final.items(), COLORS)):
        rows = [v["voice6_per_class"] for k, v in pc.items() if k.split("|")[1] == "test" and
                (k.split("|")[0] == m or k.split("|")[0].rsplit("/", 1)[0] == m)]
        if not rows:
            continue
        val = [np.mean([r[c] for r in rows]) for c in voices]
        ax.bar(np.arange(6) + i * w - 0.4 + w / 2, val, w * 0.92, color=color, label=name)
    ax.set_xticks(np.arange(6), vname)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_title("発話の型 6区分のクラスごとの F1（最終評価用）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_voice6_per_class.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
