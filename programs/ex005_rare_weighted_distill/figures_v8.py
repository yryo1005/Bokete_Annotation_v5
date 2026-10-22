"""report_009 の図を作る．

- fig_v8_teachers.png：教師の構成ごとの通常の評価用データ（dev）の3軸の平均と，少ない区分の評価用データ（dev）の4区分の F1 の平均
- fig_v8_students.png：生徒の条件ごとの通常の評価用データ（test）と少ない区分の評価用データ（test）の指標（seed 間の平均 ± 標準偏差）
- fig_v8_rare_items.png：生徒の条件ごとの少ない区分（test）の区分別の F1
- fig_v8_f8_errors.png：F_v8_en が「お題文」と誤った通常の画像の例
- fig_v8_pc_examples.png：絵画と CG の独立の質問で教師の判定が変わった画像の例（少ない区分の評価用データ）

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/figures_v8.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combine_v8 as C  # noqa: E402
from common import OUTPUT_DIR, plot_image_grid  # noqa: E402
from data import MEDIA, load_eval_labels  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
TEACHER_ORDER = ["base", "F8", "F8b", "F8c", "J8", "M8", "Cen", "Cja", "TLen", "TLja", "TLen_fmt", "TLja_fmt", "PCen", "PCja",
                 "PCen_paint", "PCen_cgvote", "PCen_cgand", "PCja_cgand", "M6M8paint"]
SHORT = {"photo": "写真", "anime": "アニメ", "game": "ゲーム", "cg": "CG", "manga": "マンガ", "text_graphic": "テキスト",
         "painting": "絵画", "illustration": "イラスト"}


def setup_font() -> None:
    """図に日本語のフォントを設定する．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()


def plot_teachers() -> None:
    """教師の構成の比較の図を描く．

    Returns:
        None
    """
    rows = []
    for name in TEACHER_ORDER:
        p = os.path.join(OUTPUT_DIR, f"teacher_v8_{name}", "rules", "0", "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            rows.append((name, m["dev"]["overall_macro_f1"], m["dev"]["overall_robust_f1"], m.get("rare_dev", {}).get("macro_f1")))
    names = [r[0] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.5))
    ax.barh(y - 0.27, [r[1] for r in rows], 0.27, label="通常 dev：3軸の平均（overall）")
    ax.barh(y, [r[2] for r in rows], 0.27, label="通常 dev：頑健な指標（robust）")
    ax.barh(y + 0.27, [r[3] or 0 for r in rows], 0.27, label="少ない区分 dev：4区分の F1 の平均")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("F1（調整用データ）")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v8_teachers.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def student_table() -> pl.DataFrame:
    """生徒の条件ごとに通常の評価用データと少ない区分の評価用データの指標を集計する．

    Returns:
        pl.DataFrame: 条件ごとの平均と標準偏差の表．
    """
    rows = []
    for p in glob.glob(os.path.join(OUTPUT_DIR, "*", "*", "*", "metrics_rare_v8.json")):
        d = p.split(os.sep)
        m = json.load(open(p.replace("metrics_rare_v8.json", "metrics.json")))
        r = json.load(open(p))
        rows.append({"method": d[-4], "setting": d[-3], "seed": d[-2], "test_overall": m["test"]["overall_macro_f1"],
                     "test_robust": m["test"]["overall_robust_f1"], "rare_test": r["rare_test"]["macro_f1"],
                     **{f"rare_{c}": r["rare_test"][c]["f1"] for c in C.RARE_CLASSES}})
    df = pl.DataFrame(rows).with_columns(
        (pl.col("method").replace({"student_mlp": "MLP", "finetune_siglip2_base": "微調整"}) + "｜"
         + pl.col("setting").str.replace("__", "｜").str.replace("__", "｜")).alias("name"))
    cols = ["test_overall", "test_robust", "rare_test"] + [f"rare_{c}" for c in C.RARE_CLASSES]
    return df.group_by("name").agg([pl.col(c).mean().alias(f"{c}_mean") for c in cols]
                                   + [pl.col(c).std().alias(f"{c}_std") for c in cols]).sort("rare_test_mean")


def plot_students(df: pl.DataFrame) -> None:
    """生徒の比較の図を描く．

    Args:
        df (pl.DataFrame): student_table の戻り値．

    Returns:
        None
    """
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1.5))
    for i, (k, lab) in enumerate([("test_overall", "通常 test：overall"), ("test_robust", "通常 test：robust"),
                                  ("rare_test", "少ない区分 test：4区分の平均")]):
        ax.barh(y + (i - 1) * 0.27, df[f"{k}_mean"], 0.27, xerr=df[f"{k}_std"].fill_null(0), capsize=2, label=lab)
    ax.set_yticks(y, df["name"], fontsize=7.5)
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("F1（最終評価用データ，3 seed の平均）")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v8_students.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 1.5))
    mat = np.array([[df[f"rare_{c}_mean"][i] for c in C.RARE_CLASSES] for i in range(len(df))])
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4), ["ゲーム", "CG", "絵画", "カルタ"])
    ax.set_yticks(range(len(df)), df["name"], fontsize=7.5)
    for i, j in np.ndindex(mat.shape):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if mat[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="F1（少ない区分 test）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v8_rare_items.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def plot_examples() -> None:
    """プロンプトの改善策の実例の図を描く．

    Returns:
        None
    """
    labels = load_eval_labels()
    p = pl.read_csv(os.path.join(OUTPUT_DIR, "teacher_components", "F_v8_en", "eval.csv"))
    d = labels.select("fileId", "format").join(p.select("fileId", pl.col("format").alias("p")), on="fileId")
    bad = d.filter((pl.col("format") == "normal") & (pl.col("p") == "odaibun")).sample(n=8, seed=0)
    plot_image_grid(bad["fileId"].to_list(), [f"id={i}\n目視: 通常 / F_v8_en: お題文" for i in bad["fileId"]],
                    "改訂した形式のプロンプト（F_v8_en）がお題文と誤った通常の画像", os.path.join(FIG, "fig_v8_f8_errors.png"), ncols=4)
    rare = C.load_rare_eval()
    parts_b, parts_n = [], []
    for s in ["rare_eval", "rare_eval_v2"]:
        ids = rare.filter(pl.col("set") == s)["fileId"].to_list()
        parts_b.append(C.build_v8(C.TEACHERS_V8["base"], s, ids).select(MEDIA))
        parts_n.append(C.build_v8(C.TEACHERS_V8["PCen_cgand"], s, ids).select(MEDIA))
    b, n = pl.concat(parts_b), pl.concat(parts_n)
    changed = [i for i in range(len(rare)) if (b["painting"][i] != n["painting"][i] or b["cg"][i] != n["cg"][i])
               and (rare["painting"][i] or rare["cg"][i])]
    sel = [int(i) for i in np.random.default_rng(0).choice(changed, min(12, len(changed)), replace=False)]

    def ms(row) -> str:
        """媒体の行を表示用の文字列にする．"""
        return "+".join(SHORT[m] for m in MEDIA if row[m]) or "なし"

    titles = [f"id={rare['fileId'][i]}\n目視: {ms(rare.row(int(i), named=True))}\n従来: {ms(b.row(int(i), named=True))}"
              f"\n改善後: {ms(n.row(int(i), named=True))}" for i in sel]
    plot_image_grid([rare["fileId"][int(i)] for i in sel], titles, "絵画と CG の独立の質問で教師の判定が変わった画像（目視で絵画または CG）",
                    os.path.join(FIG, "fig_v8_pc_examples.png"), ncols=4)


def main() -> None:
    """全ての図を作る．

    Returns:
        None
    """
    setup_font()
    os.makedirs(FIG, exist_ok=True)
    plot_teachers()
    df = student_table()
    df.write_csv(os.path.join(FIG, "summary_v8_students.csv"))
    plot_students(df)
    plot_examples()
    print(df.select("name", "test_overall_mean", "test_robust_mean", "rare_test_mean", "rare_test_std"))


if __name__ == "__main__":
    main()
