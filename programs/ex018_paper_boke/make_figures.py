"""論文（paper_boke/）の図を作る（unified_eval.py の unified.csv と，ex016 の学習曲線の予測から）．

- fig_progress.pdf：主な段階の方法の最終評価用の F1（発話の型6区分，画像外の知識．95%信頼区間）
- fig_teacher_prompts.pdf：教師（32B）の指示ごとの調整用の F1
- fig_learning_curve.pdf：目視ラベルの件数と F1（Qwen3-VL-4B，教師ラベルで学習済みの重みから微調整）
図には題を入れず，論文のキャプションで説明する．

出力：paper_boke/figures/

実行例:
    .venv_annotate/bin/python programs/ex018_paper_boke/make_figures.py
"""
import os
import sys

import matplotlib
import numpy as np
import polars as pl
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_eval import D16, OUT, OUTPUT_DIR, PROJECT_DIR, VOICES  # noqa: E402

FIG = os.path.join(PROJECT_DIR, "paper_boke", "figures")
plt.rcParams["font.family"] = ["IPAexGothic", "sans-serif"]
plt.rcParams["font.size"] = 9
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#7a7a7a"


def progress(u: pl.DataFrame) -> None:
    """主な段階の方法の最終評価用の F1 を描く．

    Args:
        u (pl.DataFrame): unified.csv．

    Returns:
        None
    """
    keys = [("rules", "規則"), ("t32_v3", "32B 教師（第3版）"), ("s14_w_stack", "文の生徒＋補正"), ("t32_kt", "32B 教師（最終の指示）"),
            ("s15_extra", "文の生徒（3.5万件）"), ("ft1_q4", "VLM 段階1"), ("hum_q4", "VLM 目視ラベルのみ"), ("ft2_q4", "VLM 段階2（4B）"),
            ("ft2_q8", "VLM 段階2（8B）"), ("ens_q4q8", "VLM 4B＋8B")]
    t = {r["key"]: r for r in u.filter(pl.col("split") == "test").iter_rows(named=True)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    y = np.arange(len(keys))
    for ax, (col, lo, hi, color, label) in zip(axes, [("voice6_f1", "voice6_lo", "voice6_hi", BLUE, "発話の型（6区分のマクロ F1）"),
                                                      ("outside_f1", "outside_lo", "outside_hi", ORANGE, "画像外の知識（F1）")]):
        v = np.array([t[k][col] for k, _ in keys])
        err = np.array([[t[k][col] - t[k][lo], t[k][hi] - t[k][col]] for k, _ in keys]).T
        ax.barh(y, v, 0.6, color=color, xerr=err, ecolor="#555555", capsize=2)
        for yi, vi in zip(y, v):
            ax.text(0.02, yi, f"{vi:.2f}", va="center", fontsize=7, color="white")
        ax.axvline({"voice6_f1": 0.799, "outside_f1": 0.849}[col], color=GRAY, ls="--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [n for _, n in keys], fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_progress.pdf"))
    plt.close(fig)


def teacher_prompts(u: pl.DataFrame) -> None:
    """教師（32B）の指示ごとの調整用の F1 を描く．

    Args:
        u (pl.DataFrame): unified.csv．

    Returns:
        None
    """
    keys = [("t32_v3", "第3版"), ("t32_one", "one"), ("t32_one_cap", "one_cap"), ("t32_kt", "one_kt_cap"), ("t32_dec", "dec"),
            ("t32_dec_fs", "dec_fs")]
    d = {r["key"]: r for r in u.filter(pl.col("split") == "dev").iter_rows(named=True)}
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(keys))
    for i, (col, color, label) in enumerate([("voice6_f1", BLUE, "発話の型 6区分"), ("voice3_f1", GREEN, "発話の型 3区分"),
                                             ("outside_f1", ORANGE, "画像外の知識")]):
        ax.bar(x + (i - 1) * 0.27, [d[k][col] for k, _ in keys], 0.26, color=color, label=label)
    ax.set_xticks(x, [n for _, n in keys], fontsize=7, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("調整用の F1")
    ax.legend(fontsize=6, loc="upper left", ncol=3, frameon=False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_teacher_prompts.pdf"))
    plt.close(fig)


def learning_curve() -> None:
    """目視ラベルの件数と F1 を描く（Qwen3-VL-4B）．

    Returns:
        None
    """
    ev = D16.load_eval()
    base = os.path.join(OUT, "ex016_boke_small_vlm", "ft", "q4", "teacher_lr0.0001_ep2_bs16_r16")
    runs = [(0, base), (232, base + "__curve0.25"), (465, base + "__curve0.5"), (931, base + "__stage2")]
    res = {"dev": {"v": [], "o": []}, "test": {"v": [], "o": []}}
    for _, d in runs:
        p = ev.select("no").join(pl.read_csv(os.path.join(d, "0", "eval.csv")), on="no", maintain_order="left")
        for split in res:
            m = (ev["split"] == split).to_numpy()
            res[split]["v"].append(f1_score(ev["voice"].to_numpy()[m], p["voice"].to_numpy()[m], labels=VOICES, average="macro"))
            res[split]["o"].append(f1_score(ev["outside"].to_numpy()[m], p["p_outside"].to_numpy()[m] > 0.5))
    n = [r[0] for r in runs]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for split, ls in [("dev", "-"), ("test", "--")]:
        ax.plot(n, res[split]["v"], ls, marker="o", color=BLUE, label=f"発話の型 6区分（{ '調整用' if split == 'dev' else '最終評価用'}）")
        ax.plot(n, res[split]["o"], ls, marker="s", color=ORANGE, label=f"画像外の知識（{ '調整用' if split == 'dev' else '最終評価用'}）")
    ax.set_xticks(n)
    ax.set_xlabel("段階2に用いた目視ラベルの件数")
    ax.set_ylabel("F1")
    ax.set_ylim(0.5, 0.85)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.grid(color="#dddddd", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_learning_curve.pdf"))
    plt.close(fig)
    pl.DataFrame({"n_human": n, "dev_voice6": res["dev"]["v"], "test_voice6": res["test"]["v"], "dev_outside": res["dev"]["o"],
                  "test_outside": res["test"]["o"]}).write_csv(os.path.join(OUTPUT_DIR, "learning_curve.csv"))


def main() -> None:
    """図を作る．

    Returns:
        None
    """
    os.makedirs(FIG, exist_ok=True)
    u = pl.read_csv(os.path.join(OUTPUT_DIR, "unified.csv"))
    progress(u)
    teacher_prompts(u)
    learning_curve()


if __name__ == "__main__":
    main()
