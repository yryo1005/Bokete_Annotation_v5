"""ex007 の教師と生徒の結果を集計し，図を作る．

- 通常の評価用データ（dev/test）の3軸のマクロ F1 と頑健な指標
- 少ない区分の評価用データ（dev/test）の少ない区分（ゲーム，絵画，カルタ）の F1 の平均（生徒は rare_probs.npz から）
- report_009 の最良の生徒（定義書 v2 の MLP）を v3 に読み替えた値（参考）

出力：outputs/ex007_label_v3/figures/summary.csv，fig_summary.png

実行例:
    .venv_annotate/bin/python programs/ex007_label_v3/summarize.py
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
from common import OUTPUT_DIR  # noqa: E402
from data import load_rare_labels  # noqa: E402
from students import to_frame  # noqa: E402
from teacher_v3 import rare_scores  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")


def collect() -> pl.DataFrame:
    """全ての実行の指標を1行1実行の表にする．

    Returns:
        pl.DataFrame: 列 method, setting, seed と指標を持つ表．
    """
    rare = load_rare_labels()
    rows = []
    for p in glob.glob(os.path.join(OUTPUT_DIR, "*", "*", "*", "metrics.json")):
        d = p.split(os.sep)
        m = json.load(open(p))
        r = {"method": d[-4], "setting": d[-3], "seed": d[-2]}
        for s in ["dev", "test"]:
            for k in ["format_macro_f1", "attr_macro_f1", "media_macro_f1", "overall_macro_f1", "overall_robust_f1"]:
                r[f"{s}_{k}"] = m[s][k]
        if "rare_dev" not in m:
            npz = os.path.join(os.path.dirname(p), "rare_probs.npz")
            probs = dict(np.load(npz))
            m.update(rare_scores(to_frame(probs), rare))
        for s in ["dev", "test"]:
            r[f"rare_{s}"] = m[f"rare_{s}"]["macro_f1"]
            for c in ["game", "painting", "karuta"]:
                r[f"rare_{s}_{c}"] = m[f"rare_{s}"][c]
        rows.append(r)
    return pl.DataFrame(rows)


def main() -> None:
    """集計と図の作成を行う．

    Returns:
        None
    """
    os.makedirs(FIG, exist_ok=True)
    df = collect()
    cols = [c for c in df.columns if c.startswith(("dev_", "test_", "rare_"))]
    g = df.group_by("method", "setting").agg([pl.len().alias("n")] + [pl.col(c).mean() for c in cols]
                                             + [pl.col(c).std().alias(f"{c}_std") for c in
                                                ["test_overall_macro_f1", "rare_test"]]).sort("dev_overall_macro_f1",
                                                                                             descending=True)
    g.write_csv(os.path.join(FIG, "summary.csv"))
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    g = g.sort("test_overall_macro_f1")
    names = [f"{m.replace('student_mlp', 'MLP').replace('teacher_', '教師 ')}｜{s.replace('__', '｜')}"
             for m, s in zip(g["method"], g["setting"])]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(g) + 1.5))
    for i, (k, lab) in enumerate([("test_overall_macro_f1", "通常 test：3軸の平均"), ("test_overall_robust_f1", "通常 test：robust"),
                                  ("rare_test", "少ない区分 test：ゲーム・絵画・カルタの平均")]):
        err = g[f"{k}_std"].fill_null(0) if f"{k}_std" in g.columns else None
        ax.barh(y + (i - 1) * 0.27, g[k], 0.27, xerr=err, capsize=2, label=lab)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("F1（最終評価用データ，生徒は3 seed の平均）")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_summary.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(g.sort("dev_overall_macro_f1", descending=True).select(
        "method", "setting", "n", "dev_overall_macro_f1", "rare_dev", "test_format_macro_f1", "test_attr_macro_f1",
        "test_media_macro_f1", "test_overall_macro_f1", "test_overall_robust_f1", "rare_test"))


if __name__ == "__main__":
    main()
