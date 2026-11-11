"""ex016 の結果の図と表を作る（evaluate.py の summary.csv，各方法の speed.json / best.json から）．

- fig_accuracy_speed.png：方法ごとの最終評価用の F1（6区分，画像外の知識）と1秒あたりの処理件数（対数軸）
- fig_bars.png：主要な方法の6区分・3区分・画像外の知識の F1（最終評価用．目視のラベルの再現性を破線で示す）
- table_methods.csv：方法ごとの指標と速度，9,250万件の推定処理時間

正解は合意ラベル（cons）とする．速度は，ゼロショットは vLLM で620件を一括で生成したときの件数/秒，
微調整は分類器の順伝播のみ（バッチ32）の件数/秒とする．

出力：outputs/ex016_boke_small_vlm/figures/

実行例:
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/figures.py
"""
import glob
import json
import os
import sys

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
N_ALL = 92_500_000
COLORS = {"zeroshot": "#eb6834", "ft_teacher": "#2a78d6", "ft_human": "#1baf7a", "ref": "#7a7a7a"}
plt.rcParams["font.family"] = ["IPAexGothic", "sans-serif"]
# 目視のラベルの再現性（元のラベルと，見ずに付け直したラベルの一致．620件）
HUMAN = {"voice6_f1": 0.799, "voice3_f1": 0.928, "outside_f1": 0.849}


def speed_of(method: str) -> float:
    """方法の処理速度（件/秒）を返す．

    Args:
        method (str): 方法名（summary.csv の method から seed を除いたもの）．

    Returns:
        float: 件/秒（不明なら None）．
    """
    if method.startswith("zeroshot/"):
        return json.load(open(os.path.join(OUTPUT_DIR, method, "speed.json")))["per_sec"]
    if method.startswith("ft/"):
        parts = method.split("/")
        base = os.path.join(OUTPUT_DIR, "ft", parts[1], parts[2].split("__")[0], "0", "speed.json")
        if os.path.exists(base):
            return json.load(open(base))["bs32"]["model_per_sec"]
    if method == "ex015/teacher_q32_one_kt_cap":
        return 3.9
    return None


def kind(method: str) -> str:
    """方法の種類を返す．

    Args:
        method (str): 方法名．

    Returns:
        str: zeroshot / ft_teacher / ft_human / ref．
    """
    if method.startswith("zeroshot/"):
        return "zeroshot"
    if method.startswith("ft/"):
        return "ft_teacher" if "__" not in method and "human_only" not in method else "ft_human"
    return "ref"


def main() -> None:
    """図と表を作る．

    Returns:
        None
    """
    d = pl.read_csv(os.path.join(FIG, "summary.csv")).filter(pl.col("gold") == "cons")
    d = d.with_columns(pl.col("method").str.replace(r"/\d+$", "").alias("m"))
    g = d.group_by("m", "split").agg([pl.col(c).mean() for c in ["voice6_f1", "voice3_f1", "outside_f1", "outside_prec", "outside_rec"]])
    g = g.pivot(on="split", index="m", values=["voice6_f1", "voice3_f1", "outside_f1", "outside_prec", "outside_rec"])
    g = g.with_columns(pl.col("m").map_elements(speed_of, return_dtype=pl.Float64).alias("per_sec"),
                       pl.col("m").map_elements(kind, return_dtype=pl.String).alias("kind"))
    g = g.with_columns((N_ALL / pl.col("per_sec") / 86400).alias("days_for_all"))
    cols = ["m", "kind", "per_sec", "days_for_all"] + [f"{c}_{s}" for c in ["voice6_f1", "voice3_f1", "outside_f1", "outside_prec", "outside_rec"]
                                                       for s in ["dev", "test"]]
    g = g.select(cols).sort("voice6_f1_dev", descending=True)
    g.write_csv(os.path.join(FIG, "table_methods.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (c, title) in zip(axes, [("voice6_f1_test", "発話の型 6区分（マクロ F1）"), ("outside_f1_test", "画像外の知識（F1）")]):
        for r in g.filter(pl.col("per_sec").is_not_null()).iter_rows(named=True):
            ax.scatter(r["per_sec"], r[c], color=COLORS[r["kind"]], s=40, zorder=3)
            ax.annotate(r["m"].replace("teacher_lr0.0001_ep2_bs16_r16", "t").replace("teacher_lr0.0001_ep1_bs16_r16", "t"),
                        (r["per_sec"], r[c]), fontsize=6, xytext=(3, 2), textcoords="offset points", color="#333333")
        ax.axhline(HUMAN[c.replace("_test", "")], color="#7a7a7a", ls="--", lw=1)
        ax.axhline(0.8, color="#e87ba4", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("処理速度（件/秒，対数軸）")
        ax.set_ylabel("最終評価用の F1")
        ax.set_title(title, fontsize=10)
        ax.grid(color="#dddddd", lw=0.6)
        ax.set_ylim(0, 1)
    handles = [plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in
               [("ゼロショット", COLORS["zeroshot"]), ("教師のラベルで微調整", COLORS["ft_teacher"]),
                ("＋目視のラベルで微調整", COLORS["ft_human"]), ("32B 教師", COLORS["ref"])]]
    handles += [plt.Line2D([], [], ls="--", color="#7a7a7a", label="目視のラベルの再現性"), plt.Line2D([], [], ls=":", color="#e87ba4", label="目標 0.8")]
    axes[1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_accuracy_speed.png"), dpi=150)
    plt.close(fig)
    with pl.Config(tbl_rows=60, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=300, float_precision=3):
        print(g)


if __name__ == "__main__":
    main()
