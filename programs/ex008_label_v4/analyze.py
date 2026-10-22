"""ex008 の教師と生徒を，通常の評価用データ，少ない区分の評価用データ，実データの目視ラベルで評価し，集計する．

実データの目視ラベルは2種類ある．
- 無作為抽出160枚（random）：実際の分布での精度をそのまま表す．
- 層別抽出220枚（strat）：生徒の予測に基づく11層から20枚ずつ抽出したもの．各クラスの件数がおおむね一様になり，
  少ないクラスの精度を測れる．集計は (1) 重みなし（クラスごとの精度を比べるため）と，
  (2) 抽出確率の逆数で重み付け（実際の分布での精度の推定）の両方を示す．
  実際の分布での推定（pooled_ipw）には，無作為抽出160枚も合わせる．層別抽出の各画像に 層の画像数 / 抽出数，
  無作為抽出の各画像に 母集団の画像数 / 160 の重みを与えるため，両者の重みの合計はともに母集団の画像数となり，
  推定値は2つの標本による推定値をおおむね等分に平均したものになる．

出力：outputs/ex008_label_v4/figures/summary.csv，real_check.json，fig_summary.png，fig_strat_items.png

実行例:
    .venv_annotate/bin/python programs/ex008_label_v4/analyze.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR  # noqa: E402
from data import ATTRIBUTES, DATASET_DIR, FORMATS, MEDIA, PROJECT_DIR, load_rare_labels, load_real_labels  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
EX7_BEST = os.path.join(PROJECT_DIR, "outputs", "ex007_label_v3", "student_mlp", "T_v8m__base+rare__unweighted", "0")
MIN_SUPPORT = 5
# 項目ごとの図に示す実行
KEY_RUNS = ["ex007_best(v3)", "teacher_T_v8m/rules", "teacher_T_v10oJXk/rules", "student_mlp/T_v10o__base+rare__unweighted",
            "student_mlp/T_v10oJXk__base+rare__unweighted"]
# 集計の図に示す実行（教師は形式と文字ありの構成の比較に要るもの）
SUMMARY_RUNS = KEY_RUNS + ["teacher_T_v10a/rules", "teacher_T_v10c/rules", "teacher_T_v10o/rules", "teacher_T_v10x/rules",
                           "student_mlp/T_v10oJXk__base__unweighted"]


def to_frame(probs: dict) -> pl.DataFrame:
    """出力層ごとの確率を予測の表に変換する（students.py の to_frame と同じ規則）．

    Args:
        probs (dict): キー format（形状 (N, 4)），attr（(N, 3)），media（(N, 7)）の確率．

    Returns:
        pl.DataFrame: 列 format と各項目（bool）を持つ表．
    """
    from students import to_frame as tf

    return tf(probs)


def real_metrics(truth: pl.DataFrame, pred: pl.DataFrame, w: np.ndarray = None) -> dict:
    """実データの目視ラベルに対する精度を求める．

    Args:
        truth (pl.DataFrame): 目視ラベル（列 format と各項目）．
        pred (pl.DataFrame): 予測（truth と同じ順）．
        w (np.ndarray): 画像ごとの重み（形状 (N,)，None で重みなし）．

    Returns:
        dict: 形式の正解率とマクロ F1，各項目の F1，軸ごとのマクロ F1（件数 MIN_SUPPORT 以上の項目），全一致の割合．
    """
    w = np.ones(len(truth)) if w is None else w
    tf, pf = truth["format"].to_numpy(), pred["format"].to_numpy()
    out = {"format_accuracy": float(np.average(tf == pf, weights=w)), "per_item": {}, "support": {}}
    for f in FORMATS:
        out["per_item"][f] = float(f1_score(tf == f, pf == f, sample_weight=w, zero_division=0))
        out["support"][f] = int((tf == f).sum())
    for c in ATTRIBUTES + MEDIA:
        t, p = truth[c].to_numpy().astype(bool), pred[c].to_numpy().astype(bool)
        out["per_item"][c] = float(f1_score(t, p, sample_weight=w, zero_division=0))
        out["support"][c] = int(t.sum())
    for axis, items in [("format", FORMATS), ("attr", ATTRIBUTES), ("media", MEDIA)]:
        ok = [out["per_item"][c] for c in items if out["support"][c] >= MIN_SUPPORT]
        out[f"{axis}_macro_f1"] = float(np.mean(ok))
    out["overall_macro_f1"] = float(np.mean([out[f"{a}_macro_f1"] for a in ["format", "attr", "media"]]))
    cols = ATTRIBUTES + MEDIA
    same = (tf == pf) & np.all(truth.select(cols).to_numpy() == pred.select(cols).to_numpy().astype(bool), axis=1)
    out["all_correct"] = float(np.average(same, weights=w))
    out["media_multi_rate"] = float(np.average(pred.select(MEDIA).to_numpy().astype(bool).sum(1) >= 2, weights=w))
    return out


def load_check_sets() -> tuple:
    """実データの目視ラベル（無作為抽出と層別抽出）と，予測の並び（real_check_v4.csv）を読み込む．

    Returns:
        tuple: (無作為抽出の表，層別抽出の表，real_check_v4 のお題IDのリスト)．
    """
    ids = pl.read_csv(os.path.join(DATASET_DIR, "candidates", "real_check_v4.csv"))["fileId"].to_list()
    return load_real_labels("random"), load_real_labels("strat"), ids


def check_scores(pred_all: pl.DataFrame, rnd: pl.DataFrame, strat: pl.DataFrame, n_pop: int) -> dict:
    """実データの目視ラベルに対する3種類の集計を行う．

    Args:
        pred_all (pl.DataFrame): 列 fileId と予測を持つ表（目視ラベルの画像を全て含む）．
        rnd (pl.DataFrame): 無作為抽出の目視ラベル．
        strat (pl.DataFrame): 層別抽出の目視ラベル（列 weight を持つ）．
        n_pop (int): 母集団（未使用の実データ）の画像数．

    Returns:
        dict: キー random，strat_uniform，pooled_ipw の real_metrics の戻り値．
    """
    def align(t):
        return t.select("fileId").join(pred_all, on="fileId", how="left", maintain_order="left")

    both = pl.concat([rnd.select(["fileId", "format"] + ATTRIBUTES + MEDIA), strat.select(["fileId", "format"] + ATTRIBUTES + MEDIA)])
    w = np.concatenate([np.full(len(rnd), n_pop / len(rnd)), strat["weight"].to_numpy()])
    return {"random": real_metrics(rnd, align(rnd)), "strat_uniform": real_metrics(strat, align(strat)),
            "pooled_ipw": real_metrics(both, align(both), w)}


def collect() -> tuple:
    """全ての実行の指標を集める．

    Returns:
        tuple: (1行1実行の指標の表，実データの集計の辞書)．
    """
    from teacher_v4 import rare_scores

    rnd, strat, ids = load_check_sets()
    n_pop = int(strat["weight"].sum())
    rare = load_rare_labels()
    rows, detail = [], {}
    paths = glob.glob(os.path.join(OUTPUT_DIR, "*", "*", "*", "metrics.json")) + [os.path.join(EX7_BEST, "metrics.json")]
    for p in paths:
        d = p.split(os.sep)
        name = "ex007_best(v3)" if p.startswith(EX7_BEST) else f"{d[-4]}/{d[-3]}"
        m = json.load(open(p))
        r = {"run": name, "seed": d[-2]}
        run_dir = os.path.dirname(p)
        if name.startswith("ex007"):
            ann = pl.read_parquet(os.path.join(run_dir, "real_annotation_v3.parquet"))
            pred_all = ann.filter(pl.col("fileId").is_in(ids)).select(["fileId", "format"] + ATTRIBUTES + MEDIA)
        elif name.startswith("teacher"):
            pred_all = pl.read_csv(os.path.join(run_dir, "check_predictions.csv"))
        else:
            pred_all = pl.concat([pl.DataFrame({"fileId": ids}),
                                  to_frame(dict(np.load(os.path.join(run_dir, "check_probs.npz"))))], how="horizontal")
            if "rare_dev" not in m:
                m.update(rare_scores(to_frame(dict(np.load(os.path.join(run_dir, "rare_probs.npz")))), rare))
        if not name.startswith("ex007"):
            for s in ["dev", "test"]:
                for k in ["format_macro_f1", "attr_macro_f1", "media_macro_f1", "overall_macro_f1", "overall_robust_f1"]:
                    r[f"{s}_{k}"] = m[s][k]
                r[f"rare_{s}"] = m[f"rare_{s}"]["macro_f1"]
        cs = check_scores(pred_all, rnd, strat, n_pop)
        for k, v in cs.items():
            for kk in ["format_accuracy", "format_macro_f1", "attr_macro_f1", "media_macro_f1", "overall_macro_f1",
                       "all_correct", "media_multi_rate"]:
                r[f"{k}_{kk}"] = v[kk]
        detail.setdefault(name, []).append(cs)
        rows.append(r)
    return pl.DataFrame(rows), detail


def plot(g: pl.DataFrame, detail: dict) -> None:
    """集計の図を作る．

    Args:
        g (pl.DataFrame): 条件ごとの平均の表．
        detail (dict): 実行ごとの実データの集計．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    keys = [("test_overall_macro_f1", "通常 test：3軸の平均"), ("rare_test", "少ない区分 test"),
            ("random_overall_macro_f1", "実データ 無作為160：3軸の平均"),
            ("strat_uniform_overall_macro_f1", "実データ 層別220：3軸の平均"),
            ("pooled_ipw_all_correct", "実データ 推定：全一致の割合")]
    g = g.filter(pl.col("run").is_in(SUMMARY_RUNS))
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(g) + 1.5))
    for i, (k, lab) in enumerate(keys):
        ax.barh(y + (i - 2) * 0.17, g[k].fill_null(0), 0.17, label=lab)
    ax.set_yticks(y, g["run"].to_list(), fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("F1 または割合（生徒は3 seed の平均）")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_summary.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    items = FORMATS + ATTRIBUTES + MEDIA
    names = [n for n in KEY_RUNS if n in detail]
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = np.mean([[d["strat_uniform"]["per_item"][c] for c in items] for d in detail[n]], axis=0)
        ax.bar(np.arange(len(items)) + (i - len(names) / 2 + 0.5) * width, vals, width, label=n)
    sup = detail[names[0]][0]["strat_uniform"]["support"]
    ax.set_xticks(np.arange(len(items)), [f"{c}\n(n={sup[c]})" for c in items], fontsize=7, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1（層別抽出220枚，重みなし）")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_strat_items.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def main() -> None:
    """集計と図の作成を行う．

    Returns:
        None
    """
    os.makedirs(FIG, exist_ok=True)
    df, detail = collect()
    cols = [c for c in df.columns if c not in ("run", "seed")]
    g = df.group_by("run").agg([pl.len().alias("n")] + [pl.col(c).mean() for c in cols]).sort(
        "pooled_ipw_all_correct", descending=True)
    g.write_csv(os.path.join(FIG, "summary.csv"))
    json.dump(detail, open(os.path.join(FIG, "real_check.json"), "w"), indent=1, ensure_ascii=False)
    plot(g.sort("pooled_ipw_all_correct"), detail)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, tbl_rows=50, float_precision=3):
        print(g.select(["run", "n"] + [c for c in g.columns if c.startswith(("dev_overall", "test_overall_macro", "rare_test",
                                                                             "random_", "strat_uniform_", "pooled_ipw_"))
                                       and ("overall" in c or "all_correct" in c or "rare" in c or "format_acc" in c)]))


if __name__ == "__main__":
    main()
