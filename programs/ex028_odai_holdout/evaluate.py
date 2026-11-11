"""新しい評価用データ（ex028 の600枚）で，v5.3.0 単体と後段（ex027 の組み合わせ）のお題文ありの精度を比べる．

評価の内容：
1. 手法ごとのお題文ありの適合率・再現率・F1（全データの推定．層の重み N_h / n_h）と，95% 区間・v5.3.0 との F1 の差の95% 区間
   （同じ標本の対で層ごとに1000回の復元抽出）．乱数の種ごとの値と，5つの種の確率の平均（ens）．
   補足として，v5.3.0 の確率が低い層 D・F を除いた部分の推定（excl_DF）も求める．
2. ex027 の1,620枚の交差検証での推定（ex027 の odai_summary.csv）との比較．
3. 正解の種類ごとの誤りの割合（全データの推定．ex027 の error_group）と，層ごとの正解率．
4. 判断のずれの確認：ex026 の1,620枚から再び目視した36枚の，記号とお題文あり（R1b）の一致．

出力：outputs/ex028_odai_holdout/eval/odai.csv，odai_summary.csv，compare_ex027.csv，error_breakdown.csv，strata.csv，repeat.csv，
      fig_holdout.png，fig_errors.png，evaluate.log

実行例:
    .venv_annotate/bin/python programs/ex028_odai_holdout/evaluate.py
"""
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX026, OUTPUT_DIR, PROJECT_DIR, load_all, load_ex027_module  # noqa: E402
from train import METHODS, PRED_DIR, SEEDS, TRUTH  # noqa: E402

plt.rcParams["font.family"] = ["IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "DejaVu Sans"]
E27 = load_ex027_module("evaluate")
EX026_EVAL = E27.EX026_EVAL
OUT = os.path.join(OUTPUT_DIR, "eval")
EX027_SUMMARY = os.path.join(PROJECT_DIR, "outputs", "ex027_odai_text_improve", "eval", "odai_summary.csv")
SETS = {"population": lambda d: np.ones(d.height, bool),
        "excl_DF": lambda d: ~d["stratum2"].is_in(["D_blank_p_lo", "F_normal_p_lo"]).to_numpy()}
NAMES = {"v5.3.0": "v5.3.0 単体", "pkg+emb": "後段：確率＋埋め込み", "pkg+ocr32b": "後段：確率＋32B の書き起こし"}


def load_preds(ids: list) -> dict:
    """後段の予測を読み込む．

    Args:
        ids (list[int]): 評価の画像のお題ID．

    Returns:
        dict[tuple[str, str], np.ndarray]: (手法, 種（"0"〜"4" と，確率の平均の "ens"）) ごとの形状 (N,) の予測（bool）．
    """
    out = {}
    base = pl.DataFrame({"rep_odaiId": ids})
    for method in METHODS:
        probs = []
        for seed in SEEDS:
            p = base.join(pl.read_csv(os.path.join(PRED_DIR, method, TRUTH, str(seed), "pred.csv")), on="rep_odaiId", how="left")["prob"].to_numpy()
            out[(method, str(seed))] = p >= 0.5
            probs.append(p)
        out[(method, "ens")] = np.mean(probs, axis=0) >= 0.5
    return out


def eval_odai(d: pl.DataFrame, preds: dict) -> pl.DataFrame:
    """手法・種ごとの精度と区間を求める．

    集合：population（全データの推定），excl_DF（全データのうち層 D と F（v5.3.0 の確率が低い穴埋め・通常）を除いた部分の推定．補足）．

    Args:
        d (pl.DataFrame): 新しい評価用データ（600枚）の表．
        preds (dict): load_preds の戻り値．

    Returns:
        pl.DataFrame: 列 set, method, seed, precision, recall, f1, lo, hi, d_lo, d_hi, f1_unweighted, n_pos_pred を持つ表．
    """
    base_all = d["has_odai_text"].to_numpy()
    rows = []
    for name, keep in SETS.items():
        k = keep(d)
        y, w, st, base = d["odai_R1b"].to_numpy()[k], d["w"].to_numpy()[k], d["stratum2"].to_numpy()[k], base_all[k]
        for (method, seed), p in [(("v5.3.0", "-"), base_all), *preds.items()]:
            p = p[k]
            ci = E27.boot_pair(y, p, base, w, st)
            rows.append({"set": name, "method": method, "seed": seed, **EX026_EVAL.prf(y, p, w), "lo": ci[0], "hi": ci[1], "d_lo": ci[2],
                         "d_hi": ci[3], "f1_unweighted": EX026_EVAL.prf(y, p)["f1"], "n_pos_pred": int(p.sum())})
    return pl.DataFrame(rows)


def summarize(od: pl.DataFrame) -> pl.DataFrame:
    """乱数の種ごとの値の平均と標準偏差，差の区間が0を超えた種の数をまとめる．

    Args:
        od (pl.DataFrame): eval_odai の表．

    Returns:
        pl.DataFrame: 列 set, method, precision, recall, f1, f1_sd, d_lo_min, d_hi_max, n_seed_sig, ens_f1, ens_lo, ens_hi, ens_d_lo, ens_d_hi を持つ表．
    """
    s = od.filter(pl.col("seed") != "ens").group_by("set", "method", maintain_order=True).agg(
        pl.col("precision", "recall", "f1").mean(), pl.col("f1").std().alias("f1_sd"), pl.col("d_lo").min().alias("d_lo_min"),
        pl.col("d_hi").max().alias("d_hi_max"), (pl.col("d_lo") > 0).sum().alias("n_seed_sig"))
    e = od.filter(pl.col("seed").is_in(["ens", "-"])).select("set", "method", pl.col("f1").alias("ens_f1"), pl.col("lo").alias("ens_lo"),
                                                             pl.col("hi").alias("ens_hi"), pl.col("d_lo").alias("ens_d_lo"),
                                                             pl.col("d_hi").alias("ens_d_hi"))
    return s.join(e, on=["set", "method"], how="left")


def compare_ex027(s: pl.DataFrame) -> pl.DataFrame:
    """ex027 の1,620枚の交差検証での推定（v6 の正解，全データの推定）と並べる．

    Args:
        s (pl.DataFrame): summarize の表．

    Returns:
        pl.DataFrame: 列 method, f1_ex027, d_lo_ex027, d_hi_ex027, f1_holdout, ens_d_lo, ens_d_hi を持つ表．
    """
    e = pl.read_csv(EX027_SUMMARY).filter((pl.col("truth") == "v6") & (pl.col("set") == "population"))
    e = e.select("method", pl.col("f1").alias("f1_ex027"), pl.col("d_lo").alias("d_lo_ex027"), pl.col("d_hi").alias("d_hi_ex027"))
    s = s.filter(pl.col("set") == "population")
    return s.select("method", pl.col("f1").alias("f1_holdout"), "ens_d_lo", "ens_d_hi").join(e, on="method", how="left")


def error_breakdown(d: pl.DataFrame, preds: dict) -> pl.DataFrame:
    """正解の種類ごとの誤りの割合（全データの推定）を求める．

    Args:
        d (pl.DataFrame): 新しい評価用データの表．
        preds (dict): load_preds の戻り値．

    Returns:
        pl.DataFrame: 列 group, n, share（全データでの割合）, method, error（正例は見逃し，負例は誤検出の割合）を持つ表．
    """
    g = np.array([E27.error_group(r) for r in d.iter_rows(named=True)])
    w, y = d["w"].to_numpy(), d["odai_R1b"].to_numpy()
    rows = []
    for method, p in [("v5.3.0", d["has_odai_text"].to_numpy()), *[(m, preds[(m, "ens")]) for m in METHODS]]:
        for k in np.unique(g):
            i = g == k
            err = (p[i] != y[i]).astype(float)
            rows.append({"group": str(k), "n": int(i.sum()), "share": float(w[i].sum() / w.sum()), "method": method,
                         "error": float((w[i] * err).sum() / w[i].sum())})
    return pl.DataFrame(rows)


def strata_table(d: pl.DataFrame, preds: dict) -> pl.DataFrame:
    """層ごとのお題文ありの割合と正解率を求める．

    Args:
        d (pl.DataFrame): 新しい評価用データの表．
        preds (dict): load_preds の戻り値．

    Returns:
        pl.DataFrame: 列 stratum2, N_h, n_h, odai_rate, acc_v5.3.0, acc_{手法} を持つ表．
    """
    cols = {"acc_v5.3.0": d["has_odai_text"].to_numpy()} | {f"acc_{m}": preds[(m, "ens")] for m in METHODS}
    t = d.with_columns(*[pl.Series(k, v == d["odai_R1b"].to_numpy()) for k, v in cols.items()])
    return t.group_by("stratum2").agg(pl.col("N_h").first(), pl.len().alias("n_h"), pl.col("odai_R1b").mean().alias("odai_rate"),
                                      *[pl.col(k).mean() for k in cols]).sort("stratum2")


def eval_repeat(rep: pl.DataFrame) -> tuple:
    """判断のずれの確認の36枚で，ex026 の目視との一致を求める．

    Args:
        rep (pl.DataFrame): load_all の kind が repeat の行．

    Returns:
        tuple[pl.DataFrame, dict]: 1枚ごとの表（列 rep_odaiId, code, code_ex026, odai_R1b, odai_ex026）と，一致の要約．
    """
    old = EX026.load_all().select("rep_odaiId", pl.col("code").alias("code_ex026"), pl.col("odai_R1b").alias("odai_ex026"))
    t = rep.select("rep_odaiId", "code", "odai_R1b").join(old, on="rep_odaiId", how="left")
    a, b = t["odai_R1b"].to_numpy(), t["odai_ex026"].to_numpy()
    summary = {"n": t.height, "code_match": float((t["code"] == t["code_ex026"]).mean()), "odai_match": float((a == b).mean()),
               "odai_kappa": EX026_EVAL.kappa(a, b), "odai_rate_now": float(a.mean()), "odai_rate_ex026": float(b.mean())}
    return t, summary


def plot_holdout(cmp: pl.DataFrame, s: pl.DataFrame, out: str) -> None:
    """ex027 の推定と新しい評価用データの F1（95% 区間．層 D・F を除いた推定も）を手法ごとに描く．

    Args:
        cmp (pl.DataFrame): compare_ex027 の表．
        s (pl.DataFrame): summarize の表．
        out (str): 保存先．

    Returns:
        None
    """
    order = list(NAMES)
    ex = s.filter(pl.col("set") == "excl_DF").select("method", pl.col("ens_f1").alias("x_f1"))
    s = s.filter(pl.col("set") == "population").select("method", "ens_f1", "ens_lo", "ens_hi")
    t = pl.DataFrame({"method": order}).join(cmp, on="method", how="left").join(s, on="method").join(ex, on="method")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    yy = np.arange(len(order))[::-1]
    ax.scatter(t["x_f1"], yy - 0.12, marker="^", s=40, color="#1baf7a", label="ex028（層 D・F を除く．補足）", zorder=3)
    ax.scatter(t["f1_ex027"], yy + 0.12, marker="s", s=40, color="#999999", label="ex027（1,620枚の交差検証）", zorder=3)
    ax.errorbar(t["ens_f1"], yy - 0.12, xerr=[t["ens_f1"] - t["ens_lo"], t["ens_hi"] - t["ens_f1"]], fmt="o", color="#2a78d6",
                ecolor="#9cc3ef", elinewidth=2, label="ex028（新しい600枚）", zorder=3)
    for y0, f in zip(yy, t["ens_f1"]):
        ax.text(f, y0 - 0.42, f"{f:.3f}", ha="center", fontsize=8, color="#2a78d6")
    for y0, f in zip(yy, t["f1_ex027"]):
        ax.text(f, y0 + 0.3, f"{f:.3f}", ha="center", fontsize=8, color="#666666")
    ax.set_yticks(yy)
    ax.set_yticklabels([NAMES[m] for m in order], fontsize=9)
    ax.set_xlabel("お題文ありの F1（全データの推定，正解：v6 の R1b）")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_errors(eb: pl.DataFrame, out: str) -> None:
    """正解の種類ごとの誤りの割合を手法ごとに描く．

    Args:
        eb (pl.DataFrame): error_breakdown の表．
        out (str): 保存先．

    Returns:
        None
    """
    groups = eb.filter(pl.col("method") == "v5.3.0").sort("share", descending=True)["group"].to_list()
    methods = list(NAMES)
    colors = ["#888888", "#2a78d6", "#eb6834"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(groups))
    for k, (m, c) in enumerate(zip(methods, colors)):
        v = [eb.filter((pl.col("method") == m) & (pl.col("group") == g))["error"][0] for g in groups]
        ax.bar(x + (k - 1) * 0.27, v, width=0.25, color=c, label=NAMES[m])
    share = {r["group"]: r["share"] for r in eb.filter(pl.col("method") == "v5.3.0").iter_rows(named=True)}
    labels = [g.replace("：", "：\n") + f"\n（{share[g] * 100:.1f}%）" for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("誤りの割合（正例は見逃し，負例は誤検出）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """評価を行い，表と図を保存する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    allt = load_all()
    d = allt.filter(pl.col("kind") == "new")
    assert d["has_odai_text"].null_count() == 0, "v5.3.0 の予測が欠けている"
    preds = {k: v for k, v in load_preds(allt["rep_odaiId"].to_list()).items()}
    is_new = (allt["kind"] == "new").to_numpy()
    preds = {k: v[is_new] for k, v in preds.items()}
    od = eval_odai(d, preds)
    od.write_csv(os.path.join(OUT, "odai.csv"))
    s = summarize(od)
    s.write_csv(os.path.join(OUT, "odai_summary.csv"))
    cmp = compare_ex027(s)
    cmp.write_csv(os.path.join(OUT, "compare_ex027.csv"))
    eb = error_breakdown(d, preds)
    eb.write_csv(os.path.join(OUT, "error_breakdown.csv"))
    st = strata_table(d, preds)
    st.write_csv(os.path.join(OUT, "strata.csv"))
    rep, rs = eval_repeat(allt.filter(pl.col("kind") == "repeat"))
    rep.write_csv(os.path.join(OUT, "repeat.csv"))
    plot_holdout(cmp, s, os.path.join(OUT, "fig_holdout.png"))
    plot_errors(eb, os.path.join(OUT, "fig_errors.png"))
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(40)
    print(f"お題文ありの割合（全データの推定）：{float((d['w'] * d['odai_R1b']).sum() / d['w'].sum()):.3f}，600枚：{d['odai_R1b'].mean():.3f}")
    print(od)
    print(s)
    print(cmp)
    print(eb.pivot(on="method", index=["group", "n", "share"], values="error"))
    print(st)
    print("判断のずれ：", rs)
    print(rep.filter(pl.col("code") != pl.col("code_ex026")))


if __name__ == "__main__":
    main()
