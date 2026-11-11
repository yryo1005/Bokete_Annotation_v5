"""train.py の分割外の予測を評価し，v5.3.0 と比べる．

評価の内容：
1. 手法ごとのお題文ありの適合率・再現率・F1（全データの推定と1,620枚）．乱数の種ごとの値の平均と標準偏差．
2. 全データの推定の F1 の95% 区間と，v5.3.0 との F1 の差の95% 区間（seed 0．同じ標本の対で層ごとに復元抽出）．
3. 文の形を問う教師（FORM，FORM2）の形ごとの精度を，ex026 の V6 の教師（層をまとめた形）と比べる．

4. 正解の種類（例：余白のその他の文のみ，空欄を含む文）ごとの誤りの割合（全データの推定）．

出力：outputs/ex027_odai_text_improve/eval/odai.csv，odai_summary.csv，teacher_forms.csv，error_breakdown.csv，fig_odai_{v6,v5}.png，fig_errors.png

実行例:
    .venv_annotate/bin/python programs/ex027_odai_text_improve/evaluate.py
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX026_EVAL, FORMS, OUTPUT_DIR, load_eval  # noqa: E402
from train import PRED_DIR, TEACHERS, TRUTHS, load_form  # noqa: E402

plt.rcParams["font.family"] = ["IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "DejaVu Sans"]
OUT = os.path.join(OUTPUT_DIR, "eval")
N_BOOT = 1000
NAMES = {"v5.3.0": "v5.3.0（既存）", "p_only": "v5.3.0 の確率の較正", "pkg": "v5.3.0 の確率（15種）", "pkg+emb": "確率＋埋め込み",
         "pkg+yomi": "確率＋YomiToku", "pkg+ocr32b": "確率＋32B の書き起こし", "pkg+emb+yomi": "確率＋埋め込み＋YomiToku",
         "pkg+emb+ocr32b": "確率＋埋め込み＋32B の書き起こし", "emb+yomi": "埋め込み＋YomiToku（確率なし）",
         **{k: v for p, lab in [("form", "FORM"), ("form2", "FORM2")] for k, v in {
             f"{p}_teacher": f"{lab} の教師＋余白の生徒", f"{p}_teacher_own": f"{lab} の教師（教師の場所）",
             f"{p}_student": f"{lab} の生徒（プール）＋余白の生徒", f"{p}_student_direct": f"{lab} の生徒（プール）の直接学習",
             f"pkg+{p}": f"確率＋{lab} の生徒", f"pkg+emb+yomi+{p}": f"確率＋埋め込み＋YomiToku＋{lab} の生徒"}.items()}}


def boot_pair(y: np.ndarray, p: np.ndarray, base: np.ndarray, w: np.ndarray, strata: np.ndarray) -> tuple:
    """層ごとの復元抽出で，F1 とその基準との差の95% 区間を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の予測．
        base (np.ndarray): 形状 (N,) の基準の予測（v5.3.0）．
        w (np.ndarray): 形状 (N,) の重み．
        strata (np.ndarray): 形状 (N,) の層の名前．

    Returns:
        tuple[float, float, float, float]: F1 の2.5% 点，97.5% 点，差の2.5% 点，97.5% 点．
    """
    rng = np.random.default_rng(0)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    f, df = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        a = EX026_EVAL.prf(y[idx], p[idx], w[idx])["f1"]
        b = EX026_EVAL.prf(y[idx], base[idx], w[idx])["f1"]
        f.append(a)
        df.append(a - b)
    return tuple(float(np.nanpercentile(v, q)) for v in (f, df) for q in (2.5, 97.5))


def load_preds() -> pl.DataFrame:
    """保存した予測を全て読み込む．

    Returns:
        pl.DataFrame: 列 method, truth, seed, rep_odaiId, prob, pred を持つ表．
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "*", "*", "*", "pred.csv"))):
        method, truth, seed = path.split(os.sep)[-4:-1]
        rows.append(pl.read_csv(path).with_columns(pl.lit(method).alias("method"), pl.lit(truth).alias("truth"), pl.lit(int(seed)).alias("seed")))
    return pl.concat(rows)


def eval_odai(d: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    """手法・正解・乱数の種ごとの精度を求める．

    Args:
        d (pl.DataFrame): 評価の1,620枚の表（列 has_odai_text を含む）．
        preds (pl.DataFrame): load_preds の表．

    Returns:
        pl.DataFrame: 列 method, truth, seed, set, precision, recall, f1, lo, hi, d_lo, d_hi を持つ表．
    """
    is25 = (d["source"] == "ex025").to_numpy()
    w25 = d["w"].fill_null(0).to_numpy()[is25]
    st25 = d["stratum2"].fill_null("").to_numpy()[is25]
    base = d["has_odai_text"].to_numpy()
    ids = d.select("rep_odaiId")
    rows = []
    groups = [("v5.3.0", t, 0, base) for t in TRUTHS]
    for (method, truth, seed), g in preds.group_by(["method", "truth", "seed"], maintain_order=True):
        groups.append((method, truth, seed, ids.join(g, on="rep_odaiId", how="left")["pred"].to_numpy()))
    for method, truth, seed, p in groups:
        y = d[TRUTHS[truth]].to_numpy()
        r = EX026_EVAL.prf(y[is25], p[is25], w25)
        ci = boot_pair(y[is25], p[is25], base[is25], w25, st25) if seed == 0 else (float("nan"),) * 4
        rows.append({"method": method, "truth": truth, "seed": seed, "set": "population", **r, "lo": ci[0], "hi": ci[1], "d_lo": ci[2],
                     "d_hi": ci[3]})
        rows.append({"method": method, "truth": truth, "seed": seed, "set": "all_1620", **EX026_EVAL.prf(y, p), "lo": float("nan"),
                     "hi": float("nan"), "d_lo": float("nan"), "d_hi": float("nan")})
    return pl.DataFrame(rows)


def summarize(od: pl.DataFrame) -> pl.DataFrame:
    """乱数の種の平均と標準偏差，seed 0 の区間をまとめる．

    Args:
        od (pl.DataFrame): eval_odai の表．

    Returns:
        pl.DataFrame: 列 method, truth, set, n_seed, precision, recall, f1, f1_sd, lo, hi, d_lo, d_hi を持つ表．
    """
    agg = od.group_by(["method", "truth", "set"], maintain_order=True).agg(
        pl.len().alias("n_seed"), pl.col("precision").mean(), pl.col("recall").mean(), pl.col("f1").mean(), pl.col("f1").std().alias("f1_sd"))
    ci = od.filter(pl.col("seed") == 0).select("method", "truth", "set", "lo", "hi", "d_lo", "d_hi")
    return agg.join(ci, on=["method", "truth", "set"], how="left")


def eval_teacher(d: pl.DataFrame) -> pl.DataFrame:
    """文の形だけを問う教師の形ごとの精度を，V6 の教師の層をまとめた形と比べる（1,620枚，重みなし）．

    Args:
        d (pl.DataFrame): 評価の1,620枚の表（列 A_{形}, q_A_{形} を含む）．

    Returns:
        pl.DataFrame: 列 teacher, form, n_pos, precision, recall, f1 を持つ表．
    """
    t = d
    names = [("V6", "q_A_")]
    for prompt in TEACHERS:
        fe = load_form(prompt, "eval")
        if fe is not None:
            t = t.join(fe.select("rep_odaiId", *[pl.col(f"T_{f}").alias(f"{prompt}_A_{f}") for f in FORMS]), on="rep_odaiId", how="left")
            names.append((prompt, f"{prompt}_A_"))
    rows = []
    for f in FORMS:
        y = t[f"A_{f}"].to_numpy()
        for name, pre in names:
            rows.append({"teacher": name, "form": f, **EX026_EVAL.prf(y, t[pre + f].to_numpy())})
    return pl.DataFrame(rows)


def error_group(r: dict) -> str:
    """目視の属性から，お題文あり（R1b）の正例・負例の種類を決める．

    Args:
        r (dict): 評価の表の1行（列 F_o_*, F_m_*, L_*, odai_R1b を含む）．

    Returns:
        str: 種類の名前．
    """
    task = lambda l: any(r[f"F_{l}_{f}"] for f in "qit")  # noqa: E731
    if r["odai_R1b"]:
        return "正：余白の問い・指示・題目" if task("m") else "正：重ねた問い・指示・題目" if task("o") else "正：余白のその他の文のみ"
    if r["F_m_k"] or r["F_o_k"]:
        return "負：空欄を含む文"
    if r["F_o_x"] or r["F_m_x"]:
        return "負：重ねた文字のその他の文"
    return "負：場面・吹き出しのみ" if r["L_s"] or r["L_b"] else "負：文字なし"


def error_breakdown(d: pl.DataFrame, preds: pl.DataFrame, methods: list) -> pl.DataFrame:
    """正解（v6 の R1b）の種類ごとに，誤りの割合（全データの推定）を手法ごとに求める（seed 0）．

    Args:
        d (pl.DataFrame): 評価の1,620枚の表．
        preds (pl.DataFrame): load_preds の表．
        methods (list[str]): 比べる手法（v5.3.0 を含めてよい）．

    Returns:
        pl.DataFrame: 列 group, rate（その種類の画像の割合），{手法}（その種類の中の誤りの割合）を持つ表．
    """
    t = d.filter(pl.col("source") == "ex025")
    t = t.with_columns(pl.Series("group", [error_group(r) for r in t.iter_rows(named=True)]), pl.col("has_odai_text").alias("v5.3.0"))
    for m in methods:
        if m == "v5.3.0":
            continue
        g = preds.filter((pl.col("method") == m) & (pl.col("truth") == "v6") & (pl.col("seed") == 0)).select("rep_odaiId", pl.col("pred").alias(m))
        t = t.join(g, on="rep_odaiId", how="left")
    w_all = t["w"].sum()
    agg = [(pl.col("w").sum() / w_all).alias("rate")] + [((pl.col("w") * (pl.col(m) != pl.col("odai_R1b"))).sum() / pl.col("w").sum()).alias(m)
                                                         for m in methods]
    return t.group_by("group").agg(pl.len().alias("n"), *agg).sort("group")


def plot_errors(eb: pl.DataFrame, out: str) -> None:
    """正解の種類ごとの誤りの割合を手法ごとに描く（文字なしの種類は誤りがないため除く）．

    Args:
        eb (pl.DataFrame): error_breakdown の表．
        out (str): 保存先．

    Returns:
        None
    """
    t = eb.filter(pl.col("group") != "負：文字なし")
    methods = [c for c in eb.columns if c not in ("group", "n", "rate")]
    colors = ["#888888", "#2a78d6", "#1baf7a", "#eda100", "#eb6834"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(methods)
    x = np.arange(t.height)
    for k, (m, col) in enumerate(zip(methods, colors)):
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, t[m], width, label=NAMES[m], color=col)
    ax.set_xticks(x)
    labels = [g.replace("：", "：\n") + f"\n（{r * 100:.1f}%）" for g, r in zip(t["group"], t["rate"])]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("種類の中の誤りの割合（全データの推定）")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot(s: pl.DataFrame, truth: str, out: str) -> None:
    """全データの推定の F1 と95% 区間を手法ごとに描く（乱数の種の平均を点，seed 0 の区間を線）．

    Args:
        s (pl.DataFrame): summarize の表．
        truth (str): 正解の名前（v6 / v5）．
        out (str): 保存先．

    Returns:
        None
    """
    t = s.filter((pl.col("truth") == truth) & (pl.col("set") == "population"))
    order = [m for m in NAMES if t.filter(pl.col("method") == m).height > 0]
    t = pl.DataFrame({"method": order}).join(t, on="method", how="left")
    base = t.filter(pl.col("method") == "v5.3.0")["f1"][0]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.2))
    yy = np.arange(len(order))[::-1]
    colors = ["#2a78d6" if m.startswith(("pkg", "p_only")) else "#eb6834" if m.startswith("form") else "#888888" for m in order]
    colors = ["#1baf7a" if m.startswith("pkg+") and m.endswith(("form", "form2")) else c for m, c in zip(order, colors)]
    ax.errorbar(t["f1"], yy, xerr=[t["f1"] - t["lo"], t["hi"] - t["f1"]], fmt="none", ecolor="#bbbbbb", elinewidth=2)
    ax.scatter(t["f1"], yy, c=colors, s=40, zorder=3)
    ax.axvline(base, color="#444444", ls="--", lw=1)
    for y0, f in zip(yy, t["f1"]):
        ax.text(f, y0 + 0.25, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_yticks(yy)
    ax.set_yticklabels([NAMES[m] for m in order], fontsize=9)
    ax.set_xlabel(f"お題文ありの F1（全データの推定，正解：{'v6 の R1b' if truth == 'v6' else 'v5'}）")
    ax.set_xlim(0.3, 1.0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """評価を行い，表と図を保存する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    d, _ = load_eval()
    od = eval_odai(d, load_preds())
    od.write_csv(os.path.join(OUT, "odai.csv"))
    s = summarize(od)
    s.write_csv(os.path.join(OUT, "odai_summary.csv"))
    for truth in TRUTHS:
        plot(s, truth, os.path.join(OUT, f"fig_odai_{truth}.png"))
    preds = load_preds()
    eb = error_breakdown(d, preds, ["v5.3.0", "pkg+emb", "pkg+ocr32b", "pkg+emb+yomi", "form2_teacher"])
    eb.write_csv(os.path.join(OUT, "error_breakdown.csv"))
    plot_errors(eb, os.path.join(OUT, "fig_errors.png"))
    print(eb)
    te = eval_teacher(d)
    if te.height:
        te.write_csv(os.path.join(OUT, "teacher_forms.csv"))
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(30)
    print(s.filter(pl.col("set") == "population").sort("truth", "f1", descending=[False, True]).select(
        "method", "truth", "n_seed", "precision", "recall", "f1", "f1_sd", "lo", "hi", "d_lo", "d_hi"))
    print(te)
    metas = {p.split(os.sep)[-4]: json.load(open(p)) for p in glob.glob(os.path.join(PRED_DIR, "form*_student", "v6", "0", "meta.json"))}
    print(json.dumps(metas, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
