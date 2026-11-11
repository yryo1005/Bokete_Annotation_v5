"""層 D・F の追加の抽出（order_049）を加えて，v5.3.0 の見逃しと全データの推定を求め直す．

評価の内容：
1. 層 D・F のお題文ありの割合（Wilson の95% 区間）と種類，v5.3.0 と後段の予測．
   抽出の出所は ex025（ex026 の目視．後段の学習に含まれる），ex028（600枚の一部），追加（各300枚）の3つ．
2. 全データの推定（combined）：層 D・F は ex028 と追加の抽出をまとめ（後段の学習に含まれない画像のみ），
   他の層は ex028 の600枚を用いる．重みは層の枚数 / 層の抽出数．お題文ありの割合，適合率・再現率・F1 と，
   v5.3.0 との F1 の差の95% 区間（層ごとの復元抽出．後段は5つの種の確率の平均（ens））．
3. 参考：v5.3.0 単体について，層 D・F に ex025 の画像も加えた推定（combined_v53）．

出力：outputs/ex028_odai_holdout/eval_df/（df_rates.csv，df_positives.csv，odai_combined.csv，fig_df.png，evaluate_df.log）

実行例:
    .venv_annotate/bin/python programs/ex028_odai_holdout/evaluate_df.py
"""
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, DF_STRATA, EX026, OUTPUT_DIR, _load, load_all, load_df, load_package_probs  # noqa: E402
from train import METHODS, PRED_DIR, SEEDS, TRUTH  # noqa: E402

# 名前が同じ ex025 の evaluate と区別するため，ファイルから別名で読み込む
EV = _load("ex028_evaluate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
E27, EX026_EVAL, NAMES = EV.E27, EV.EX026_EVAL, EV.NAMES
plt.rcParams["font.family"] = ["IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "DejaVu Sans"]
OUT = os.path.join(OUTPUT_DIR, "eval_df")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """二項の割合の Wilson の95% 区間を求める．

    Args:
        k (int): 成功の数．
        n (int): 試行の数．
        z (float): 正規分布の分位点．

    Returns:
        tuple[float, float]: 下限と上限．
    """
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def load_post(ids: list) -> dict:
    """後段の5つの種の確率の平均（ens）による予測を読み込む（ex028 の600枚と追加の抽出）．

    Args:
        ids (list[int]): 代表の画像のお題ID．

    Returns:
        dict[str, np.ndarray]: 手法ごとの形状 (N,) の予測（bool）．
    """
    base = pl.DataFrame({"rep_odaiId": ids})
    out = {}
    for method in METHODS:
        probs = []
        for seed in SEEDS:
            d = os.path.join(PRED_DIR, method, TRUTH, str(seed))
            p = pl.concat([pl.read_csv(os.path.join(d, f)) for f in ("pred.csv", "pred_df.csv")]).unique("rep_odaiId")
            probs.append(base.join(p, on="rep_odaiId", how="left")["prob"].to_numpy())
        out[method] = np.mean(probs, axis=0) >= 0.5
    return out


def build_pool() -> pl.DataFrame:
    """3つの出所の画像を1つの表にまとめる．

    Returns:
        pl.DataFrame: 列 rep_odaiId, source, stratum2, code, memo, odai_R1b, has_odai_text, p_has_odai_text と L_*, F_* を持つ表．
    """
    cols = ["rep_odaiId", "stratum2", "code", "memo", "odai_R1b", "has_odai_text", "p_has_odai_text"]
    e28 = load_all().filter(pl.col("kind") == "new")
    e28 = e28.select(*cols, *[c for c in e28.columns if c.startswith(("L_", "F_"))], pl.lit("ex028").alias("source"))
    df = load_df()
    df = df.select(*cols, *[c for c in df.columns if c.startswith(("L_", "F_"))], pl.lit("df").alias("source"))
    e25 = EX026.load_all().filter(pl.col("source") == "ex025")
    probs = load_package_probs(e25["rep_odaiId"].to_list()).drop("odaiId")
    e25 = e25.with_columns(*[probs[c] for c in ["has_odai_text", "p_has_odai_text"]])
    e25 = e25.select(*cols, *[c for c in e25.columns if c.startswith(("L_", "F_"))], pl.lit("ex025").alias("source"))
    return pl.concat([e28, df, e25], how="diagonal_relaxed")


def df_rates(pool: pl.DataFrame) -> pl.DataFrame:
    """層 D・F のお題文ありの割合と v5.3.0 の見逃しを，出所ごととまとめた値で求める．

    Args:
        pool (pl.DataFrame): build_pool の表．

    Returns:
        pl.DataFrame: 列 stratum2, source, n, n_pos, rate, lo, hi, v53_tp, v53_fp を持つ表．
    """
    rows = []
    d = pool.filter(pl.col("stratum2").is_in(DF_STRATA))
    groups = [(s, src) for s in DF_STRATA for src in ["ex025", "ex028", "df", "ex028+df", "all"]]
    for st, src in groups:
        srcs = {"ex028+df": ["ex028", "df"], "all": ["ex025", "ex028", "df"]}.get(src, [src])
        t = d.filter((pl.col("stratum2") == st) & pl.col("source").is_in(srcs))
        k, n = int(t["odai_R1b"].sum()), t.height
        lo, hi = wilson(k, n)
        rows.append({"stratum2": st, "source": src, "n": n, "n_pos": k, "rate": k / n, "lo": lo, "hi": hi,
                     "v53_tp": int((t["odai_R1b"] & t["has_odai_text"]).sum()), "v53_fp": int((~t["odai_R1b"] & t["has_odai_text"]).sum())})
    return pl.DataFrame(rows)


def combined_eval(pool: pl.DataFrame, n_pop: dict, sources_df: list, methods: dict) -> pl.DataFrame:
    """層 D・F に追加の抽出を加えた全データの推定を求める．

    Args:
        pool (pl.DataFrame): build_pool の表．
        n_pop (dict[str, int]): 層ごとの代表の画像の数．
        sources_df (list[str]): 層 D・F に用いる出所．
        methods (dict[str, np.ndarray]): 手法ごとの pool の行の予測（v5.3.0 を含む）．

    Returns:
        pl.DataFrame: 列 method, odai_rate, precision, recall, f1, lo, hi, d_lo, d_hi, n を持つ表．
    """
    is_df = pool["stratum2"].is_in(DF_STRATA).to_numpy()
    src = pool["source"].to_numpy()
    keep = np.where(is_df, np.isin(src, sources_df), src == "ex028")
    d = pool.filter(pl.Series(keep))
    n_h = d.group_by("stratum2").len()
    d = d.join(n_h, on="stratum2").with_columns((pl.col("stratum2").replace_strict(n_pop) / pl.col("len")).alias("w"))
    y, w, st = d["odai_R1b"].to_numpy(), d["w"].to_numpy(), d["stratum2"].to_numpy()
    base = methods["v5.3.0"][keep]
    rows = []
    for m, p in methods.items():
        p = p[keep]
        ci = E27.boot_pair(y, p, base, w, st)
        rows.append({"method": m, "odai_rate": float((w * y).sum() / w.sum()), **EX026_EVAL.prf(y, p, w), "lo": ci[0], "hi": ci[1],
                     "d_lo": ci[2], "d_hi": ci[3], "n": int(keep.sum())})
    return pl.DataFrame(rows)


def plot_df(r: pl.DataFrame, comb: pl.DataFrame, out: str) -> None:
    """層 D・F のお題文ありの割合（出所ごと）と，全データの F1 の推定（ex028 のみと追加後）を描く．

    Args:
        r (pl.DataFrame): df_rates の表．
        comb (pl.DataFrame): 列 set, method, f1, lo, hi を持つ表（set は ex028 / combined）．
        out (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    srcs = ["ex025", "ex028", "df", "all"]
    labels = {"ex025": "ex025", "ex028": "ex028", "df": "追加", "all": "全て"}
    colors = {"D_blank_p_lo": "#2a78d6", "F_normal_p_lo": "#eb6834"}
    for k, st in enumerate(DF_STRATA):
        t = pl.DataFrame({"source": srcs}).join(r.filter(pl.col("stratum2") == st), on="source", how="left")
        x = np.arange(len(srcs)) + (k - 0.5) * 0.3
        ax.errorbar(x, t["rate"] * 100, yerr=[(t["rate"] - t["lo"]) * 100, (t["hi"] - t["rate"]) * 100], fmt="o", color=colors[st],
                    capsize=3, label=f"層 {st[0]}")
        for xi, (kk, nn) in zip(x, zip(t["n_pos"], t["n"])):
            ax.text(xi, -0.9, f"{kk}/{nn}", ha="center", fontsize=7, color=colors[st])
    ax.set_xticks(np.arange(len(srcs)))
    ax.set_xticklabels([labels[s] for s in srcs])
    ax.set_ylabel("お題文ありの割合（%，95% 区間）")
    ax.set_ylim(-1.5, 15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("層 D・F のお題文あり（v5.3.0 は全て見逃し）", fontsize=10)
    ax = axes[1]
    order = list(NAMES)
    yy = np.arange(len(order))[::-1]
    for k, (s, c, mk) in enumerate([("ex028", "#999999", "s"), ("combined", "#2a78d6", "o")]):
        t = pl.DataFrame({"method": order}).join(comb.filter(pl.col("set") == s), on="method", how="left")
        ax.errorbar(t["f1"], yy + (0.12 if k == 0 else -0.12), xerr=[t["f1"] - t["lo"], t["hi"] - t["f1"]], fmt=mk, color=c, ecolor=c,
                    alpha=0.9, elinewidth=1.5, capsize=2, label="ex028 の600枚" if k == 0 else "層 D・F を追加")
        for y0, f in zip(yy, t["f1"]):
            ax.text(f, y0 + (0.3 if k == 0 else -0.42), f"{f:.3f}", ha="center", fontsize=8, color=c)
    ax.set_yticks(yy)
    ax.set_yticklabels([NAMES[m] for m in order], fontsize=9)
    ax.set_xlabel("お題文ありの F1（全データの推定，正解：v6 の R1b）")
    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """評価を行い，表と図を保存する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    pool = build_pool()
    n_pop = dict(pl.read_csv(os.path.join(DATASET_DIR, "population_strata.csv")).select("stratum2", "N").iter_rows())
    r = df_rates(pool)
    r.write_csv(os.path.join(OUT, "df_rates.csv"))
    post = load_post(pool["rep_odaiId"].to_list())
    is_e25 = (pool["source"] == "ex025").to_numpy()
    methods = {"v5.3.0": pool["has_odai_text"].to_numpy()} | {m: np.where(is_e25, False, p) for m, p in post.items()}
    df_mask = pool["stratum2"].is_in(DF_STRATA).to_numpy()
    flags = pool.with_columns(*[pl.Series(f"pred_{m}", p) for m, p in methods.items()]).filter(pl.Series(df_mask))
    fp_post = flags.filter(pl.col("source") != "ex025").group_by("stratum2").agg(
        pl.len(), *[(pl.col(f"pred_{m}") & ~pl.col("odai_R1b")).sum().alias(f"fp_{m}") for m in methods],
        *[(pl.col(f"pred_{m}") & pl.col("odai_R1b")).sum().alias(f"tp_{m}") for m in methods]).sort("stratum2")
    pos = flags.filter(pl.col("odai_R1b") | pl.col("pred_v5.3.0") | pl.col("pred_pkg+emb") | pl.col("pred_pkg+ocr32b"))
    pos.select("source", "stratum2", "code", "memo", "odai_R1b", "p_has_odai_text", *[f"pred_{m}" for m in methods]).write_csv(
        os.path.join(OUT, "df_positives.csv"))
    comb = pl.concat([
        combined_eval(pool, n_pop, ["ex028"], methods).with_columns(pl.lit("ex028").alias("set")),
        combined_eval(pool, n_pop, ["ex028", "df"], methods).with_columns(pl.lit("combined").alias("set")),
        combined_eval(pool, n_pop, ["ex025", "ex028", "df"], {"v5.3.0": methods["v5.3.0"]}).with_columns(pl.lit("combined_v53").alias("set"))])
    comb.write_csv(os.path.join(OUT, "odai_combined.csv"))
    plot_df(r, comb, os.path.join(OUT, "fig_df.png"))
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_str_lengths(40)
    print(r)
    print(fp_post)
    print(pos.select("source", "stratum2", "code", "memo", "odai_R1b", "p_has_odai_text", *[f"pred_{m}" for m in methods]))
    print(comb)


if __name__ == "__main__":
    main()
