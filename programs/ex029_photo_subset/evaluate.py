"""写真・文字なし（条件A）と写真・通常（条件B）の件数と判定精度を求める．

評価の内容：
1. 件数：全てのお題（重複を含む）と代表の画像の数，ボケの件数と画像外の知識の割合（v5.3.0 の予測による）．
2. 判定精度：1,020枚の目視の正解で，条件ごとの適合率・再現率・F1．実データ（real．重み付き，群ごとの2000回の復元抽出の95% 区間）と
   評価用データ（eval．重みなし）．条件A は v4 の文字ありと ex026 の v6 の文字の層（付け直し）の2つの正解，
   条件B は v4 の通常と，v6 の規則 R1b で導いた通常の2つの正解で求める．
3. 実データの推定の割合：目視の正解による条件の割合（v5.3.0 の予測の割合と比べる）．

出力：outputs/ex029_photo_subset/（counts.csv，metrics.csv，confusion.csv，fig_metrics.png，evaluate.log）

実行例:
    .venv_annotate/bin/python programs/ex029_photo_subset/evaluate.py
"""
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CONDITIONS, OUTPUT_DIR, boot_groups, count_boke, load_pred, load_truth  # noqa: E402

plt.rcParams["font.family"] = ["IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "DejaVu Sans"]
N_BOOT = 2000
# 評価する（条件, 正解の列）
TARGETS = [("photo_only", "photo_only"), ("A_photo_notext", "A_photo_notext"), ("A_photo_notext", "A_photo_notext_v6"),
           ("B_photo_normal", "B_photo_normal"), ("B_photo_normal", "B_photo_normal_v6")]
TRUTH_JA = {"photo_only": "v4", "A_photo_notext": "v4", "A_photo_notext_v6": "v6", "B_photo_normal": "v4", "B_photo_normal_v6": "v6（R1b）"}


def prf(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    """重み付きの適合率・再現率・F1 と割合を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解（bool）．
        p (np.ndarray): 形状 (N,) の予測（bool）．
        w (np.ndarray): 形状 (N,) の重み．

    Returns:
        dict[str, float]: precision, recall, f1, rate_true, rate_pred．
    """
    tp, npred, ntrue = (w * (y & p)).sum(), (w * p).sum(), (w * y).sum()
    return {"precision": tp / npred if npred else np.nan, "recall": tp / ntrue if ntrue else np.nan,
            "f1": 2 * tp / (npred + ntrue) if npred + ntrue else np.nan, "rate_true": ntrue / w.sum(), "rate_pred": npred / w.sum()}


def boot(y: np.ndarray, p: np.ndarray, w: np.ndarray, groups: list) -> dict:
    """群ごとの復元抽出で95% 区間を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の予測．
        w (np.ndarray): 形状 (N,) の重み．
        groups (list[np.ndarray]): 群ごとの行番号．

    Returns:
        dict[str, float]: 指標ごとの {名前}_lo と {名前}_hi．
    """
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(N_BOOT):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(prf(y[i], p[i], w[i]))
    out = {}
    for k in ["precision", "recall", "f1", "rate_true"]:
        v = np.array([r[k] for r in vals], dtype=float)
        out[f"{k}_lo"], out[f"{k}_hi"] = float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
    return out


def counts() -> pl.DataFrame:
    """条件ごとの件数を求める．

    Returns:
        pl.DataFrame: 列 condition, n_odai, n_rep, n_boke, outside_rate, share_odai, share_boke を持つ表．
    """
    a = load_pred().join(count_boke(), on="odaiId", how="left").with_columns(pl.col("n_boke", "n_outside").fill_null(0))
    tot = {"n_odai": a.height, "n_rep": a["rep_odaiId"].n_unique(), "n_boke": int(a["n_boke"].sum())}
    rows = [{"condition": "all", **tot, "outside_rate": float(a["n_outside"].sum() / a["n_boke"].sum())}]
    for c in CONDITIONS:
        s = a.filter(pl.col(c))
        rows.append({"condition": c, "n_odai": s.height, "n_rep": s["rep_odaiId"].n_unique(), "n_boke": int(s["n_boke"].sum()),
                     "outside_rate": float(s["n_outside"].sum() / max(s["n_boke"].sum(), 1))})
    s = a.filter(pl.col("A_photo_notext") & pl.col("B_photo_normal"))
    rows.append({"condition": "A_and_B", "n_odai": s.height, "n_rep": s["rep_odaiId"].n_unique(), "n_boke": int(s["n_boke"].sum()),
                 "outside_rate": float(s["n_outside"].sum() / max(s["n_boke"].sum(), 1))})
    return pl.DataFrame(rows).with_columns((pl.col("n_odai") / tot["n_odai"]).alias("share_odai"),
                                           (pl.col("n_boke") / tot["n_boke"]).alias("share_boke"))


def metrics(t: pl.DataFrame, pred: pl.DataFrame) -> tuple:
    """条件ごとの精度と混同行列を求める．

    Args:
        t (pl.DataFrame): load_truth の表．
        pred (pl.DataFrame): load_pred の表（同じ画像）．

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: 精度の表と，混同行列（重み付きの件数）の表．
    """
    d = t.join(pred.rename({"odaiId": "fileId"}).select("fileId", *[pl.col(c).alias(f"pred_{c}") for c in CONDITIONS]), on="fileId")
    rows, conf = [], []
    for s in ["real", "eval"]:
        x = d.filter(pl.col("set").is_in(["random", "strat"]) if s == "real" else pl.col("set") == "eval")
        w = x["w"].to_numpy() if s == "real" else np.ones(x.height)
        groups = boot_groups(x) if s == "real" else [np.arange(x.height)]
        for cond, truth in TARGETS:
            y, p = x[truth].to_numpy(), x[f"pred_{cond}"].to_numpy()
            rows.append({"set": s, "condition": cond, "truth": truth, "n": x.height, "n_true": int(y.sum()), "n_pred": int(p.sum()),
                         **prf(y, p, w), **boot(y, p, w, groups)})
            for a, b in [(True, True), (True, False), (False, True), (False, False)]:
                m = (y == a) & (p == b)
                conf.append({"set": s, "condition": cond, "truth": truth, "true": a, "pred": b, "n": int(m.sum()), "w": float(w[m].sum())})
    return pl.DataFrame(rows), pl.DataFrame(conf)


def plot(m: pl.DataFrame, out: str) -> None:
    """実データの適合率・再現率・F1（95% 区間）を条件ごとに描く．

    Args:
        m (pl.DataFrame): metrics の精度の表．
        out (str): 保存先．

    Returns:
        None
    """
    r = m.filter(pl.col("set") == "real")
    labels = [f"{CONDITIONS[c]}\n（正解 {TRUTH_JA[t]}）" for c, t in zip(r["condition"], r["truth"])]
    x = np.arange(r.height)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for k, (name, col) in enumerate([("precision", "#2a78d6"), ("recall", "#eb6834"), ("f1", "#1baf7a")]):
        v = r[name].to_numpy()
        ax.errorbar(x + (k - 1) * 0.22, v, yerr=[v - r[f"{name}_lo"].to_numpy(), r[f"{name}_hi"].to_numpy() - v], fmt="o", color=col,
                    capsize=3, label={"precision": "適合率", "recall": "再現率", "f1": "F1"}[name])
        for xi, vi in zip(x + (k - 1) * 0.22, v):
            ax.text(xi + 0.04, vi, f"{vi:.2f}", ha="left", va="center", fontsize=7, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.82, 1.02)
    ax.set_ylabel("実データの推定（95% 区間）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """件数と精度を求めて保存する．

    Returns:
        None
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c = counts()
    c.write_csv(os.path.join(OUTPUT_DIR, "counts.csv"))
    t = load_truth()
    m, conf = metrics(t, load_pred(t["fileId"].to_list()))
    m.write_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))
    conf.write_csv(os.path.join(OUTPUT_DIR, "confusion.csv"))
    plot(m, os.path.join(OUTPUT_DIR, "fig_metrics.png"))
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(40)
    print(c)
    print(m.select("set", "condition", "truth", "n", "n_true", "n_pred", "precision", "precision_lo", "precision_hi", "recall", "recall_lo",
                   "recall_hi", "f1", "rate_true", "rate_true_lo", "rate_true_hi", "rate_pred"))
    print(conf.filter(pl.col("set") == "real"))


if __name__ == "__main__":
    main()
