"""新しい評価用データ（層別抽出の420枚）で，お題文ありの判定の構成を評価し，目視のラベルの再現性を求める．

構成（p はお題文ありの生徒（ex024 の emb+probs，lr 1e-3，3 seed の平均）の確率，閾値 THRESHOLD）：
- 現在のパッケージ：v5.2.0 の形式（確認の質問を反映）がお題文
- 現在の生徒（確認の質問なし）：形式の確率が最大のものがお題文
- 生徒（学習し直し）：p が閾値以上
- 現在＋生徒（確認の質問あり）：形式がお題文，または形式が穴埋め・カルタで p が閾値以上（ex024 の推奨の構成）
- 現在＋生徒（確認の質問なし）：上の形式を，確率が最大の形式に置き換えたもの
- パッケージ v5.3：パッケージの出力の has_odai_text（列があれば）

値は層の重み（層の枚数 / 抽出数）で母集団（代表の画像）の値を推定し，95% 信頼区間は層ごとの復元抽出（N_BOOT 回）で求める．

既存の目視のラベル（1,020枚．ex024 の正解）でも，パッケージ v5.2.0（ANNOTATION_V52）と v5.3 のお題文ありと形式を比べる．
形式の正解は，正解の属性から定義書 v5 の順で導いたもの（カルタ記号あり → カルタ，空欄あり → 穴埋め，お題文あり → お題文，それ以外 → 通常）である．

出力：outputs/ex025_odaibun_eval_v5/（metrics.csv，strata.csv，repeat.csv，repeat_pairs.csv，old_set.csv，threshold_blank.csv，gate_odaibun.csv，
      fig_metrics.png，fig_strata.png，fig_threshold_blank.png）

実行例:
    .venv_annotate/bin/python programs/ex025_odaibun_eval_v5/evaluate.py
"""
import itertools
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ANNOTATION, ANNOTATION_V52, OUTPUT_DIR, STRATA, THRESHOLD, load_eval, load_population, load_repeat, load_truth_ex024  # noqa: E402

FONT = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
N_BOOT = 2000
COLORS = {"precision": "#2a78d6", "recall": "#eb6834", "f1": "#1baf7a"}


def prf(t: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    """重み付きの適合率・再現率・F1 を求める．

    Args:
        t (np.ndarray): 形状 (N,) の正解（bool）．
        p (np.ndarray): 形状 (N,) の予測（bool）．
        w (np.ndarray): 形状 (N,) の重み．

    Returns:
        dict[str, float]: precision, recall, f1．
    """
    tp, npred, ntrue = (w * (t & p)).sum(), (w * p).sum(), (w * t).sum()
    return {"precision": tp / npred if npred else float("nan"), "recall": tp / ntrue if ntrue else float("nan"),
            "f1": 2 * tp / (npred + ntrue) if npred + ntrue else float("nan")}


def boot_ci(stratum: np.ndarray, t: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    """層ごとの復元抽出で，適合率・再現率・F1 の95% 信頼区間を求める．

    Args:
        stratum (np.ndarray): 形状 (N,) の層の名前．
        t (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の予測．
        w (np.ndarray): 形状 (N,) の重み．

    Returns:
        dict[str, tuple[float, float]]: 指標ごとの (下限, 上限)．
    """
    rng = np.random.default_rng(0)
    idx = [np.flatnonzero(stratum == s) for s in np.unique(stratum)]
    vals = {k: [] for k in COLORS}
    for _ in range(N_BOOT):
        b = np.concatenate([rng.choice(i, len(i)) for i in idx])
        for k, v in prf(t[b], p[b], w[b]).items():
            vals[k].append(v)
    return {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))) for k, v in vals.items()}


def boot_diff(stratum: np.ndarray, t: np.ndarray, p: np.ndarray, base: np.ndarray, w: np.ndarray) -> tuple:
    """層ごとの復元抽出（同じ標本の対）で，F1 の差（p − base）の95% 信頼区間を求める．

    Args:
        stratum (np.ndarray): 形状 (N,) の層の名前．
        t (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の比べる構成の予測．
        base (np.ndarray): 形状 (N,) の基準の構成の予測．
        w (np.ndarray): 形状 (N,) の重み．

    Returns:
        tuple[float, float]: (下限, 上限)．
    """
    rng = np.random.default_rng(0)
    idx = [np.flatnonzero(stratum == s) for s in np.unique(stratum)]
    d = []
    for _ in range(N_BOOT):
        b = np.concatenate([rng.choice(i, len(i)) for i in idx])
        d.append(prf(t[b], p[b], w[b])["f1"] - prf(t[b], base[b], w[b])["f1"])
    return float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))


def configs(ev: pl.DataFrame) -> dict:
    """構成ごとのお題文ありの予測を求める．

    Args:
        ev (pl.DataFrame): 評価用データに load_population の列を加えた表．

    Returns:
        dict[str, np.ndarray]: 構成の名前と，形状 (N,) の予測（bool）．
    """
    p = ev["p"].to_numpy() >= THRESHOLD
    fmt, am = ev["format"].to_numpy(), ev["format_argmax"].to_numpy()
    bk, bk_am = np.isin(fmt, ["anaume", "karuta"]), np.isin(am, ["anaume", "karuta"])
    out = {"現在のパッケージ": fmt == "odaibun", "現在の生徒（確認の質問なし）": am == "odaibun", "生徒（学習し直し）": p,
           "現在＋生徒（確認の質問あり）": (fmt == "odaibun") | (bk & p), "現在＋生徒（確認の質問なし）": (am == "odaibun") | (bk_am & p)}
    if "has_odai_text" in ev.columns:
        out["パッケージ v5.3"] = ev["has_odai_text"].to_numpy()
    return out


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """2値のラベルの Cohen の κ を求める．

    Args:
        a (np.ndarray): 形状 (N,) のラベル（bool）．
        b (np.ndarray): 形状 (N,) のラベル（bool）．

    Returns:
        float: κ．
    """
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def evaluate_repeat(out_dir: str) -> pl.DataFrame:
    """再現性の確認の画像で，付け直したラベルと過去のラベルの一致を求める．

    Args:
        out_dir (str): 保存先．

    Returns:
        pl.DataFrame: 群ごと（と全体）の n, agreement, kappa, n_old_pos, n_new_pos を持つ表．
    """
    r = load_repeat()
    r.filter(pl.col("label") != pl.col("label_old")).write_csv(os.path.join(out_dir, "repeat_pairs.csv"))
    rows = []
    for g in ["odaibun", "blank_karuta", "normal", "全体"]:
        s = r if g == "全体" else r.filter(pl.col("group") == g)
        a, b = s["label_old"].to_numpy(), s["label"].to_numpy()
        rows.append({"group": g, "n": s.height, "agreement": float((a == b).mean()), "kappa": kappa(a, b),
                     "n_old_pos": int(a.sum()), "n_new_pos": int(b.sum())})
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(out_dir, "repeat.csv"))
    return df


def evaluate_old(out_dir: str) -> pl.DataFrame:
    """既存の目視のラベル（1,020枚）で，パッケージ v5.2.0 と v5.3 のお題文ありと形式を比べる．

    Args:
        out_dir (str): 保存先．

    Returns:
        pl.DataFrame: 列 version, split, odai_precision, odai_recall, odai_f1, format_accuracy を持つ表．
    """
    t = load_truth_ex024()
    fmt_true = np.where(t["has_karuta_mark"], "karuta", np.where(t["has_blank"], "anaume",
                                                                  np.where(t["odaibun_attr"], "odaibun", "normal")))
    t = t.with_columns(pl.Series("format_v5", fmt_true))
    cols = ["odaiId", "format"]
    rows = []
    for version, path, odai in [("v5.2.0", ANNOTATION_V52, pl.col("format") == "odaibun"), ("v5.3.0", ANNOTATION, pl.col("has_odai_text"))]:
        a = pl.scan_parquet(path).select(*cols, odai.alias("odai")).collect().rename({"odaiId": "fileId", "format": "format_pred"})
        d = t.join(a, on="fileId", how="left")
        for s in ["dev", "test", "real"]:
            m = d.filter(pl.col("split") == s)
            w = m["w"].to_numpy()
            r = prf(m["odaibun_attr"].to_numpy(), m["odai"].to_numpy(), w)
            rows.append({"version": version, "split": s, "odai_precision": r["precision"], "odai_recall": r["recall"], "odai_f1": r["f1"],
                         "format_accuracy": float(np.average(m["format_pred"].to_numpy() == m["format_v5"].to_numpy(), weights=w))})
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(out_dir, "old_set.csv"))
    return df


def old_with_population() -> pl.DataFrame:
    """既存の目視のラベル（1,020枚）に，load_population の列（v5.2.0 の予測とお題文ありの生徒の確率）を加える．

    Returns:
        pl.DataFrame: load_truth_ex024 の列と format_argmax, p などを持つ表．
    """
    rep = pl.scan_parquet(ANNOTATION).select(pl.col("odaiId").alias("fileId"), "rep_odaiId").collect()
    return load_truth_ex024().join(rep, on="fileId", how="left").join(load_population().drop("stratum"), on="rep_odaiId", how="left")


def sweep_tables(ev: pl.DataFrame, out_dir: str) -> tuple:
    """穴埋め・カルタの画像の閾値と，お題文の画像の下限（gate）を変えたときの値を，新しい評価用データと既存の1,020枚で求める．

    Args:
        ev (pl.DataFrame): 評価用データに load_population の列を加えた表．
        out_dir (str): 保存先．

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: threshold_blank.csv と gate_odaibun.csv の表．
    """
    old = old_with_population()
    sets = {"new": (ev, "label", ev["w"].to_numpy().astype(float)),
            "old_eval": (old.filter(pl.col("split") != "real"), "odaibun_attr", None),
            "old_real": (old.filter(pl.col("split") == "real"), "odaibun_attr", "w")}
    rows = []
    for th, (name, (d, lab, w)) in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], sets.items()):
        d = d.filter(pl.col("format_argmax").is_in(["anaume", "karuta"]))
        ww = np.ones(d.height) if w is None else (d[w].to_numpy() if isinstance(w, str) else d["w"].to_numpy().astype(float))
        rows.append({"threshold": th, "set": name, "n": d.height, "n_pos": int(d[lab].sum()),
                     **prf(d[lab].to_numpy(), d["p"].to_numpy() >= th, ww)})
    blank = pl.DataFrame(rows)
    blank.write_csv(os.path.join(out_dir, "threshold_blank.csv"))
    rows = []
    for g, (name, (d, lab, _)) in itertools.product([0.02, 0.05, 0.1, 0.2, 0.3], sets.items()):
        d = d.filter(pl.col("format_argmax") == "odaibun")
        cut = d["p"].to_numpy() < g
        t = d[lab].to_numpy()
        rows.append({"gate": g, "set": name, "n": d.height, "n_removed": int(cut.sum()), "n_removed_pos": int((t & cut).sum())})
    gate = pl.DataFrame(rows)
    gate.write_csv(os.path.join(out_dir, "gate_odaibun.csv"))
    return blank, gate


def plot_threshold(blank: pl.DataFrame, out_dir: str) -> None:
    """穴埋め・カルタの画像の閾値と，お題文ありの適合率・再現率・F1 の関係の図を作る．

    Args:
        blank (pl.DataFrame): threshold_blank.csv の表．
        out_dir (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    titles = {"new": "新しい評価用データ（層の重み）", "old_eval": "既存の評価用データ（640枚）", "old_real": "既存の実データ（重み付き）"}
    for ax, (name, title) in zip(axes, titles.items()):
        d = blank.filter(pl.col("set") == name)
        for m, mk, lab in [("precision", "o", "適合率"), ("recall", "s", "再現率"), ("f1", "D", "F1")]:
            ax.plot(d["threshold"], d[m], marker=mk, color=COLORS[m], lw=2, ms=6, label=lab)
        ax.axvline(THRESHOLD, color="#999999", lw=1, ls="--")
        n, n_pos = d["n"][0], d["n_pos"][0]
        ax.set_title(f"{title}\n穴埋め・カルタ {n}枚（お題文あり {n_pos}枚）", fontsize=10)
        ax.set_xlabel("お題文ありの生徒の確率の閾値")
        ax.set_ylim(0, 1.05)
        ax.grid(color="#dddddd", lw=0.6)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out_dir, "fig_threshold_blank.png"), dpi=150)
    plt.close(fig)


def plot_metrics(rows: pl.DataFrame, out_dir: str) -> None:
    """構成ごとの適合率・再現率・F1（95% 信頼区間付き）の図を作る．

    Args:
        rows (pl.DataFrame): metrics.csv の表．
        out_dir (str): 保存先．

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    names = rows["config"].to_list()
    y = np.arange(len(names))[::-1]
    for k, (m, mk, lab) in enumerate([("precision", "o", "適合率"), ("recall", "s", "再現率"), ("f1", "D", "F1")]):
        v = rows[m].to_numpy()
        err = np.abs(np.stack([v - rows[f"{m}_lo"].to_numpy(), rows[f"{m}_hi"].to_numpy() - v]))
        yy = y + 0.22 - 0.22 * k
        ax.errorbar(v, yy, xerr=err, fmt="none", ecolor=COLORS[m], elinewidth=1.2, alpha=0.7)
        ax.scatter(v, yy, marker=mk, color=COLORS[m], s=44, label=lab, edgecolor="white", lw=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.0, 1.02)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_xlabel("値（層の重みで推定した母集団の値．誤差棒は95% 信頼区間）")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out_dir, "fig_metrics.png"), dpi=150)
    plt.close(fig)


def plot_strata(st: pl.DataFrame, out_dir: str) -> None:
    """層ごとのお題文ありの割合（目視）の図を作る．

    Args:
        st (pl.DataFrame): strata.csv の表．
        out_dir (str): 保存先．

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    y = np.arange(st.height)[::-1]
    ax.barh(y, st["pos_rate"].to_numpy(), color="#2a78d6", height=0.6)
    for yy, r in zip(y, st.iter_rows(named=True)):
        ax.text(r["pos_rate"] + 0.01, yy, f"{r['pos_rate']:.2f}（{r['n_pos']}/{r['n']}）", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']}\n層の枚数 {r['N']:,}" for r in st.iter_rows(named=True)], fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("お題文ありの割合（目視）")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_strata.png"), dpi=150)
    plt.close(fig)


STRATUM_LABEL = {"A_odaibun_yes": "A：お題文（確認は真）", "B_odaibun_rejected": "B：お題文（確認は偽）",
                 "C_blank_p_hi": f"C：穴埋め・カルタ，p が{THRESHOLD}以上", "D_blank_p_lo": f"D：穴埋め・カルタ，p が{THRESHOLD}未満",
                 "E_normal_p_mid": "E：通常，p が0.1以上", "F_normal_p_lo": "F：通常，p が0.1未満"}


def main() -> None:
    """評価して保存する．

    Returns:
        None
    """
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    ev = load_eval()
    pop = load_population().filter(pl.col("rep_odaiId").is_in(ev["rep_odaiId"].to_list())).drop("stratum")
    ev = ev.join(pop, on="rep_odaiId", how="left")
    ann_cols = pl.scan_parquet(ANNOTATION).collect_schema().names()
    if "has_odai_text" in ann_cols:
        ev = ev.join(pl.scan_parquet(ANNOTATION).select(pl.col("odaiId").alias("rep_odaiId"), "has_odai_text").collect(),
                     on="rep_odaiId", how="left")
    t, w, s = ev["label"].to_numpy(), ev["w"].to_numpy().astype(float), ev["stratum"].to_numpy()
    rows = []
    cf = configs(ev)
    base = cf["現在のパッケージ"]
    for name, pred in cf.items():
        r = {"config": name, **prf(t, pred, w), "n_pred": int(pred.sum()), "pred_rate": float(np.average(pred, weights=w))}
        for k, (lo, hi) in boot_ci(s, t, pred, w).items():
            r[f"{k}_lo"], r[f"{k}_hi"] = lo, hi
        r["f1_diff"] = r["f1"] - prf(t, base, w)["f1"]
        r["f1_diff_lo"], r["f1_diff_hi"] = boot_diff(s, t, pred, base, w)
        rows.append(r)
    rows = pl.DataFrame(rows)
    rows.write_csv(os.path.join(out_dir, "metrics.csv"))
    st = ev.group_by("stratum").agg(pl.len().alias("n"), pl.col("label").sum().alias("n_pos"), pl.col("N_h").first().alias("N"),
                                     pl.col("p").mean().alias("mean_p")).sort("stratum")
    st = st.with_columns((pl.col("n_pos") / pl.col("n")).alias("pos_rate"),
                         pl.col("stratum").replace_strict(STRATUM_LABEL).alias("label"))
    st.write_csv(os.path.join(out_dir, "strata.csv"))
    rep = evaluate_repeat(out_dir)
    old = evaluate_old(out_dir) if os.path.exists(ANNOTATION_V52) and "has_odai_text" in ann_cols else None
    blank, gate = sweep_tables(ev, out_dir)
    plot_threshold(blank, out_dir)
    plot_metrics(rows, out_dir)
    plot_strata(st, out_dir)
    pos_rate = float(np.average(t, weights=w))
    print(f"正例 {int(t.sum())} / {len(t)} 枚，母集団のお題文ありの割合の推定 {pos_rate:.4f}")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200):
        print(rows.select("config", "precision", "recall", "f1", "f1_lo", "f1_hi", "f1_diff", "f1_diff_lo", "f1_diff_hi", "pred_rate"))
        print(st)
        print(rep)
        print(old)
        print(blank.filter(pl.col("set") == "new"))
        print(gate)


if __name__ == "__main__":
    main()
