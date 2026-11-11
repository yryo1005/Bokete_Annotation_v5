"""形式を通常と予測した画像でのお題文ありの見逃し（report_042 の層 E）への対応を検討する（order_043）．

サブコマンド：
- sample：通常と予測し p（お題文ありの生徒の確率）が0.3以上の画像から，p の区間（BINS）ごとに N_PER_BIN 枚を追加で抽出し，
  番号だけを付けた画像の一覧を作る（targets_normal.csv，sheets_normal/）．目視のラベルは normal_labels.txt に付ける．
- qwen：新しい評価用データ（420枚）と追加の画像に，Qwen3-VL-32B（ex024 の指示 P3，最大 802,816 画素）でお題文ありかを問う．
- evaluate：候補を比べる．
  - 候補1：v5.3.0 に加えて，通常と予測し p が閾値以上の画像をお題文ありとする
  - 候補2：候補1の画像のうち，Qwen が真と答えた画像だけをお題文ありとする

通常と予測した画像の値は，p の区間で事後層別する（元の層 E の画像と追加の画像を区間ごとにまとめ，重みは区間の枚数 / 区間の目視の枚数）．
他の層の重みは report_042 と同じである．

出力：outputs/ex025_odaibun_eval_v5/normal_miss/（qwen.csv，bins.csv，metrics.csv（normal_* は通常と予測した画像に限った値），fig_normal.png）

実行例:
    .venv_annotate/bin/python programs/ex025_odaibun_eval_v5/normal_miss.py sample
    .venv_annotate/bin/python programs/ex025_odaibun_eval_v5/normal_miss.py qwen
    .venv_annotate/bin/python programs/ex025_odaibun_eval_v5/normal_miss.py evaluate
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (DATASET_DIR, EX024_DIR, OUTPUT_DIR, SEED, TARGETS, THRESHOLD, _load, excluded_reps, load_eval,  # noqa: E402
                  load_population)
from evaluate import FONT, N_BOOT, prf  # noqa: E402
from make_sheets import draw  # noqa: E402

OUT = os.path.join(OUTPUT_DIR, "normal_miss")
TARGETS_NORMAL = os.path.join(DATASET_DIR, "targets_normal.csv")
LABELS_NORMAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normal_labels.txt")
# p の区間（下限，上限）．0.1〜0.3 は元の層 E の画像のみ
BINS = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
N_PER_BIN = 60
GATE = 0.05
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QWEN_MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
QWEN_PIXELS = 802816


def bin_of(p: np.ndarray) -> np.ndarray:
    """p の区間の番号を求める．

    Args:
        p (np.ndarray): 形状 (N,) の確率．

    Returns:
        np.ndarray: 形状 (N,) の区間の番号（0.1未満は −1）．
    """
    out = np.full(len(p), -1)
    for k, (lo, hi) in enumerate(BINS):
        out[(p >= lo) & (p < hi)] = k
    return out


def cmd_sample() -> None:
    """追加の画像を抽出し，一覧を作る（targets_normal.csv があれば抽出し直さない）．

    Returns:
        None
    """
    os.makedirs(os.path.join(DATASET_DIR, "sheets_normal"), exist_ok=True)
    if not os.path.exists(TARGETS_NORMAL):
        used = excluded_reps() | set(pl.read_csv(TARGETS)["rep_odaiId"].to_list())
        pop = load_population().filter((pl.col("format_argmax") == "normal") & ~pl.col("rep_odaiId").is_in(list(used)))
        pop = pop.with_columns(pl.Series("bin", bin_of(pop["p"].to_numpy())))
        parts = [pop.filter(pl.col("bin") == k).sample(N_PER_BIN, seed=SEED + 10 + k) for k in range(1, len(BINS))]
        t = pl.concat(parts).select("rep_odaiId", "bin")
        t = t[np.random.default_rng(SEED).permutation(t.height)].with_row_index("no")
        t.write_csv(TARGETS_NORMAL)
    rows = pl.read_csv(TARGETS_NORMAL).to_dicts()
    for k in range(0, len(rows), 9):
        draw(rows[k:k + 9], os.path.join(DATASET_DIR, "sheets_normal", f"sheet_{k // 9:03d}.png"))
    print(pl.read_csv(TARGETS_NORMAL).group_by("bin").len().sort("bin"))


def cmd_qwen() -> None:
    """新しい評価用データと追加の画像に Qwen3-VL-32B で問い，qwen.csv に保存する．

    Returns:
        None
    """
    path = os.path.join(OUT, "qwen.csv")
    if os.path.exists(path):
        print("skip: qwen.csv がある")
        return
    os.makedirs(OUT, exist_ok=True)
    # qwen_odaibun.py は同じディレクトリの data（ex024）を読み込むため，読み込む間だけ ex024 の data に差し替える
    own = sys.modules["data"]
    sys.modules["data"] = sys.modules["ex024_data_for_ex025"]
    try:
        q = _load("ex024_qwen_for_ex025", os.path.join(EX024_DIR, "qwen_odaibun.py"))
    finally:
        sys.modules["data"] = own
    ids = sorted(set(pl.read_csv(TARGETS).filter(pl.col("kind") == "new")["rep_odaiId"].to_list())
                 | set(pl.read_csv(TARGETS_NORMAL)["rep_odaiId"].to_list()))
    llm = q.verify.build_llm(QWEN_MODEL, QWEN_PIXELS)
    df, sec, _ = q.run(llm, "P3", ids, QWEN_PIXELS, 1024)
    df.rename({"fileId": "rep_odaiId"}).select("rep_odaiId", "answer", "raw").write_csv(path)
    print(f"{len(ids)}枚，{sec:.1f}秒（{len(ids) / sec:.2f}枚/秒），解釈できない出力 {df['answer'].null_count()}")


def load_labels_normal() -> pl.DataFrame:
    """追加の画像の目視のラベルを読み込む（書式は「番号 0/1 [メモ]」）．

    Returns:
        pl.DataFrame: 列 rep_odaiId, label, memo を持つ表．
    """
    rows = []
    for line in open(LABELS_NORMAL, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            parts = line.rstrip("\n").split(maxsplit=2)
            rows.append({"no": int(parts[0]), "label": parts[1] == "1", "memo": parts[2] if len(parts) > 2 else ""})
    t = pl.read_csv(TARGETS_NORMAL).join(pl.DataFrame(rows), on="no", how="left")
    assert t["label"].null_count() == 0, "目視のラベルが欠けている"
    return t.select("rep_odaiId", "label", "memo")


def build_set() -> pl.DataFrame:
    """評価に用いる表（新しい評価用データと追加の画像．事後層別の重み付き）を作る．

    Returns:
        pl.DataFrame: 列 rep_odaiId, label, memo, stratum2（評価の層）, w, format_argmax, p, qwen を持つ表．
    """
    pop = load_population()
    ev = load_eval().select("rep_odaiId", "label", "memo", "stratum", "N_h")
    add = load_labels_normal().with_columns(pl.lit("added").alias("stratum"), pl.lit(None, pl.Int64).alias("N_h"))
    d = pl.concat([ev, add]).join(pop.drop("stratum"), on="rep_odaiId", how="left")
    b = bin_of(d["p"].to_numpy())
    normal = d["stratum"].is_in(["E_normal_p_mid", "added"]).to_numpy()
    st2 = np.where(normal, np.char.add("N", b.astype(str)), d["stratum"].to_numpy())
    d = d.with_columns(pl.Series("stratum2", st2))
    # 通常の区間の母集団の枚数（評価用データから除いた画像を含む全ての代表の画像で数える．除いた割合は1%未満）
    nb = bin_of(pop.filter(pl.col("format_argmax") == "normal")["p"].to_numpy())
    n_bin = {f"N{k}": int((nb == k).sum()) for k in range(len(BINS))}
    cnt = d.group_by("stratum2").len()
    d = d.join(cnt, on="stratum2")
    w = pl.when(pl.col("stratum2").str.starts_with("N")).then(pl.col("stratum2").replace_strict(n_bin, default=None) / pl.col("len")) \
        .otherwise(pl.col("N_h") / pl.col("len"))
    q = pl.read_csv(os.path.join(OUT, "qwen.csv")).select("rep_odaiId", pl.col("answer").fill_null(False).alias("qwen"))
    return d.with_columns(w.cast(pl.Float64).alias("w")).join(q, on="rep_odaiId", how="left").drop("len")


def v53(d: pl.DataFrame) -> np.ndarray:
    """パッケージ v5.3.0 のお題文ありを求める．

    Args:
        d (pl.DataFrame): build_set の戻り値．

    Returns:
        np.ndarray: 形状 (N,) の bool の配列．
    """
    am, p = d["format_argmax"].to_numpy(), d["p"].to_numpy()
    return ((am == "odaibun") & (p >= GATE)) | (np.isin(am, ["anaume", "karuta"]) & (p >= THRESHOLD))


def boot(d: pl.DataFrame, preds: dict, base: np.ndarray) -> dict:
    """層ごとの復元抽出で，F1 と基準との F1 の差の95% 信頼区間を求める．

    Args:
        d (pl.DataFrame): build_set の戻り値．
        preds (dict[str, np.ndarray]): 構成ごとの予測．
        base (np.ndarray): 基準（v5.3.0）の予測．

    Returns:
        dict[str, tuple]: 構成ごとの (F1 の下限, 上限, 差の下限, 差の上限)．
    """
    rng = np.random.default_rng(0)
    s, t, w = d["stratum2"].to_numpy(), d["label"].to_numpy(), d["w"].to_numpy()
    idx = [np.flatnonzero(s == k) for k in np.unique(s)]
    vals = {k: ([], []) for k in preds}
    for _ in range(N_BOOT):
        b = np.concatenate([rng.choice(i, len(i)) for i in idx])
        f0 = prf(t[b], base[b], w[b])["f1"]
        for k, p in preds.items():
            f = prf(t[b], p[b], w[b])["f1"]
            vals[k][0].append(f)
            vals[k][1].append(f - f0)
    q = lambda v, a: float(np.nanquantile(v, a))  # noqa: E731
    return {k: (q(f, 0.025), q(f, 0.975), q(df, 0.025), q(df, 0.975)) for k, (f, df) in vals.items()}


def plot(m: pl.DataFrame, out: str) -> None:
    """閾値ごとの全体の適合率・再現率・F1（候補1と候補2）の図を作る．

    Args:
        m (pl.DataFrame): metrics.csv の表．
        out (str): 保存先．

    Returns:
        None
    """
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    base = m.filter(pl.col("candidate") == "v5.3.0").row(0, named=True)
    colors = {"precision": "#2a78d6", "recall": "#eb6834", "f1": "#1baf7a"}
    for ax, (cand, title) in zip(axes, [("候補1", "候補1：通常で p が閾値以上"), ("候補2", "候補2：候補1かつ Qwen が真")]):
        d = m.filter(pl.col("candidate") == cand).sort("threshold")
        for k, mk, lab in [("precision", "o", "適合率"), ("recall", "s", "再現率"), ("f1", "D", "F1")]:
            ax.plot(d["threshold"], d[k], marker=mk, color=colors[k], lw=2, ms=6, label=lab)
            ax.axhline(base[k], color=colors[k], lw=1, ls="--", alpha=0.6)
        lo, hi = d["f1_lo"].to_numpy(), d["f1_hi"].to_numpy()
        ax.fill_between(d["threshold"], lo, hi, color=colors["f1"], alpha=0.12, lw=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("通常の画像でお題文ありとする p の閾値")
        ax.set_ylim(0.6, 1.0)
        ax.grid(color="#dddddd", lw=0.6)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, frameon=False, loc="lower center")
    fig.text(0.99, 0.02, "破線は v5.3.0，帯は F1 の95% 信頼区間", ha="right", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def cmd_evaluate() -> None:
    """候補を比べて保存する．

    Returns:
        None
    """
    d = build_set()
    t, w = d["label"].to_numpy(), d["w"].to_numpy()
    am, p, q = d["format_argmax"].to_numpy(), d["p"].to_numpy(), d["qwen"].to_numpy()
    base = v53(d)
    normal = am == "normal"
    preds = {("v5.3.0", None): base, ("Qwen のみ（全画像）", None): q}
    for th in THRESHOLDS:
        preds[("候補1", th)] = base | (normal & (p >= th))
        preds[("候補2", th)] = base | (normal & (p >= th) & q)
    ci = boot(d, {k: v for k, v in preds.items()}, base)
    pop = load_population()
    pn = pop.filter(pl.col("format_argmax") == "normal")["p"].to_numpy()
    rows = []
    for (cand, th), pr in preds.items():
        lo, hi, dlo, dhi = ci[(cand, th)]
        r = {"candidate": cand, "threshold": th, **prf(t, pr, w), "f1_lo": lo, "f1_hi": hi,
             "f1_diff": prf(t, pr, w)["f1"] - prf(t, base, w)["f1"], "f1_diff_lo": dlo, "f1_diff_hi": dhi,
             "n_normal_target": int((pn >= th).sum()) if th is not None else 0}
        sub = prf(t[normal], pr[normal], w[normal])
        r["normal_precision"], r["normal_recall"], r["normal_f1"] = sub["precision"], sub["recall"], sub["f1"]
        rows.append(r)
    m = pl.DataFrame(rows)
    m.write_csv(os.path.join(OUT, "metrics.csv"))
    # 通常の区間ごとのお題文ありの割合と Qwen の一致
    nd = d.filter(pl.col("stratum2").str.starts_with("N"))
    bins = nd.group_by("stratum2").agg(pl.len().alias("n"), pl.col("label").sum().alias("n_pos"), pl.col("w").first().alias("w"),
                                        (pl.col("qwen") & pl.col("label")).sum().alias("qwen_tp"), pl.col("qwen").sum().alias("qwen_pos"))
    bins = bins.with_columns((pl.col("n") * pl.col("w")).round().alias("N"), (pl.col("n_pos") / pl.col("n")).alias("pos_rate"),
                             (pl.col("qwen_tp") / pl.col("qwen_pos")).alias("qwen_precision"),
                             (pl.col("qwen_tp") / pl.col("n_pos")).alias("qwen_recall")).sort("stratum2")
    bins = bins.with_columns(pl.col("stratum2").replace_strict({f"N{k}": f"{lo}〜{min(hi, 1.0)}" for k, (lo, hi) in enumerate(BINS)}).alias("p_bin"))
    bins.write_csv(os.path.join(OUT, "bins.csv"))
    plot(m, os.path.join(OUT, "fig_normal.png"))
    with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=220):
        print(bins)
        print(m.select("candidate", "threshold", "precision", "recall", "f1", "f1_lo", "f1_hi", "f1_diff", "f1_diff_lo", "f1_diff_hi",
                       "n_normal_target", "normal_precision", "normal_recall"))


def main() -> None:
    """サブコマンドを実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["sample", "qwen", "evaluate"])
    args = parser.parse_args()
    {"sample": cmd_sample, "qwen": cmd_qwen, "evaluate": cmd_evaluate}[args.cmd]()


if __name__ == "__main__":
    main()
