"""お題画像のラベルのカテゴリごとの適合率・再現率を，判断者（Qwen，生徒，全プロセス）ごとに求める．

判断者：
- qwen：Qwen3-VL-8B 単独のラベル（教師 T_v11）
- qwen_verify：Qwen3-VL-8B の確認の質問（お題文か）を全ての画像に行った回答（お題文のみ）
- student：確認の質問を反映する前の生徒
- full：全プロセス（生徒＋確認の質問．パッケージの出力）
- full_attr：全プロセスで，確認の質問が偽のときに生徒の属性から形式を導く場合の模擬

評価の集合：
- real：実データ（無作為抽出160枚と層別抽出220枚を重み付けした推定値．ブートストラップの95%区間を付ける）
- eval：評価用データ640枚（少ない区分を多く含む．重みなし）

出力：outputs/ex023_image_label_pr/（metrics.csv，verify_stage.csv，overall.csv，tables.md，fig_pr_*.png）

実行例:
    .venv_annotate/bin/python programs/ex023_image_label_pr/evaluate.py
"""
import os
import sys

import matplotlib
import numpy as np
import polars as pl
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import AXIS, CATEGORIES, JA, OUTPUT_DIR, load_pipeline, load_qwen, load_truth  # noqa: E402

JUDGES = ["qwen", "qwen_verify", "student", "full", "full_attr"]
JUDGE_JA = {"qwen": "Qwen 単独", "qwen_verify": "Qwen 確認の質問（全画像）", "student": "生徒", "full": "全プロセス",
            "full_attr": "全プロセス（属性から形式）"}
COLORS = {"qwen": "#2a78d6", "qwen_verify": "#7a7a7a", "student": "#eb6834", "full": "#1baf7a"}
MARKERS = {"qwen": "o", "qwen_verify": "D", "student": "s", "full": "^"}
FONT = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
N_BOOT = 2000


def build() -> tuple:
    """正解と判断者のラベルを，同じ行の順の行列にそろえる．

    Returns:
        tuple[pl.DataFrame, np.ndarray, dict, pl.DataFrame]:
            正解の表（列 fileId, set, w, stratum），形状 (N, C) の正解の bool 行列，
            判断者名 → 形状 (N, C) の予測の bool 行列（qwen_verify はお題文以外が nan の float），確認の質問の情報．
    """
    truth = load_truth()
    ids = truth["fileId"].unique().to_list()
    pipe = load_pipeline(ids)
    qv = pl.read_csv(os.path.join(OUTPUT_DIR, "qwen_verify_all.csv"))
    preds = {"qwen": load_qwen(), "student": pipe["student"], "full": pipe["full"], "full_attr": pipe["full_attr"]}
    key = truth.select("fileId")
    mats = {k: key.join(v, on="fileId", how="left").select(CATEGORIES).to_numpy().astype(bool) for k, v in preds.items()}
    v = key.join(qv, on="fileId", how="left")["qwen_verify"].to_numpy().astype(float)
    m = np.full((len(key), len(CATEGORIES)), np.nan)
    m[:, CATEGORIES.index("format_odaibun")] = v
    mats["qwen_verify"] = m
    info = key.join(pipe["_info"], on="fileId", how="left")
    return truth, truth.select(CATEGORIES).to_numpy().astype(bool), mats, info


def pr_counts(t: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """重み付きの真陽性，予測の正例，正解の正例の和を求める．

    Args:
        t (np.ndarray): 形状 (N, C) の正解（bool）．
        p (np.ndarray): 形状 (N, C) の予測（bool または nan を含む float）．
        w (np.ndarray): 形状 (N,) の重み．

    Returns:
        np.ndarray: 形状 (3, C)．行は真陽性，予測の正例，正解の正例の重み付きの和．
    """
    pb = np.nan_to_num(p.astype(float)) > 0.5
    return np.stack([(w[:, None] * (t & pb)).sum(0), (w[:, None] * pb).sum(0), (w[:, None] * t).sum(0)])


def to_metrics(c: np.ndarray) -> np.ndarray:
    """真陽性などの和から適合率，再現率，F1 を求める．

    Args:
        c (np.ndarray): 形状 (3, ...) の pr_counts の戻り値（またはその積み重ね）．

    Returns:
        np.ndarray: 形状 (3, ...)．行は適合率，再現率，F1（定義できない値は nan）．
    """
    tp, npred, ntrue = c
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([tp / npred, tp / ntrue, 2 * tp / (npred + ntrue)])


def boot_indices(truth: pl.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """実データの1回分のブートストラップの行番号を作る（無作為抽出全体と，層別抽出の層ごとに復元抽出する）．

    Args:
        truth (pl.DataFrame): build の正解の表．
        rng (np.random.Generator): 乱数の生成器．

    Returns:
        np.ndarray: 行番号の配列．
    """
    groups = truth.with_row_index("i").filter(pl.col("set") != "eval").group_by("set", "stratum").agg("i")["i"].to_list()
    return np.concatenate([rng.choice(np.asarray(g), len(g)) for g in groups])


def evaluate(truth: pl.DataFrame, t: np.ndarray, mats: dict) -> pl.DataFrame:
    """判断者×カテゴリ×評価の集合の適合率，再現率，F1 を求める．

    Args:
        truth (pl.DataFrame): build の正解の表．
        t (np.ndarray): 形状 (N, C) の正解．
        mats (dict[str, np.ndarray]): 判断者名 → 形状 (N, C) の予測．

    Returns:
        pl.DataFrame: 長い形式の表（列 set, judge, category, axis, precision, recall, f1, 区間, 件数）．
    """
    sets = {"real": (truth["set"] != "eval").to_numpy(), "eval": (truth["set"] == "eval").to_numpy()}
    w = truth["w"].to_numpy()
    rng = np.random.default_rng(0)
    boots = [boot_indices(truth, rng) for _ in range(N_BOOT)]
    rows = []
    for judge in tqdm(JUDGES, desc="judges", leave=False):
        p = mats[judge]
        valid = ~np.isnan(p.astype(float)).all(0)
        for s, mask in sets.items():
            m = to_metrics(pr_counts(t[mask], p[mask], w[mask]))
            n = pr_counts(t[mask], p[mask], np.ones(mask.sum()))
            ci = np.full((2, 3, len(CATEGORIES)), np.nan)
            if s == "real":
                bm = np.stack([to_metrics(pr_counts(t[b], p[b], w[b])) for b in boots])
                ci = np.nanquantile(bm, [0.025, 0.975], axis=0)
            for j, c in enumerate(CATEGORIES):
                if not valid[j]:
                    continue
                rows.append({"set": s, "judge": judge, "category": c, "axis": AXIS[c], "precision": m[0, j], "recall": m[1, j],
                             "f1": m[2, j], "precision_lo": ci[0, 0, j], "precision_hi": ci[1, 0, j], "recall_lo": ci[0, 1, j],
                             "recall_hi": ci[1, 1, j], "f1_lo": ci[0, 2, j], "f1_hi": ci[1, 2, j],
                             "n_true": int(n[2, j]), "n_pred": int(n[1, j]), "n_tp": int(n[0, j])})
    return pl.DataFrame(rows)


def overall(truth: pl.DataFrame, t: np.ndarray, mats: dict) -> pl.DataFrame:
    """判断者ごとに，軸ごとの全項目の一致率と全項目の一致率を求める．

    Args:
        truth (pl.DataFrame): build の正解の表．
        t (np.ndarray): 形状 (N, C) の正解．
        mats (dict[str, np.ndarray]): 判断者名 → 形状 (N, C) の予測．

    Returns:
        pl.DataFrame: 列 set, judge, 形式, 属性, 媒体, 全項目 を持つ表．
    """
    sets = {"real": (truth["set"] != "eval").to_numpy(), "eval": (truth["set"] == "eval").to_numpy()}
    w = truth["w"].to_numpy()
    axes = {a: [j for j, c in enumerate(CATEGORIES) if AXIS[c] == a] for a in ["形式", "属性", "媒体"]}
    rows = []
    for judge in ["qwen", "student", "full", "full_attr"]:
        ok = mats[judge] == t
        for s, mask in sets.items():
            r = {"set": s, "judge": judge}
            for a, js in axes.items():
                r[a] = float(np.average(ok[mask][:, js].all(1), weights=w[mask]))
            r["全項目"] = float(np.average(ok[mask].all(1), weights=w[mask]))
            rows.append(r)
    return pl.DataFrame(rows)


def verify_stage(truth: pl.DataFrame, t: np.ndarray, mats: dict, info: pl.DataFrame) -> pl.DataFrame:
    """確認の質問の段階の内訳（生徒がお題文と予測した画像での，正解と Qwen の回答の組の件数）を求める．

    Args:
        truth (pl.DataFrame): build の正解の表．
        t (np.ndarray): 形状 (N, C) の正解．
        mats (dict[str, np.ndarray]): 判断者名 → 予測．
        info (pl.DataFrame): 列 verify_asked, verify_answer を持つ表（truth と同じ行の順）．

    Returns:
        pl.DataFrame: 列 set, truth_odaibun, qwen_yes, n, weighted を持つ表．
    """
    j = CATEGORIES.index("format_odaibun")
    asked = info["verify_asked"].to_numpy()
    ans = info["verify_answer"].fill_null(False).to_numpy()
    w = truth["w"].to_numpy()
    rows = []
    for s, mask in {"real": (truth["set"] != "eval").to_numpy(), "eval": (truth["set"] == "eval").to_numpy()}.items():
        for tv in (True, False):
            for qv in (True, False):
                sel = mask & asked & (t[:, j] == tv) & (ans == qv)
                tot = w[mask].sum()
                rows.append({"set": s, "truth_odaibun": tv, "qwen_yes": qv, "n": int(sel.sum()), "share": float(w[sel].sum() / tot)})
    return pl.DataFrame(rows)


def fmt_cell(r: dict, k: str, ci: bool) -> str:
    """表のセルの文字列を作る（値と，必要なら95%区間）．

    Args:
        r (dict): metrics の1行．
        k (str): "precision" または "recall"．
        ci (bool): 区間を付けるか．

    Returns:
        str: セルの文字列．
    """
    v = r[k]
    if v is None or np.isnan(v):
        return "—"
    return f"{v:.2f}（{r[k + '_lo']:.2f}–{r[k + '_hi']:.2f}）" if ci else f"{v:.2f}"


def write_tables(df: pl.DataFrame) -> None:
    """レポートに貼る Markdown の表（集合ごと，カテゴリ×判断者の適合率・再現率）を書き出す．

    Args:
        df (pl.DataFrame): evaluate の戻り値．

    Returns:
        None
    """
    out = []
    for s, ci in [("real", True), ("eval", False)]:
        out.append(f"## {s}\n")
        judges = ["qwen", "qwen_verify", "student", "full"]
        head = "| 軸 | カテゴリ | 正例 | " + " | ".join(f"{JUDGE_JA[j]} 適合率 | {JUDGE_JA[j]} 再現率" for j in judges) + " |"
        out += [head, "| :--- | :--- | ---: |" + " ---: |" * (2 * len(judges))]
        for c in CATEGORIES:
            cells = []
            n_true = None
            for j in judges:
                r = df.filter((pl.col("set") == s) & (pl.col("judge") == j) & (pl.col("category") == c)).to_dicts()
                if not r:
                    cells += ["—", "—"]
                    continue
                n_true = r[0]["n_true"]
                cells += [fmt_cell(r[0], "precision", ci), fmt_cell(r[0], "recall", ci)]
            out.append(f"| {AXIS[c]} | {JA[c]} | {n_true} | " + " | ".join(cells) + " |")
        out.append("")
    with open(os.path.join(OUTPUT_DIR, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def plot(df: pl.DataFrame, s: str) -> None:
    """カテゴリごとの適合率・再現率を判断者ごとに並べた図を作る．

    Args:
        df (pl.DataFrame): evaluate の戻り値．
        s (str): 評価の集合（real / eval）．

    Returns:
        None
    """
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    judges = ["qwen", "qwen_verify", "student", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
    y = np.arange(len(CATEGORIES))[::-1]
    offs = {"qwen": 0.24, "qwen_verify": 0.08, "student": -0.08, "full": -0.24}
    for ax, k, name in zip(axes, ["precision", "recall"], ["適合率", "再現率"]):
        for j in judges:
            d = df.filter((pl.col("set") == s) & (pl.col("judge") == j))
            idx = [CATEGORIES.index(c) for c in d["category"]]
            yy = y[idx] + offs[j]
            v = d[k].to_numpy()
            if s == "real":
                err = np.abs(np.stack([v - d[f"{k}_lo"].to_numpy(), d[f"{k}_hi"].to_numpy() - v]))
                ax.errorbar(v, yy, xerr=err, fmt="none", ecolor=COLORS[j], elinewidth=1, alpha=0.6)
            ax.scatter(v, yy, color=COLORS[j], marker=MARKERS[j], s=36, label=JUDGE_JA[j], zorder=3, edgecolor="white", lw=0.8)
        for b in [3.5, 6.5]:
            ax.axhline(len(CATEGORIES) - 1 - b, color="#999999", lw=0.8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(name)
        ax.grid(axis="x", color="#dddddd", lw=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([JA[c] for c in CATEGORIES])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=4, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUTPUT_DIR, f"fig_pr_{s}.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    """集計し，表と図を保存する．

    Returns:
        None
    """
    truth, t, mats, info = build()
    df = evaluate(truth, t, mats)
    df.write_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))
    ov = overall(truth, t, mats)
    ov.write_csv(os.path.join(OUTPUT_DIR, "overall.csv"))
    vs = verify_stage(truth, t, mats, info)
    vs.write_csv(os.path.join(OUTPUT_DIR, "verify_stage.csv"))
    write_tables(df)
    for s in ["real", "eval"]:
        plot(df, s)
    with pl.Config(tbl_rows=100, tbl_cols=20):
        print(ov)
        print(vs)


if __name__ == "__main__":
    main()
