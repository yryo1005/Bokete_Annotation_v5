"""文の属性 v6 の目視のラベルを用いて，定義の変化，Qwen の教師としての精度，生徒で学べるかを評価する．

評価の内容：
1. 定義の変化：v5 の目視のお題文ありと，v6 の属性から規則 R0 / R1 / R1b / R2 で導いたお題文ありの一致．
2. Qwen3-VL-32B（指示 V6）の属性ごとの精度と，Qwen の属性から導いたお題文ありの精度．
3. 生徒の交差検証（5分割）：画像の埋め込み（SigLIP2，1152次元）のロジスティック回帰で属性ごとに学習する．
   文の形は，書き起こした文（Qwen の書き起こし）の文字 n-gram のロジスティック回帰と，埋め込みと文の両方を用いたものも比べる．
4. 生徒の属性から導いたお題文ありを，パッケージ v5.3.0 の has_odai_text と比べる．

全データの推定値は ex025 の600枚の事後層別の重みで求め，属性ごとの値は1,620枚（重みなし）でも求める．
出力：outputs/ex026_text_attr_v6/eval/*.csv，fig_*.png，evaluate.log

実行例:
    .venv_annotate/bin/python programs/ex026_text_attr_v6/evaluate.py
"""
import json
import os
import sys

import matplotlib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (EX024, FORM_NAMES, FORMS, LAYER_NAMES, LAYERS, OUTPUT_DIR, add_derived, load_all,  # noqa: E402
                  load_embeddings)

plt.rcParams["font.family"] = ["IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "DejaVu Sans"]
OUT = os.path.join(OUTPUT_DIR, "eval")
QWEN_DIR = os.path.join(OUTPUT_DIR, "qwen", "Qwen3-VL-32B-Instruct-FP8__802816")
EMB_CACHE = os.path.join(OUTPUT_DIR, "emb_1620.npy")
OCR2B = os.path.join(OUTPUT_DIR, "qwen", "Qwen3-VL-2B-Instruct__802816", "OCR.csv")
YOMI = os.path.join(OUTPUT_DIR, "ocr", "yomitoku.csv")
N_SPLITS = 5
SEED = 0
N_BOOT = 1000
RULES = ["odai_R0", "odai_R1", "odai_R1b", "odai_R2"]
# 学習・評価する属性（層の有無，層ごとの形，層をまとめた形）
LAYER_ATTRS = [f"L_{l}" for l in LAYERS]
FORM_ATTRS = [f"F_{l}_{f}" for l in ["o", "m"] for f in FORMS]
MERGED_ATTRS = [f"A_{f}" for f in FORMS]


def prf(y: np.ndarray, p: np.ndarray, w: np.ndarray = None) -> dict:
    """重み付きの適合率・再現率・F1 を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解（bool）．
        p (np.ndarray): 形状 (N,) の予測（bool）．
        w (np.ndarray): 形状 (N,) の重み．None なら1．

    Returns:
        dict: precision, recall, f1, n_pos（正解の正例の重みの和）, n（重みの和）．
    """
    w = np.ones(len(y)) if w is None else w
    tp = float((w * (y & p)).sum())
    fp = float((w * (~y & p)).sum())
    fn = float((w * (y & ~p)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    rec = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "n_pos": float((w * y).sum()), "n": float(w.sum())}


def boot_f1(y: np.ndarray, p: np.ndarray, w: np.ndarray, strata: np.ndarray) -> tuple:
    """層ごとの復元抽出で F1 の95% 区間を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の予測．
        w (np.ndarray): 形状 (N,) の重み．
        strata (np.ndarray): 形状 (N,) の層の名前．

    Returns:
        tuple[float, float]: 2.5% 点と97.5% 点．
    """
    rng = np.random.default_rng(SEED)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(prf(y[idx], p[idx], w[idx])["f1"])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def kappa(a: np.ndarray, b: np.ndarray, w: np.ndarray = None) -> float:
    """重み付きの Cohen の κ を求める．

    Args:
        a (np.ndarray): 形状 (N,) のラベル（bool）．
        b (np.ndarray): 形状 (N,) のラベル（bool）．
        w (np.ndarray): 形状 (N,) の重み．None なら1．

    Returns:
        float: κ．
    """
    w = np.ones(len(a)) if w is None else w
    w = w / w.sum()
    po = float((w * (a == b)).sum())
    pa, pb = float((w * a).sum()), float((w * b).sum())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_qwen() -> pl.DataFrame:
    """Qwen の V6 の属性と書き起こし（V6 と OCR）を読み込む．

    Returns:
        pl.DataFrame: 列 rep_odaiId, q_L_*, q_F_*，q_A_*，q_odai_R*，text_o, text_m（V6 の書き起こし），ocr_o, ocr_m（OCR の書き起こし）を持つ表．
    """
    v6 = pl.read_csv(os.path.join(QWEN_DIR, "V6.csv"))
    cols = LAYER_ATTRS + FORM_ATTRS
    v6 = v6.select("rep_odaiId", *[pl.col(c).alias("q_" + c) for c in cols], pl.col("text_o").fill_null(""), pl.col("text_m").fill_null(""))
    v6 = add_derived(v6, prefix="q_")
    ocr = pl.read_csv(os.path.join(QWEN_DIR, "OCR.csv")).select("rep_odaiId", pl.col("text_o").fill_null("").alias("ocr_o"),
                                                                  pl.col("text_m").fill_null("").alias("ocr_m"))
    v6 = v6.join(ocr, on="rep_odaiId", how="left")
    if os.path.exists(OCR2B):
        o2 = pl.read_csv(OCR2B).select("rep_odaiId", pl.col("text_o").fill_null("").alias("ocr2b_o"), pl.col("text_m").fill_null("").alias("ocr2b_m"))
        v6 = v6.join(o2, on="rep_odaiId", how="left")
    if os.path.exists(YOMI):
        v6 = v6.join(pl.read_csv(YOMI).select("rep_odaiId", pl.col("text").fill_null("").alias("yomi")), on="rep_odaiId", how="left")
    return v6


def load_emb(ids: list) -> np.ndarray:
    """対象の画像の埋め込みを読み込む（初回はパッケージの出力から集めて保存する）．

    Args:
        ids (list[int]): 代表の画像のお題IDのリスト．

    Returns:
        np.ndarray: 形状 (len(ids), 1152) の配列．
    """
    if os.path.exists(EMB_CACHE):
        e = np.load(EMB_CACHE)
        if len(e) == len(ids):
            return e
    e = load_embeddings(ids)
    np.save(EMB_CACHE, e)
    return e


def oof_logreg(x, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """5分割の交差検証でロジスティック回帰の分割外の確率を求める．

    Args:
        x (np.ndarray | scipy.sparse.csr_matrix): 形状 (N, D) の特徴．
        y (np.ndarray): 形状 (N,) の正解（bool）．
        c (float): 正則化の強さの逆数．

    Returns:
        np.ndarray: 形状 (N,) の分割外の確率．正例が5枚未満の場合は全て0．
    """
    p = np.zeros(len(y))
    if y.sum() < N_SPLITS:
        return p
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y):
        m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        m.fit(x[tr], y[tr])
        p[te] = m.predict_proba(x[te])[:, 1]
    return p


def ap_score(y: np.ndarray, s: np.ndarray) -> float:
    """平均適合率（AP）を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解．
        s (np.ndarray): 形状 (N,) の得点．

    Returns:
        float: AP．
    """
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def text_features(d: pl.DataFrame, col_o: str, col_m: str):
    """書き起こした文の文字 n-gram の特徴を作る（層ごとに別の語彙とし，層の有無を加える）．

    Args:
        d (pl.DataFrame): 書き起こしの列を持つ表．
        col_o (str): 重ねた文字の書き起こしの列名．
        col_m (str): 余白の文字の書き起こしの列名．

    Returns:
        scipy.sparse.csr_matrix: 形状 (N, D) の特徴．
    """
    mats = []
    for col in [col_o, col_m]:
        txt = d[col].fill_null("").to_list()
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
        mats.append(vec.fit_transform(txt))
        mats.append(csr_matrix(np.array([[len(t) > 0] for t in txt], dtype=float)))
    return hstack(mats).tocsr()


def eval_definition(d: pl.DataFrame, e25: pl.DataFrame) -> pl.DataFrame:
    """v5 のお題文ありと，v6 から導いたお題文ありの一致を求める．

    Args:
        d (pl.DataFrame): 1,620枚の表．
        e25 (pl.DataFrame): ex025 の600枚（重み付き）の表．

    Returns:
        pl.DataFrame: 列 rule, set, agree, kappa, rate_v5, rate_v6, v5_only, v6_only を持つ表．
    """
    rows = []
    for rule in RULES:
        for name, t, w in [("all_1620", d, None), ("population", e25, e25["w"].to_numpy())]:
            a, b = t["label_v5"].to_numpy(), t[rule].to_numpy()
            ww = np.ones(len(a)) if w is None else w / w.sum()
            rows.append({"rule": rule, "set": name, "agree": float((ww * (a == b)).sum() / ww.sum()), "kappa": kappa(a, b, w),
                         "rate_v5": float((ww * a).sum() / ww.sum()), "rate_v6": float((ww * b).sum() / ww.sum()),
                         "v5_only": float((ww * (a & ~b)).sum() / ww.sum()), "v6_only": float((ww * (~a & b)).sum() / ww.sum())})
    return pl.DataFrame(rows)


def eval_attrs(d: pl.DataFrame, pred_prefix: str, attrs: list, method: str, scores: dict = None) -> list:
    """属性ごとの予測の精度を求める（1,620枚と，ex025 の600枚の重み付き）．

    Args:
        d (pl.DataFrame): 正解の列と予測の列（pred_prefix + 属性）を持つ表．
        pred_prefix (str): 予測の列名の接頭辞．
        attrs (list[str]): 属性の列名のリスト．
        method (str): 手法の名前．
        scores (dict): 属性ごとの得点（AP を求める．None なら求めない）．

    Returns:
        list[dict]: 行のリスト（列 method, attr, set, precision, recall, f1, n_pos, n, ap）．
    """
    rows = []
    is25 = (d["source"] == "ex025").to_numpy()
    w25 = d["w"].fill_null(0).to_numpy()
    for a in attrs:
        y, p = d[a].to_numpy(), d[pred_prefix + a].to_numpy()
        s = None if scores is None else scores[a]
        for name, mask, w in [("all_1620", np.ones(len(y), bool), None), ("population", is25, w25[is25])]:
            r = prf(y[mask], p[mask], w)
            r["ap"] = ap_score(y[mask], s[mask]) if s is not None and name == "all_1620" else float("nan")
            rows.append({"method": method, "attr": a, "set": name, **r})
    return rows


def student_predictions(d: pl.DataFrame, emb: np.ndarray) -> tuple:
    """生徒の分割外の予測を求める（層は埋め込み，形は埋め込み・文・両方）．

    Args:
        d (pl.DataFrame): 1,620枚の表（Qwen の書き起こしを含む）．
        emb (np.ndarray): 形状 (N, 1152) の埋め込み．

    Returns:
        tuple[dict, dict]: 手法ごとの {属性: 確率} の辞書と，手法ごとの予測の表（列 pred_{属性}）の辞書．
    """
    x_emb = StandardScaler().fit_transform(emb)
    x_txt = text_features(d, "text_o", "text_m")
    x_ocr = text_features(d, "ocr_o", "ocr_m")
    x_both = hstack([csr_matrix(x_emb * 0.1), x_txt]).tocsr()
    feats = {"emb": (x_emb, 0.05), "text_v6": (x_txt, 4.0), "text_ocr": (x_ocr, 4.0), "emb+text_v6": (x_both, 1.0)}
    if "ocr2b_o" in d.columns:
        feats["text_ocr2b"] = (text_features(d, "ocr2b_o", "ocr2b_m"), 4.0)
        feats["emb+text_ocr2b"] = (hstack([csr_matrix(x_emb * 0.1), feats["text_ocr2b"][0]]).tocsr(), 1.0)
    if "yomi" in d.columns:
        # YomiToku は層を区別しないため，全ての文字を1つの文字列とし，埋め込みと合わせる
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
        x_yomi = vec.fit_transform(d["yomi"].fill_null("").to_list())
        feats["emb+text_yomi"] = (hstack([csr_matrix(x_emb * 0.1), x_yomi]).tocsr(), 1.0)
    probs = {}
    for name, (x, c) in feats.items():
        attrs = LAYER_ATTRS + FORM_ATTRS if name == "emb" else FORM_ATTRS
        probs[name] = {a: oof_logreg(x, d[a].to_numpy(), c) for a in tqdm(attrs, desc=name, leave=False)}
    return probs


def to_pred(d: pl.DataFrame, probs: dict, layer_src: str = "emb") -> pl.DataFrame:
    """確率を0.5で真偽にし，層の予測と形の予測を合わせ，お題文ありを導く．

    層は layer_src の手法の予測とし，形の予測は「その層があると予測した場合」に限る．

    Args:
        d (pl.DataFrame): 1,620枚の表．
        probs (dict): {属性: 確率}（形の属性を含む）．
        layer_src (str): 未使用（層は常に埋め込みの生徒）．

    Returns:
        pl.DataFrame: d に列 pred_{属性} と pred_A_*，pred_odai_R* を加えた表．
    """
    cols = {}
    for a in LAYER_ATTRS + FORM_ATTRS:
        cols["pred_" + a] = probs[a] >= 0.5
    t = d.with_columns(*[pl.Series(k, v) for k, v in cols.items()])
    return add_derived(t, prefix="pred_")


def eval_odai(d: pl.DataFrame, preds: dict) -> pl.DataFrame:
    """導いたお題文ありの精度を，v6 の目視（R1 など）と v5 の目視に対して求める．

    Args:
        d (pl.DataFrame): 1,620枚の表（列 v53 を含む）．
        preds (dict): {手法: お題文ありの予測（規則ごとの辞書 {rule: np.ndarray}）}．

    Returns:
        pl.DataFrame: 列 method, rule, truth, set, precision, recall, f1, lo, hi を持つ表．
    """
    rows = []
    is25 = (d["source"] == "ex025").to_numpy()
    w25 = d["w"].fill_null(0).to_numpy()[is25]
    st25 = d["stratum2"].fill_null("").to_numpy()[is25]
    for method, by_rule in preds.items():
        for rule, p in by_rule.items():
            for truth in [rule, "label_v5"]:
                y = d[truth].to_numpy()
                r = prf(y[is25], p[is25], w25)
                lo, hi = boot_f1(y[is25], p[is25], w25, st25)
                rows.append({"method": method, "rule": rule, "truth": "v6" if truth != "label_v5" else "v5", "set": "population", **r,
                             "lo": lo, "hi": hi})
                r = prf(y, p)
                rows.append({"method": method, "rule": rule, "truth": "v6" if truth != "label_v5" else "v5", "set": "all_1620", **r,
                             "lo": float("nan"), "hi": float("nan")})
    return pl.DataFrame(rows)


def load_v53(ids: list) -> np.ndarray:
    """パッケージ v5.3.0 の has_odai_text を読み込む．

    Args:
        ids (list[int]): 代表の画像のお題IDのリスト．

    Returns:
        np.ndarray: 形状 (len(ids),) の bool．
    """
    # ex024 の画像には代表でない画像も含まれるため，お題IDで引く
    a = pl.scan_parquet(EX024.ANNOTATION).filter(pl.col("odaiId").is_in(pl.Series(ids).implode())).select(
        "odaiId", "has_odai_text").collect()
    m = dict(zip(a["odaiId"].to_list(), a["has_odai_text"].to_list()))
    return np.array([bool(m[i]) for i in ids])


def plot_attrs(m: pl.DataFrame, out: str) -> None:
    """属性ごとの F1（1,620枚）を手法ごとに描く．

    Args:
        m (pl.DataFrame): eval_attrs の行を集めた表．
        out (str): 保存先．

    Returns:
        None
    """
    t = m.filter(pl.col("set") == "all_1620")
    attrs = LAYER_ATTRS + FORM_ATTRS
    label = {**{f"L_{l}": LAYER_NAMES[l] for l in LAYERS},
             **{f"F_{l}_{f}": f"{LAYER_NAMES[l][:2]}:{FORM_NAMES[f]}" for l in ["o", "m"] for f in FORMS}}
    methods = ["qwen", "emb", "text_ocr", "emb+text_v6", "emb+text_ocr2b", "emb+text_yomi"]
    methods = [mm for mm in methods if t.filter(pl.col("method") == mm).height > 0]
    names = {"qwen": "Qwen-32B（教師）", "emb": "生徒：埋め込み", "text_ocr": "生徒：文（32B の書き起こし）",
             "emb+text_v6": "生徒：埋め込み＋文（32B の V6 の書き起こし）", "emb+text_ocr2b": "生徒：埋め込み＋文（2B の書き起こし）",
             "emb+text_yomi": "生徒：埋め込み＋文（YomiToku）"}
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    width = 0.8 / len(methods)
    for k, (meth, col) in enumerate(zip(methods, colors)):
        vals = []
        for a in attrs:
            r = t.filter((pl.col("method") == meth) & (pl.col("attr") == a))
            vals.append(r["f1"][0] if r.height else np.nan)
        ax.bar(np.arange(len(attrs)) + (k - (len(methods) - 1) / 2) * width, vals, width, label=names[meth], color=col)
    npos = [int(t.filter((pl.col("method") == "qwen") & (pl.col("attr") == a))["n_pos"][0]) for a in attrs]
    ax.set_xticks(np.arange(len(attrs)))
    ax.set_xticklabels([f"{label[a]}\n(n={n})" for a, n in zip(attrs, npos)], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("F1（1,620枚）")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def learning_curve(x, y: np.ndarray, c: float, fracs: list) -> list:
    """学習に用いる画像の割合を変えて，分割外の F1（1,620枚，重みなし）を求める．

    各分割の学習用の画像から割合 frac を無作為に選んで学習し，検証用の画像を予測する．

    Args:
        x (np.ndarray | scipy.sparse.csr_matrix): 形状 (N, D) の特徴．
        y (np.ndarray): 形状 (N,) の正解．
        c (float): 正則化の強さの逆数．
        fracs (list[float]): 学習の画像の割合のリスト．

    Returns:
        list[dict]: 行のリスト（列 frac, n_train, f1）．
    """
    rows = []
    rng = np.random.default_rng(SEED)
    for frac in fracs:
        p = np.zeros(len(y))
        n_train = 0
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y):
            sub = rng.choice(tr, int(len(tr) * frac), replace=False)
            m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(x[sub], y[sub])
            p[te] = m.predict_proba(x[te])[:, 1]
            n_train = len(sub)
        rows.append({"frac": frac, "n_train": n_train, "f1": prf(y, p >= 0.5)["f1"]})
    return rows


def plot_curve(lc: pl.DataFrame, out: str) -> None:
    """学習曲線を描く．

    Args:
        lc (pl.DataFrame): 列 target, feature, n_train, f1 を持つ表．
        out (str): 保存先．

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = {"emb": "埋め込み", "emb+text_v6": "埋め込み＋文（32B の書き起こし）"}
    tnames = {"odai_R1b": "お題文あり（R1b，直接学習）", "L_o": "重ねた文字", "L_m": "余白の文字", "A_q": "問い", "A_x": "その他の文"}
    for (target, feat), g in lc.group_by(["target", "feature"], maintain_order=True):
        g = g.sort("n_train")
        ax.plot(g["n_train"], g["f1"], marker="o", ls="-" if feat == "emb" else "--", label=f"{tnames[target]}：{names[feat]}")
    ax.set_xlabel("学習の画像の数")
    ax.set_ylabel("F1（分割外，1,620枚）")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """全ての評価を行い，表と図を保存する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    d = load_all().join(load_qwen(), on="rep_odaiId", how="left")
    ids = d["rep_odaiId"].to_list()
    d = d.with_columns(pl.Series("v53", load_v53(ids)))
    e25 = d.filter(pl.col("source") == "ex025")
    log = {}

    # 1. 定義の変化
    dfn = eval_definition(d, e25)
    dfn.write_csv(os.path.join(OUT, "definition_shift.csv"))
    diff = d.filter(pl.col("label_v5") != pl.col("odai_R1")).select("no", "rep_odaiId", "source", "label_v5", "code", "memo")
    diff.write_csv(os.path.join(OUT, "definition_diff_R1.csv"))
    diff = d.filter(pl.col("label_v5") != pl.col("odai_R1b")).select("no", "rep_odaiId", "source", "label_v5", "code", "memo")
    diff.write_csv(os.path.join(OUT, "definition_diff_R1b.csv"))
    log["code_freq"] = {c: float(d[c].mean()) for c in LAYER_ATTRS + FORM_ATTRS + MERGED_ATTRS + RULES}
    log["both_o_m"] = int((d["L_o"] & d["L_m"]).sum())
    log["both_o_m_diff_form"] = int(sum(1 for r in d.filter(pl.col("L_o") & pl.col("L_m")).iter_rows(named=True)
                                        if any(r[f"F_o_{f}"] != r[f"F_m_{f}"] for f in FORMS)))

    # 2. Qwen
    rows = eval_attrs(d, "q_", LAYER_ATTRS + FORM_ATTRS + MERGED_ATTRS, "qwen")

    # 3. 生徒
    emb = load_emb(ids)
    probs = student_predictions(d, emb)
    for meth, pr in probs.items():
        t = d.with_columns(*[pl.Series("pred_" + a, pr[a] >= 0.5) for a in pr])
        attrs = list(pr)
        rows += eval_attrs(t, "pred_", attrs, meth, pr)
    m = pl.DataFrame(rows)
    m.write_csv(os.path.join(OUT, "attrs.csv"))
    plot_attrs(m, os.path.join(OUT, "fig_attrs.png"))

    # 4. 導いたお題文あり（生徒：層は埋め込み，形は手法ごと）
    odai_preds = {"v5.3.0": {r: d["v53"].to_numpy() for r in RULES},
                  "qwen": {r: d["q_" + r].to_numpy() for r in RULES}}
    for meth in probs:
        pr = {**probs["emb"], **{a: probs[meth][a] for a in FORM_ATTRS}}
        t = to_pred(d, pr)
        odai_preds["student_" + meth] = {r: t["pred_" + r].to_numpy() for r in RULES}
    # 比較：属性を経ずにお題文あり（規則ごとの正解）を直接学習した生徒
    x_emb = StandardScaler().fit_transform(emb)
    x_both = hstack([csr_matrix(x_emb * 0.1), text_features(d, "text_o", "text_m")]).tocsr()
    odai_preds["direct_emb"] = {r: oof_logreg(x_emb, d[r].to_numpy(), 0.05) >= 0.5 for r in RULES}
    odai_preds["direct_emb+text_v6"] = {r: oof_logreg(x_both, d[r].to_numpy(), 1.0) >= 0.5 for r in RULES}
    od = eval_odai(d, odai_preds)

    # 5. 学習曲線
    lc = []
    for target in ["odai_R1b", "L_o", "L_m", "A_q", "A_x"]:
        for feat, (x, c) in {"emb": (x_emb, 0.05), "emb+text_v6": (x_both, 1.0)}.items():
            lc += [{"target": target, "feature": feat, **r} for r in learning_curve(x, d[target].to_numpy(), c, [0.125, 0.25, 0.5, 0.75, 1.0])]
    lc = pl.DataFrame(lc)
    lc.write_csv(os.path.join(OUT, "learning_curve.csv"))
    plot_curve(lc, os.path.join(OUT, "fig_curve.png"))
    od.write_csv(os.path.join(OUT, "odai.csv"))
    json.dump(log, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)

    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    print(dfn)
    print(m.filter(pl.col("set") == "all_1620").select("method", "attr", "n_pos", "precision", "recall", "f1", "ap"))
    print(od.filter(pl.col("set") == "population").sort("truth", "rule", "method"))
    print(json.dumps(log, ensure_ascii=False))


if __name__ == "__main__":
    main()
