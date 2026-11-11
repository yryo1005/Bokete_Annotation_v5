"""お題文ありを判定する生徒を学習し，評価の1,620枚の分割外の確率を保存する．

手法（名前：特徴）：
- 方法1（v5.3.0 の確率との組み合わせ．1,620枚の5分割の交差検証）
  - p_only：v5.3.0 の p_has_odai_text のロジット（1次元．閾値の較正にあたる）
  - pkg：v5.3.0 の確率 P_COLS のロジット（15次元）
  - pkg+emb，pkg+yomi，pkg+ocr32b，pkg+emb+yomi，pkg+emb+ocr32b：埋め込み，YomiToku の書き起こし，Qwen-32B の書き起こし（ex026 の OCR）の TF-IDF を加える
  - emb+yomi：比較（v5.3.0 の確率を用いない．ex026 の直接学習に相当）
- 方法2（文の形を問う教師．接頭辞 {p} は指示の名前 form（層を問わない）/ form2（文ごとに場所と形を問う））
  - {p}_teacher：評価の画像での教師の形 T_{形} と，余白の文字の生徒（1,620枚の目視で学習した埋め込みの生徒の分割外）から規則で導く
  - {p}_teacher_own：教師自身の文ごとの場所と形から R1b で導く（form2 のみ）
  - {p}_student：プールの教師ラベルで学習した埋め込みの生徒の形と，余白の文字の生徒から規則で導く
  - {p}_student_direct：プールの教師のお題文ありを，埋め込みの生徒で直接学習する
  - pkg+{p}：v5.3.0 の確率に，プールの生徒の形・お題文ありと余白の文字の確率を加えて1,620枚で組み合わせる
  - pkg+emb+yomi+{p}：さらに埋め込みと YomiToku の書き起こしを加える
  プールの余白の文字の生徒は1,620枚の全ての目視で学習するため，{p}_student_direct（form）と pkg+{p} では評価の画像の目視の
  余白の文字が間接的にプールの教師ラベルに入る．

規則（R1b の近似）：T_q ∨ T_i ∨ T_t ∨ (T_x ∧ ¬T_k ∧ 余白の文字あり)．教師は層を区別しないため，その他の文と空欄を含む文が
余白の文字にあるかは区別できない．

正解：v6 の目視の odai_R1b と，v5 の目視の label_v5 の2通り（それぞれで学習する）．
正則化の係数 C は，外側の分割の学習用の画像の内側の4分割の F1 で格子から選ぶ（入れ子の交差検証）．
外側の分割の乱数の種を SEEDS で変える．

出力：outputs/ex027_odai_text_improve/preds/{手法}/{正解}/{seed}/pred.csv（列 rep_odaiId, prob, pred）と meta.json

実行例:
    .venv_annotate/bin/python programs/ex027_odai_text_improve/train.py
"""
import itertools
import json
import os
import sys

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX026_EVAL, FORMS, add_derived, OUTPUT_DIR, P_COLS, build_pool, load_eval, load_pool_embeddings  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 5
N_INNER = 4
TRUTHS = {"v6": "odai_R1b", "v5": "label_v5"}
C_GRID = {"dense": [0.01, 0.03, 0.1, 0.3, 1.0], "sparse": [0.25, 1.0, 4.0]}
QWEN_DIR = os.path.join(OUTPUT_DIR, "qwen")
PRED_DIR = os.path.join(OUTPUT_DIR, "preds")
# プールの生徒（形と，余白の文字から導いたお題文あり）の正則化の係数（プールの5分割で一度だけ選ぶ）
POOL_C_GRID = [0.01, 0.03, 0.1, 0.3]
# 文の形を問う教師の指示（FORM：層を問わない，FORM2：文ごとに場所と形を問う）
TEACHERS = ["FORM", "FORM2"]


def logit(p: np.ndarray) -> np.ndarray:
    """確率をロジットに変える（0と1は丸める）．

    Args:
        p (np.ndarray): 形状 (N, K) の確率．

    Returns:
        np.ndarray: 形状 (N, K) のロジット．
    """
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def f1(y: np.ndarray, p: np.ndarray) -> float:
    """重みなしの F1 を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解（bool）．
        p (np.ndarray): 形状 (N,) の予測（bool）．

    Returns:
        float: F1．
    """
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0


def fit_predict(x_tr, y_tr: np.ndarray, x_te, c: float) -> np.ndarray:
    """ロジスティック回帰（正例と負例の重みを釣り合わせる）を学習し，確率を返す．

    Args:
        x_tr (np.ndarray | csr_matrix): 形状 (N_tr, D) の学習の特徴．
        y_tr (np.ndarray): 形状 (N_tr,) の正解．
        x_te (np.ndarray | csr_matrix): 形状 (N_te, D) の予測する特徴．
        c (float): 正則化の強さの逆数．

    Returns:
        np.ndarray: 形状 (N_te,) の確率．
    """
    m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(x_tr, y_tr)
    return m.predict_proba(x_te)[:, 1]


def choose_c(x, y: np.ndarray, grid: list, seed: int) -> float:
    """内側の分割の F1（閾値0.5）で正則化の係数を選ぶ．

    Args:
        x (np.ndarray | csr_matrix): 形状 (N, D) の特徴．
        y (np.ndarray): 形状 (N,) の正解．
        grid (list[float]): 係数の候補．
        seed (int): 分割の乱数の種．

    Returns:
        float: 選んだ係数．
    """
    if len(grid) == 1:
        return grid[0]
    scores = []
    for c in grid:
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(N_INNER, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            p[te] = fit_predict(x[tr], y[tr], x[te], c)
        scores.append(f1(y, p >= 0.5))
    return grid[int(np.argmax(scores))]


def nested_oof(x, y: np.ndarray, grid: list, seed: int) -> tuple:
    """入れ子の交差検証で分割外の確率を求める．

    Args:
        x (np.ndarray | csr_matrix): 形状 (N, D) の特徴．
        y (np.ndarray): 形状 (N,) の正解．
        grid (list[float]): 係数の候補．
        seed (int): 外側の分割の乱数の種．

    Returns:
        tuple[np.ndarray, list[float]]: 形状 (N,) の分割外の確率と，外側の分割ごとに選んだ係数．
    """
    p = np.zeros(len(y))
    cs = []
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        c = choose_c(x[tr], y[tr], grid, seed)
        p[te] = fit_predict(x[tr], y[tr], x[te], c)
        cs.append(c)
    return p, cs


def tfidf(texts: list) -> csr_matrix:
    """文字の1〜3-gram の TF-IDF と，文字があるかの列を作る．

    Args:
        texts (list[str]): 書き起こしのリスト．

    Returns:
        csr_matrix: 形状 (N, D) の特徴．
    """
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    return hstack([vec.fit_transform(texts), csr_matrix(np.array([[len(t) > 0] for t in texts], dtype=float))]).tocsr()


def form_rule(t: dict, l_m: np.ndarray) -> np.ndarray:
    """文の形と余白の文字の有無から，お題文あり（R1b の近似）を導く．

    Args:
        t (dict[str, np.ndarray]): 形ごとの形状 (N,) の真偽．
        l_m (np.ndarray): 形状 (N,) の余白の文字の有無．

    Returns:
        np.ndarray: 形状 (N,) の真偽．
    """
    return t["q"] | t["i"] | t["t"] | (t["x"] & ~t["k"] & l_m)


def load_form(prompt: str, target: str) -> pl.DataFrame:
    """文の形を問う教師の結果を読み込む．FORM2 は教師自身の層と形から規則 R1b のお題文あり（列 odai_R1b）も導く．

    Args:
        prompt (str): 指示の名前（FORM / FORM2）．
        target (str): eval か pool．

    Returns:
        pl.DataFrame | None: 列 rep_odaiId, T_{形}, text（FORM2 は odai_R1b も）を持つ表．ファイルがない場合は None．
    """
    path = os.path.join(QWEN_DIR, f"{prompt}_{target}.csv")
    if not os.path.exists(path):
        return None
    t = pl.read_csv(path, schema_overrides={"text": pl.Utf8, "text_all": pl.Utf8, "raw": pl.Utf8}).with_columns(pl.col("text").fill_null(""))
    if prompt == "FORM2":
        t = add_derived(t).select("rep_odaiId", *[f"T_{f}" for f in FORMS], "text", "odai_R1b")
    return t


def margin_student(emb_eval: np.ndarray, l_m: np.ndarray, emb_pool: np.ndarray) -> tuple:
    """余白の文字の有無を，1,620枚の目視で学習した埋め込みの生徒で予測する（評価は分割外，プールは全1,620枚で学習）．

    Args:
        emb_eval (np.ndarray): 形状 (1620, 1152) の標準化した埋め込み．
        l_m (np.ndarray): 形状 (1620,) の目視の余白の文字の有無．
        emb_pool (np.ndarray | None): 形状 (M, 1152) の標準化した埋め込み．

    Returns:
        tuple[np.ndarray, np.ndarray | None]: 評価の分割外の確率と，プールの確率．
    """
    p_eval = EX026_EVAL.oof_logreg(emb_eval, l_m, 0.05)
    p_pool = None if emb_pool is None else fit_predict(emb_eval, l_m, emb_pool, 0.05)
    return p_eval, p_pool


def pool_students(emb_pool: np.ndarray, targets: dict, emb_eval: np.ndarray) -> dict:
    """プールの教師ラベルで埋め込みの生徒を学習し，評価の画像の確率を求める（係数はプールの5分割の F1 で選ぶ）．

    Args:
        emb_pool (np.ndarray): 形状 (M, 1152) の標準化した埋め込み．
        targets (dict[str, np.ndarray]): 名前ごとの形状 (M,) の教師ラベル．
        emb_eval (np.ndarray): 形状 (1620, 1152) の標準化した埋め込み．

    Returns:
        dict: 名前ごとの {"prob": 形状 (1620,) の確率, "C": 係数, "pool_cv_f1": プールの交差検証の F1}．
    """
    out = {}
    for name, y in tqdm(targets.items(), desc="pool students", leave=False):
        scores = []
        for c in POOL_C_GRID:
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
                p[te] = fit_predict(emb_pool[tr], y[tr], emb_pool[te], c)
            scores.append(f1(y, p >= 0.5))
        c = POOL_C_GRID[int(np.argmax(scores))]
        out[name] = {"prob": fit_predict(emb_pool, y, emb_eval, c), "C": c, "pool_cv_f1": float(max(scores))}
    return out


def save(method: str, truth: str, seed: int, ids: list, prob: np.ndarray, pred: np.ndarray, meta: dict) -> None:
    """分割外の確率と予測を保存する．

    Args:
        method (str): 手法の名前．
        truth (str): 正解の名前（v6 / v5）．
        seed (int): 乱数の種．
        ids (list[int]): 代表の画像のお題ID．
        prob (np.ndarray): 形状 (N,) の確率．
        pred (np.ndarray): 形状 (N,) の予測（bool）．
        meta (dict): 条件のメタデータ．

    Returns:
        None
    """
    d = os.path.join(PRED_DIR, method, truth, str(seed))
    os.makedirs(d, exist_ok=True)
    pl.DataFrame({"rep_odaiId": ids, "prob": prob, "pred": pred}).write_csv(os.path.join(d, "pred.csv"))
    json.dump({"method": method, "truth": truth, "seed": seed, **meta}, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)


def done(method: str, truth: str, seed: int) -> bool:
    """結果が保存済みかを調べる．

    Args:
        method (str): 手法の名前．
        truth (str): 正解の名前．
        seed (int): 乱数の種．

    Returns:
        bool: 保存済みなら True．
    """
    return os.path.exists(os.path.join(PRED_DIR, method, truth, str(seed), "pred.csv"))


def build_form_features(d: pl.DataFrame, x_emb: np.ndarray, ids: list, prompt: str) -> tuple:
    """方法2の教師と生徒の予測を作る（教師の結果がない場合は空）．

    手法の名前の接頭辞は指示の名前の小文字（form / form2）とする．
    - {p}_teacher：評価の画像での教師の形と，余白の文字の生徒から規則で導く
    - {p}_teacher_own：教師自身の層と形から R1b で導く（FORM2 のみ）
    - {p}_student：プールの教師ラベルで学習した形の生徒と，余白の文字の生徒から規則で導く
    - {p}_student_direct：プールの教師のお題文あり（FORM2 は教師自身の R1b，FORM は形と余白の文字の生徒から導いたもの）を直接学習する

    Args:
        d (pl.DataFrame): 評価の1,620枚の表．
        x_emb (np.ndarray): 形状 (1620, 1152) の標準化した埋め込み．
        ids (list[int]): 評価の画像のお題ID．
        prompt (str): 指示の名前（FORM / FORM2）．

    Returns:
        tuple[dict, np.ndarray | None, dict]: 規則で導く手法ごとの {"prob", "pred"}，組み合わせに加える形状 (1620, K) の確率の特徴，メタデータ．
    """
    p = prompt.lower()
    fe = load_form(prompt, "eval")
    methods, meta = {}, {}
    if fe is None:
        return methods, None, meta
    fe = pl.DataFrame({"rep_odaiId": ids}).join(fe, on="rep_odaiId", how="left")
    pm_eval, _ = margin_student(x_emb, d["L_m"].to_numpy(), None)
    r = form_rule({f: fe[f"T_{f}"].to_numpy() for f in FORMS}, pm_eval >= 0.5)
    methods[f"{p}_teacher"] = {"prob": r.astype(float), "pred": r}
    if "odai_R1b" in fe.columns:
        r = fe["odai_R1b"].to_numpy()
        methods[f"{p}_teacher_own"] = {"prob": r.astype(float), "pred": r}
    fp = load_form(prompt, "pool")
    if fp is None:
        return methods, None, meta
    pool_all = build_pool(fp.height, ids)
    emb_pool = load_pool_embeddings(pool_all["rep_odaiId"].to_list())
    keep = pool_all["rep_odaiId"].is_in(fp["rep_odaiId"].implode()).to_numpy()
    pool = pool_all.filter(keep).join(fp, on="rep_odaiId", how="left")
    scaler = StandardScaler().fit(emb_pool[keep])
    e_pool = scaler.transform(emb_pool[keep])
    e_eval = scaler.transform(EX026_EVAL.load_emb(ids))
    _, pm_pool = margin_student(e_eval, d["L_m"].to_numpy(), e_pool)
    t_pool = {f: pool[f"T_{f}"].to_numpy() for f in FORMS}
    targets = {f"T_{f}": t_pool[f] for f in FORMS}
    targets["odai"] = pool["odai_R1b"].to_numpy() if "odai_R1b" in pool.columns else form_rule(t_pool, pm_pool >= 0.5)
    st = pool_students(e_pool, targets, e_eval)
    r = form_rule({f: st[f"T_{f}"]["prob"] >= 0.5 for f in FORMS}, pm_eval >= 0.5)
    methods[f"{p}_student"] = {"prob": r.astype(float), "pred": r}
    methods[f"{p}_student_direct"] = {"prob": st["odai"]["prob"], "pred": st["odai"]["prob"] >= 0.5}
    feat = np.column_stack([st[f"T_{f}"]["prob"] for f in FORMS] + [st["odai"]["prob"], pm_eval])
    meta = {"teacher": prompt, "n_pool": pool.height,
            "pool_students": {k: {"C": v["C"], "pool_cv_f1": v["pool_cv_f1"]} for k, v in st.items()},
            "pool_rate": {k: float(v.mean()) for k, v in targets.items()}}
    return methods, logit(feat), meta


def main() -> None:
    """全ての手法・正解・乱数の種で分割外の確率を求めて保存する（保存済みは飛ばす）．

    Returns:
        None
    """
    d, emb = load_eval()
    ids = d["rep_odaiId"].to_list()
    x_emb = StandardScaler().fit_transform(emb)
    x_p = StandardScaler().fit_transform(logit(d.select(P_COLS).to_numpy()))
    x_p1 = StandardScaler().fit_transform(logit(d.select("p_has_odai_text").to_numpy()))
    x_yomi = tfidf(d["yomi"].fill_null("").to_list())
    x_ocr = hstack([tfidf(d["ocr_o"].fill_null("").to_list()), tfidf(d["ocr_m"].fill_null("").to_list())]).tocsr()
    e01 = csr_matrix(x_emb * 0.1)
    rule_methods, x_forms, form_meta = {}, {}, {}
    for prompt in TEACHERS:
        rm, xf, meta = build_form_features(d, x_emb, ids, prompt)
        rule_methods.update({k: (v, meta) for k, v in rm.items()})
        if xf is not None:
            x_forms[prompt.lower()] = StandardScaler().fit_transform(xf)
    feats = {"p_only": (x_p1, "dense"), "pkg": (x_p, "dense"), "pkg+emb": (np.hstack([x_p, x_emb * 0.1]), "dense"),
             "pkg+yomi": (hstack([csr_matrix(x_p), x_yomi]).tocsr(), "sparse"),
             "pkg+ocr32b": (hstack([csr_matrix(x_p), x_ocr]).tocsr(), "sparse"),
             "pkg+emb+yomi": (hstack([csr_matrix(x_p), e01, x_yomi]).tocsr(), "sparse"),
             "pkg+emb+ocr32b": (hstack([csr_matrix(x_p), e01, x_ocr]).tocsr(), "sparse"),
             "emb+yomi": (hstack([e01, x_yomi]).tocsr(), "sparse")}
    for p, x_f in x_forms.items():
        feats[f"pkg+{p}"] = (np.hstack([x_p, x_f]), "dense")
        feats[f"pkg+emb+yomi+{p}"] = (hstack([csr_matrix(x_p), csr_matrix(x_f), e01, x_yomi]).tocsr(), "sparse")
    for tname, tcol in TRUTHS.items():
        y = d[tcol].to_numpy()
        for method, (rm, meta) in rule_methods.items():
            if not done(method, tname, 0):
                save(method, tname, 0, ids, rm["prob"], rm["pred"], {"kind": "rule", **meta})
        for (method, (x, kind)), seed in tqdm(list(itertools.product(feats.items(), SEEDS)), desc=f"stack {tname}", leave=False):
            if done(method, tname, seed):
                continue
            p, cs = nested_oof(x, y, C_GRID[kind], seed)
            save(method, tname, seed, ids, p, p >= 0.5, {"kind": "stack", "C_grid": C_GRID[kind], "C_chosen": cs, "n_features": int(x.shape[1])})
    print("done")


if __name__ == "__main__":
    main()
