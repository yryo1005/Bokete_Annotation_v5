"""ex027 の後段（v5.3.0 の確率との組み合わせ）を1,620枚の全てで学習し，新しい評価用データ（ex028）の確率を求める．

手法（ex027 と同じ特徴）：
- pkg+emb：v5.3.0 の確率 P_COLS のロジット（標準化）と，SigLIP2 の埋め込み（標準化して0.1倍）
- pkg+ocr32b：v5.3.0 の確率と，Qwen3-VL-32B の書き起こし（重ねた文字・余白の文字）の文字 1〜3-gram の TF-IDF

正解は v6 の目視の odai_R1b（ex026 の1,620枚）である．標準化と TF-IDF の語彙は1,620枚で求め，新しい画像に当てはめる．
正則化の係数 C は1,620枚の分割（乱数の種 SEEDS）の F1 で ex027 と同じ格子から選び，1,620枚の全てで学習し直す．
閾値は ex027 と同じ0.5とする．

出力：outputs/ex028_odai_holdout/preds/{手法}/v6/{seed}/pred.csv（列 rep_odaiId, prob, pred．636枚）と meta.json（選んだ C，学習の1,620枚の F1，
      係数を選んだ分割の F1），pred_df.csv（追加の抽出の層 D・F の600枚．order_049）

order_049：追加の抽出（targets_df.csv）の予測 pred_df.csv を加えた．同じ係数と学習データで学習し直すため，pred.csv の値は変わらない．

実行例:
    .venv_annotate/bin/python programs/ex028_odai_holdout/train.py
"""
import itertools
import json
import os
import sys

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX027, OUTPUT_DIR, P_COLS, TARGETS, TARGETS_DF, load_embeddings, load_ex027_module, load_ocr  # noqa: E402

T27 = load_ex027_module("train")
SEEDS = T27.SEEDS
C_GRID = T27.C_GRID
METHODS = {"pkg+emb": "dense", "pkg+ocr32b": "sparse"}
PRED_DIR = os.path.join(OUTPUT_DIR, "preds")
TRUTH = "v6"


def tfidf_pair(train: list, test: list) -> tuple:
    """文字の1〜3-gram の TF-IDF と文字があるかの列を，学習の書き起こしで語彙を決めて作る（ex027 の tfidf と同じ設定）．

    Args:
        train (list[str]): 学習の書き起こし．
        test (list[str]): 予測する画像の書き起こし．

    Returns:
        tuple[csr_matrix, csr_matrix]: 形状 (N_tr, D) と (N_te, D) の特徴．
    """
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True).fit(train)
    has = lambda t: csr_matrix(np.array([[len(s) > 0] for s in t], dtype=float))  # noqa: E731
    return hstack([vec.transform(train), has(train)]).tocsr(), hstack([vec.transform(test), has(test)]).tocsr()


def build_features() -> tuple:
    """1,620枚（学習）と新しい画像（予測）の特徴を作る．

    Returns:
        tuple[dict, np.ndarray, list]: 手法ごとの (学習の特徴, 予測の特徴)，形状 (1620,) の正解，予測する画像のお題ID
        （targets.csv の636枚と targets_df.csv の600枚の順）．
    """
    d27, emb27 = EX027.load_eval()
    ids = pl.read_csv(TARGETS)["rep_odaiId"].to_list() + pl.read_csv(TARGETS_DF)["rep_odaiId"].to_list()
    pk = EX027.load_package_probs(ids)
    assert pk["p_has_odai_text"].null_count() == 0, "v5.3.0 の確率が欠けている"
    sp = StandardScaler().fit(T27.logit(d27.select(P_COLS).to_numpy()))
    xp_tr, xp_te = sp.transform(T27.logit(d27.select(P_COLS).to_numpy())), sp.transform(T27.logit(pk.select(P_COLS).to_numpy()))
    se = StandardScaler().fit(emb27)
    xe_tr, xe_te = se.transform(emb27), se.transform(load_embeddings(ids))
    ocr = load_ocr(ids)
    o_tr, o_te = tfidf_pair(d27["ocr_o"].fill_null("").to_list(), ocr["ocr_o"].to_list())
    m_tr, m_te = tfidf_pair(d27["ocr_m"].fill_null("").to_list(), ocr["ocr_m"].to_list())
    feats = {"pkg+emb": (np.hstack([xp_tr, xe_tr * 0.1]), np.hstack([xp_te, xe_te * 0.1])),
             "pkg+ocr32b": (hstack([csr_matrix(xp_tr), o_tr, m_tr]).tocsr(), hstack([csr_matrix(xp_te), o_te, m_te]).tocsr())}
    return feats, d27["odai_R1b"].to_numpy(), ids


def cv_f1(x, y: np.ndarray, c: float, seed: int) -> float:
    """1,620枚の5分割の交差検証の F1（閾値0.5）を求める（選んだ係数の検証の値として記録する）．

    Args:
        x (np.ndarray | csr_matrix): 形状 (N, D) の特徴．
        y (np.ndarray): 形状 (N,) の正解．
        c (float): 正則化の強さの逆数．
        seed (int): 分割の乱数の種．

    Returns:
        float: F1．
    """
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(T27.N_SPLITS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        p[te] = T27.fit_predict(x[tr], y[tr], x[te], c)
    return T27.f1(y, p >= 0.5)


def main() -> None:
    """全ての手法と乱数の種で学習し，新しい画像の確率を保存する（保存済みは飛ばす）．

    Returns:
        None
    """
    feats, y, ids = build_features()
    n_main = pl.read_csv(TARGETS).height
    for method, seed in tqdm(list(itertools.product(METHODS, SEEDS)), desc="train", leave=False):
        out = os.path.join(PRED_DIR, method, TRUTH, str(seed))
        if os.path.exists(os.path.join(out, "pred.csv")) and os.path.exists(os.path.join(out, "pred_df.csv")):
            continue
        x_tr, x_te = feats[method]
        c = T27.choose_c(x_tr, y, C_GRID[METHODS[method]], seed)
        p_tr = T27.fit_predict(x_tr, y, x_tr, c)
        p_te = T27.fit_predict(x_tr, y, x_te, c)
        os.makedirs(out, exist_ok=True)
        pred = pl.DataFrame({"rep_odaiId": ids, "prob": p_te, "pred": p_te >= 0.5})
        main_path = os.path.join(out, "pred.csv")
        if os.path.exists(main_path):
            old = pl.read_csv(main_path)["prob"].to_numpy()
            assert np.allclose(old, p_te[:n_main], atol=1e-6), "既存の pred.csv と予測が一致しない"
        pred.head(n_main).write_csv(main_path)
        pred.slice(n_main).write_csv(os.path.join(out, "pred_df.csv"))
        meta = {"method": method, "truth": TRUTH, "seed": seed, "C_grid": C_GRID[METHODS[method]], "C": c, "n_train": int(len(y)),
                "n_features": int(x_tr.shape[1]), "train_f1_in_sample": T27.f1(y, p_tr >= 0.5), "train_cv_f1": cv_f1(x_tr, y, c, seed)}
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print("done")


if __name__ == "__main__":
    main()
