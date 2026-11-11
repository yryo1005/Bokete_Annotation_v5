"""生徒の入力の特徴を作る．

- 品詞の特徴（MeCab：fugashi＋UniDic-lite）：ボケの文の形態素の品詞の割合（固有名詞（人名・地名・組織名・一般），普通名詞，代名詞，
  動詞，形容詞，助詞（終助詞・格助詞），助動詞，感動詞，記号，接尾辞），固有名詞の有無，カタカナ語・英字の割合，
  文末の品詞，括弧・疑問符・感嘆符の有無，文字数．計 POS_DIM 次元．
- 文の埋め込み（ruri-v3-310m）とお題画像の埋め込み（v5.0 の SigLIP2-so400m．読み込みのみ）：ex013 の embed.py の関数を用いる．

対象：eval（検証データ620件），aug（水増し用），swap（制約の順守の検証の組），swap_train（制約の順守の分類器の学習用），
ex013 の train（学習用のボケ3.5万件．埋め込みは ex013 のものを用いる）．

出力：datasets/ex014_boke_labels_v2/feat/{集合}_{pos,ruri310,image}.npy

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/features.py
"""
import os
import re
import sys

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, EX13_DATASET, EX13_DIR, _load, load_eval  # noqa: E402

E13 = _load("ex013_embed_for_v2", os.path.join(EX13_DIR, "embed.py"))
FEAT_DIR = os.path.join(DATASET_DIR, "feat")
POS_KEYS = [("名詞", "固有名詞", "人名"), ("名詞", "固有名詞", "地名"), ("名詞", "固有名詞", "組織名"), ("名詞", "固有名詞", None),
            ("名詞", "普通名詞", None), ("代名詞", None, None), ("動詞", None, None), ("形容詞", None, None), ("助詞", "終助詞", None),
            ("助詞", "格助詞", None), ("助詞", None, None), ("助動詞", None, None), ("感動詞", None, None), ("補助記号", None, None),
            ("接尾辞", None, None)]
END_POS = ["名詞", "動詞", "助詞", "助動詞", "形容詞", "補助記号", "感動詞", "その他"]
POS_DIM = len(POS_KEYS) + 1 + 2 + len(END_POS) + 4 + 1
_TAGGER = None


def tagger():
    """MeCab の形態素解析器を返す（初回のみ作る）．

    Returns:
        fugashi.Tagger: 解析器．
    """
    global _TAGGER
    if _TAGGER is None:
        import fugashi

        _TAGGER = fugashi.Tagger()
    return _TAGGER


def pos_features(text: str) -> np.ndarray:
    """1つの文の品詞の特徴を求める．

    Args:
        text (str): ボケの文．

    Returns:
        np.ndarray: 形状 (POS_DIM,) の特徴．
    """
    t = (text or "").replace("\n", " ")[:200]
    words = [w for w in tagger()(t)]
    n = max(len(words), 1)
    f = []
    for p1, p2, p3 in POS_KEYS:
        c = sum(1 for w in words if w.feature.pos1 == p1 and (p2 is None or w.feature.pos2 == p2) and (p3 is None or w.feature.pos3 == p3))
        f.append(c / n)
    f.append(float(any(w.feature.pos2 == "固有名詞" for w in words)))
    chars = max(len(t), 1)
    f.append(len(re.findall(r"[ァ-ヶー]", t)) / chars)
    f.append(len(re.findall(r"[A-Za-z]", t)) / chars)
    content = [w for w in words if w.feature.pos1 not in ("補助記号", "空白")]
    last = content[-1].feature.pos1 if content else "その他"
    f += [float(last == p) for p in END_POS[:-1]] + [float(last not in END_POS[:-1])]
    f += [float("「" in t or "『" in t), float("（" in t or "(" in t), float(bool(re.search(r"[？?]", t))), float(bool(re.search(r"[！!]", t)))]
    f.append(min(len(t), 100) / 100)
    return np.array(f, dtype=np.float32)


def build(name: str, df: pl.DataFrame, need_emb: bool = True) -> None:
    """集合の特徴を作って保存する（既にあれば省く）．

    Args:
        name (str): 集合の名前．
        df (pl.DataFrame): 列 odaiId, text を持つ表．
        need_emb (bool): 文と画像の埋め込みも作るか．

    Returns:
        None
    """
    p = os.path.join(FEAT_DIR, f"{name}_pos.npy")
    if not os.path.exists(p):
        np.save(p, np.stack([pos_features(t) for t in tqdm(df["text"].fill_null("").to_list(), desc=f"pos:{name}", leave=False)]))
    if not need_emb:
        return
    p = os.path.join(FEAT_DIR, f"{name}_ruri310.npy")
    if not os.path.exists(p):
        np.save(p, E13.embed_texts("ruri310", df["text"].fill_null("").to_list())[0])
    p = os.path.join(FEAT_DIR, f"{name}_image.npy")
    if not os.path.exists(p):
        np.save(p, E13.image_embeddings(df["odaiId"].to_list()))


def main() -> None:
    """全ての集合の特徴を作る．

    Returns:
        None
    """
    os.makedirs(FEAT_DIR, exist_ok=True)
    build("eval", load_eval().select("odaiId", "text"))
    build("aug", pl.read_csv(os.path.join(DATASET_DIR, "aug_pool.csv")))
    build("swap", pl.read_csv(os.path.join(DATASET_DIR, "swap_pairs.csv")))
    build("swap_train", pl.read_csv(os.path.join(DATASET_DIR, "swap_train.csv")))
    build("train", pl.read_parquet(os.path.join(EX13_DATASET, "train_pool.parquet")), need_emb=False)
    print("done", POS_DIM)


if __name__ == "__main__":
    main()
