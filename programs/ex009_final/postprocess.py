"""生徒の確率に後処理（カルタの取り消し，お題文の確認の質問）を適用する（compare.py と annotate_all.py で共用）．

- veto：カルタと予測し，かなで始まるボケが MIN_KANA 件以上あり，最も多い書き出しのかなの割合が VETO_RATE 未満なら，
  カルタ以外で確率の最も高い形式にする（カルタのお題ではボケの書き出しが札のかなに集中するため）
- verify：お題文と予測した画像に Qwen の確認の質問（O_v10_en）を行い，偽ならお題文以外で確率の最も高い形式にする
- verify_noveto：verify からカルタの取り消しを除いたもの（取り消しの効果を確かめるため）
- verify11_*：確認の質問に O_v11_en（典型的なお題文の型を明示した版）を用いるもの
- verify_t{t}：さらに，通常と予測した画像のうちお題文の確率が t 以上のものも確認の質問の対象とし，真ならお題文にする
"""
import numpy as np

FORMATS = ["karuta", "anaume", "odaibun", "normal"]
POSTS = ["raw", "veto", "verify", "verify_noveto", "verify_t0.3", "verify_t0.2", "verify_t0.1", "verify11_noveto",
         "verify11_t0.2_noveto"]
# 後処理の名前 → 確認の質問の部品（verify11 は典型的なお題文の型を明示した O_v11_en）
VERIFY_COMPONENT = {True: "O_v11_en", False: "O_v10_en"}
MIN_KANA, VETO_RATE = 5, 0.3


def kana_max_rate(kana_counts: list, n_kana: list) -> np.ndarray:
    """かなで始まるボケのうち，最も多い書き出しのかなの割合を求める．

    Args:
        kana_counts (list[str]): "文字:数" を ";" で連結した文字列のリスト．
        n_kana (list[int]): かなで始まるボケの数のリスト．

    Returns:
        np.ndarray: 形状 (N,) の割合（ボケがない場合は NaN）．
    """
    out = []
    for counts, n in zip(kana_counts, n_kana):
        vals = [int(kv.split(":")[1]) for kv in (counts or "").split(";") if kv]
        out.append(max(vals) / n if n and vals else np.nan)
    return np.array(out, dtype=float)


def verify_targets(p_format: np.ndarray, fmt: np.ndarray, post: str) -> np.ndarray:
    """確認の質問の対象とする画像を求める．

    Args:
        p_format (np.ndarray): 形状 (N, 4) の形式の確率．
        fmt (np.ndarray): 形状 (N,) の形式の番号（カルタの取り消しの後）．
        post (str): 後処理の名前．

    Returns:
        np.ndarray: 形状 (N,) の bool の配列．
    """
    if not post.startswith("verify"):
        return np.zeros(len(fmt), bool)

    o_idx, n_idx = FORMATS.index("odaibun"), FORMATS.index("normal")
    t = float(post.split("_t")[1].split("_")[0]) if "_t0" in post else None
    extra = (fmt == n_idx) & (p_format[:, o_idx] >= t) if t is not None else np.zeros(len(fmt), bool)
    return (fmt == o_idx) | extra


def decide_format(p_format: np.ndarray, n_kana: np.ndarray, max_rate: np.ndarray, verify: np.ndarray,
                  post: str) -> tuple:
    """後処理を適用して形式を決める．

    Args:
        p_format (np.ndarray): 形状 (N, 4) の形式の確率．
        n_kana (np.ndarray): 形状 (N,) のかなで始まるボケの数．
        max_rate (np.ndarray): 形状 (N,) の最も多い書き出しのかなの割合．
        verify (np.ndarray): 形状 (N,) の確認の質問の回答（対象外の画像は任意の値）．
        post (str): 後処理の名前（POSTS のいずれか）．

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 形式の番号 (N,)，カルタを取り消したか (N,)，確認の質問の対象か (N,)．
    """
    k_idx, o_idx = FORMATS.index("karuta"), FORMATS.index("odaibun")
    fmt = p_format.argmax(1)
    vetoed = np.zeros(len(fmt), bool)
    if post != "raw" and not post.endswith("noveto"):
        vetoed = (fmt == k_idx) & (n_kana >= MIN_KANA) & (max_rate < VETO_RATE)
        alt = p_format.copy()
        alt[:, k_idx] = -1
        fmt = np.where(vetoed, alt.argmax(1), fmt)
    asked = verify_targets(p_format, fmt, post)
    if asked.any():
        cand = fmt == o_idx
        alt = p_format.copy()
        alt[:, o_idx] = -1
        new = np.where(cand & asked & ~verify, alt.argmax(1), fmt)
        fmt = np.where(~cand & asked & verify, o_idx, new)
    return fmt, vetoed, asked
