"""ex027（お題文ありの判定の改善：v5.3.0 の確率との組み合わせと，文の形だけを問う教師）で用いるデータを扱うモジュール．

- 評価：ex026 の目視のラベル付きの1,620枚（定義書 v6 の属性と，規則 R1b で導いたお題文あり）．
- 全データの推定：ex025 の600枚の事後層別の重み．
- 教師ラベルの画像（プール）：目視のラベルのない代表の画像から無作為に選んだ画像．パッケージ v5.3.0 が文字ありとした画像に限る．
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex027_odai_text_improve")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex027_odai_text_improve")
POOL = os.path.join(DATASET_DIR, "pool.csv")
EX026_DIR = os.path.join(PROJECT_DIR, "programs", "ex026_text_attr_v6")
SEED = 0
# パッケージ v5.3.0 の確率の列（組み合わせの特徴に用いる）
P_COLS = ["p_has_odai_text", "p_has_text", "p_has_blank", "p_has_karuta_mark", "p_format_karuta", "p_format_anaume", "p_format_odaibun",
          "p_format_normal", "p_photo", "p_anime", "p_game", "p_manga", "p_text_graphic", "p_painting", "p_illustration"]


def _load(name: str, path: str):
    """ファイルからモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．
        path (str): ファイルのパス．

    Returns:
        module: 読み込んだモジュール．
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_ex026():
    """ex026 の data と evaluate のモジュールを読み込む（evaluate は `from data import` を用いるため sys.modules を一時的に差し替える）．

    Returns:
        tuple[module, module]: ex026 の data と evaluate．
    """
    d26 = _load("ex026_data_for_ex027", os.path.join(EX026_DIR, "data.py"))
    if "ex026_evaluate_for_ex027" in sys.modules:
        return d26, sys.modules["ex026_evaluate_for_ex027"]
    saved = sys.modules.get("data")
    sys.modules["data"] = d26
    try:
        ev = _load("ex026_evaluate_for_ex027", os.path.join(EX026_DIR, "evaluate.py"))
    finally:
        if saved is None:
            sys.modules.pop("data", None)
        else:
            sys.modules["data"] = saved
    return d26, ev


EX026, EX026_EVAL = _load_ex026()
FORMS = EX026.FORMS
image_path = EX026.image_path
add_derived = EX026.add_derived
ANNOTATION = EX026.EX024.ANNOTATION


def load_package_probs(ids: list) -> pl.DataFrame:
    """パッケージ v5.3.0 の確率と予測をお題IDで引く．

    Args:
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 odaiId, has_odai_text と P_COLS を持つ表（ids の順）．
    """
    a = pl.scan_parquet(ANNOTATION).filter(pl.col("odaiId").is_in(pl.Series(ids).implode())).select(
        "odaiId", "has_odai_text", *P_COLS).collect()
    return pl.DataFrame({"odaiId": ids}).join(a, on="odaiId", how="left")


def load_eval() -> tuple:
    """評価の1,620枚の表（目視の属性，v5 の目視，Qwen と YomiToku の書き起こし，v5.3.0 の確率）と埋め込みを読み込む．

    Returns:
        tuple[pl.DataFrame, np.ndarray]: 表と，形状 (1620, 1152) の埋め込み．
    """
    d = EX026.load_all().join(EX026_EVAL.load_qwen(), on="rep_odaiId", how="left")
    ids = d["rep_odaiId"].to_list()
    pk = load_package_probs(ids).drop("odaiId")
    d = d.with_columns(*[pk[c] for c in pk.columns])
    assert d["p_has_odai_text"].null_count() == 0, "v5.3.0 の確率が欠けている"
    return d, EX026_EVAL.load_emb(ids)


def build_pool(n: int, exclude: list, seed: int = SEED) -> pl.DataFrame:
    """教師ラベルを付ける画像を，v5.3.0 が文字ありとした代表の画像から無作為に選ぶ（初回は保存し，以降は読み込む）．

    Args:
        n (int): 選ぶ画像の数（保存したプールがある場合は用いない）．
        exclude (list[int]): 除く代表の画像のお題ID（評価の画像）．
        seed (int): 乱数の種．

    Returns:
        pl.DataFrame: 列 rep_odaiId と P_COLS を持つ表．
    """
    # 画像のファイルがない画像を除くため，保存したプールは n より少ない場合がある
    if os.path.exists(POOL):
        return pl.read_csv(POOL)
    os.makedirs(DATASET_DIR, exist_ok=True)
    a = (pl.scan_parquet(ANNOTATION).filter((pl.col("odaiId") == pl.col("rep_odaiId")) & pl.col("has_text"))
         .filter(~pl.col("rep_odaiId").is_in(pl.Series(exclude).implode())).select("rep_odaiId", "has_odai_text", *P_COLS).collect())
    p = a.sample(n, seed=seed).sort("rep_odaiId")
    p = p.filter(pl.Series([os.path.exists(image_path(i)) for i in p["rep_odaiId"].to_list()]))
    p.write_csv(POOL)
    return p


def load_pool_embeddings(ids: list) -> np.ndarray:
    """プールの画像の埋め込みを読み込む（初回は保存する）．

    Args:
        ids (list[int]): 代表の画像のお題IDのリスト．

    Returns:
        np.ndarray: 形状 (len(ids), 1152) の配列．
    """
    cache = os.path.join(OUTPUT_DIR, "emb_pool.npy")
    if os.path.exists(cache):
        e = np.load(cache)
        if len(e) == len(ids):
            return e
    e = EX026.load_embeddings(ids)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(cache, e)
    return e
