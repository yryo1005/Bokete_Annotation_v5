"""ex028（独立な評価用データによる，お題文ありの後段の検証）で用いるデータを扱うモジュール．

- 新しい評価用データ：ex026 の1,620枚，ex024 の教師ラベルの画像，ex025 の抽出の画像，ex027 のプールに含まれない代表の画像から，
  ex025 の評価の層（A〜D，F と，通常と予測した画像の p の区間 N0〜N3）ごとに無作為に抽出した600枚．
  全データの値は層の重み（層の枚数 / 抽出数）で推定する．
- 判断のずれの確認：ex026 の1,620枚から無作為に選んだ REPEAT 枚を混ぜる．
- 両者を混ぜて並べ替え，番号だけを付けた画像の一覧（sheets/）で目視の v6 の属性を付ける（holdout_labels.txt）．
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex028_odai_holdout")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex028_odai_holdout")
TARGETS = os.path.join(DATASET_DIR, "targets.csv")
LABELS = os.path.join(THIS_DIR, "holdout_labels.txt")
EX027_DIR = os.path.join(PROJECT_DIR, "programs", "ex027_odai_text_improve")
# 層ごとの抽出数（ex025 の600枚の評価の層の枚数と同じ）
STRATA = {"A_odaibun_yes": 100, "B_odaibun_rejected": 40, "C_blank_p_hi": 110, "D_blank_p_lo": 50, "F_normal_p_lo": 60,
          "N0": 28, "N1": 77, "N2": 67, "N3": 68}
REPEAT = 36
SEED = 28
# 追加の抽出（order_049）：層 D・F をそれぞれ DF_N 枚
TARGETS_DF = os.path.join(DATASET_DIR, "targets_df.csv")
LABELS_DF = os.path.join(THIS_DIR, "df_labels.txt")
DF_STRATA = ["D_blank_p_lo", "F_normal_p_lo"]
DF_N = 300
SEED_DF = 49


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


EX027 = _load("ex027_data_for_ex028", os.path.join(EX027_DIR, "data.py"))
EX026 = EX027.EX026
NM = EX026._load_normal_miss()
P_COLS = EX027.P_COLS
image_path = EX027.image_path
load_package_probs = EX027.load_package_probs


def load_population() -> pl.DataFrame:
    """全ての代表の画像に評価の層（ex025 の stratum2 と同じ定義）を付けて読み込む．

    Returns:
        pl.DataFrame: 列 rep_odaiId, format_argmax, p, stratum2 を持つ表（約661万行）．
    """
    pop = NM.load_population()
    b = NM.bin_of(pop["p"].to_numpy())
    normal = (pop["format_argmax"] == "normal").to_numpy()
    st2 = np.where(normal, np.char.add("N", b.astype(str)), pop["stratum"].to_numpy())
    # 通常と予測し p が0.1未満の画像（区間 −1）は層 F である
    st2 = np.where(st2 == "N-1", "F_normal_p_lo", st2)
    return pop.select("rep_odaiId", "format_argmax", "p", pl.Series("stratum2", st2))


def excluded_reps() -> set:
    """抽出から除く代表の画像を求める．

    Returns:
        set[int]: ex024 の正解と教師ラベル，ex025 の抽出，ex026 の1,620枚，ex027 のプールの代表の画像のお題ID．
    """
    ids = NM.excluded_reps()
    for path in [NM.TARGETS, NM.TARGETS_NORMAL, EX026.TARGETS, EX027.POOL]:
        ids |= set(pl.read_csv(path)["rep_odaiId"].to_list())
    return ids


def load_labels() -> pl.DataFrame:
    """目視の v6 の属性を読み込む（書式は定義書 v6 の付録．番号は targets.csv の no）．

    Returns:
        pl.DataFrame: 列 no, code, memo と L_{層}, F_{層}_{形} を持つ表．
    """
    return EX026.load_labels(LABELS)


def load_all() -> pl.DataFrame:
    """抽出の表に目視の属性，規則で導いたお題文あり，v5.3.0 の確率と予測を付けて読み込む．

    Returns:
        pl.DataFrame: targets.csv の列（no, rep_odaiId, kind, stratum2, N_h, n_h），w（新しい評価用データの重み），
        load_labels の列，odai_R1b など add_derived の列，has_odai_text（v5.3.0 の予測），P_COLS を持つ表．
    """
    t = pl.read_csv(TARGETS).join(load_labels(), on="no", how="left")
    assert t["code"].null_count() == 0, "目視のラベルが欠けている"
    t = EX026.add_derived(t).with_columns((pl.col("N_h") / pl.col("n_h")).alias("w"))
    pk = load_package_probs(t["rep_odaiId"].to_list()).drop("odaiId")
    return t.with_columns(*[pk[c] for c in pk.columns])


def load_ex027_module(name: str):
    """ex027 の train / evaluate モジュールを読み込む（`from data import` のため sys.modules["data"] を一時的に ex027 の data に差し替える）．

    Args:
        name (str): モジュールのファイル名（拡張子なし．train / evaluate）．

    Returns:
        module: 読み込んだモジュール．
    """
    key = f"ex027_{name}_for_ex028"
    if key in sys.modules:
        return sys.modules[key]
    t27 = load_ex027_module("train") if name == "evaluate" else None
    saved = {k: sys.modules.get(k) for k in ("data", "train")}
    sys.modules["data"] = EX027
    if t27 is not None:
        sys.modules["train"] = t27
    sys.path.insert(0, EX027_DIR)
    try:
        return _load(key, os.path.join(EX027_DIR, f"{name}.py"))
    finally:
        sys.path.remove(EX027_DIR)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def load_embeddings(ids: list) -> np.ndarray:
    """抽出の画像の SigLIP2 の埋め込みを読み込む（初回は保存する）．

    Args:
        ids (list[int]): 代表の画像のお題IDのリスト（targets.csv の順）．

    Returns:
        np.ndarray: 形状 (len(ids), 1152) の配列．
    """
    cache = os.path.join(OUTPUT_DIR, "emb_holdout.npy")
    if os.path.exists(cache):
        e = np.load(cache)
        if len(e) == len(ids):
            return e
    e = EX026.load_embeddings(ids)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(cache, e)
    return e


def load_ocr(ids: list) -> pl.DataFrame:
    """Qwen3-VL-32B の書き起こし（qwen_ocr.py の出力．OCR.csv と OCR_df.csv）を読み込む．

    Args:
        ids (list[int]): 代表の画像のお題IDのリスト．

    Returns:
        pl.DataFrame: 列 rep_odaiId, ocr_o, ocr_m を持つ表（ids の順）．
    """
    paths = [os.path.join(OUTPUT_DIR, "qwen", f) for f in ("OCR.csv", "OCR_df.csv")]
    o = pl.concat([pl.read_csv(p, schema_overrides={"text_o": pl.Utf8, "text_m": pl.Utf8, "raw": pl.Utf8}) for p in paths if os.path.exists(p)])
    o = o.select("rep_odaiId", pl.col("text_o").fill_null("").alias("ocr_o"), pl.col("text_m").fill_null("").alias("ocr_m"))
    return pl.DataFrame({"rep_odaiId": ids}).join(o, on="rep_odaiId", how="left").with_columns(pl.col("ocr_o", "ocr_m").fill_null(""))


def load_df() -> pl.DataFrame:
    """追加の抽出（層 D・F）に目視の属性，規則で導いたお題文あり，v5.3.0 の確率と予測を付けて読み込む．

    Returns:
        pl.DataFrame: targets_df.csv の列（no, rep_odaiId, stratum2, N_h, n_h）と，load_all と同じ列を持つ表．
    """
    t = pl.read_csv(TARGETS_DF).join(EX026.load_labels(LABELS_DF), on="no", how="left")
    assert t["code"].null_count() == 0, "目視のラベルが欠けている"
    t = EX026.add_derived(t).with_columns((pl.col("N_h") / pl.col("n_h")).alias("w"))
    pk = load_package_probs(t["rep_odaiId"].to_list()).drop("odaiId")
    return t.with_columns(*[pk[c] for c in pk.columns])
