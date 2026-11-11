"""ex015（ボケのラベル 第3版）で用いるデータを扱うモジュール．

- 検証データ：ex014 の620件（同じ調整用・最終評価用の分割）に，定義書 第3版の画像外の知識（outside_labels.txt）を加える．
- 発話の型は6区分（S, M, T, N, L, A）と3区分（C：登場人物の言葉，X：外からの言葉，A：回答）の両方を扱う．
- ex014 のモジュールは名前が衝突するため別名で読み込む．
"""
import importlib.util
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX13_DIR = os.path.join(PROJECT_DIR, "programs", "ex013_boke_labels")
EX13_DATASET = os.path.join(PROJECT_DIR, "datasets", "ex013_boke_labels")
EX14_DIR = os.path.join(PROJECT_DIR, "programs", "ex014_boke_labels_v2")
EX14_DATASET = os.path.join(PROJECT_DIR, "datasets", "ex014_boke_labels_v2")
EX14_OUTPUT = os.path.join(PROJECT_DIR, "outputs", "ex014_boke_labels_v2")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex015_boke_labels_v3")
VOICE3 = {"S": "C", "M": "C", "T": "X", "N": "X", "L": "X", "A": "A"}


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


D14 = _load("ex014_data_for_v3", os.path.join(EX14_DIR, "data.py"))
ANN_DIR, load_odai_labels, load_bokes = D14.ANN_DIR, D14.load_odai_labels, D14.load_bokes


def load_outside_labels() -> pl.DataFrame:
    """画像外の知識の目視のラベルを読み込む．

    Returns:
        pl.DataFrame: 列 no, outside（bool）を持つ表．
    """
    rows = [ln.split() for ln in open(os.path.join(THIS_DIR, "outside_labels.txt"), encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    return pl.DataFrame({"no": [int(r[0]) for r in rows], "outside": [r[1] == "1" for r in rows]})


def load_eval() -> pl.DataFrame:
    """検証データ（620件）を読み込む．

    Returns:
        pl.DataFrame: ex014 の検証データの列に，outside（画像外の知識）と voice3（3区分）を加えた表．
    """
    e = D14.load_eval().join(load_outside_labels(), on="no", how="left")
    assert e["outside"].null_count() == 0
    return e.with_columns(pl.col("voice").replace_strict(VOICE3).alias("voice3"))
