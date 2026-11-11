"""ex014（ボケのラベル 第2版）で用いるデータを扱うモジュール．

- 検証データ：ex013 の420件（eval_labels.txt）と，少ない発話の型を多く含むよう追加した200件（eval2_labels.txt）の計620件．
  性的・不適切の項目は除く．お題の単位で，元の420件は ex013 と同じ分割，追加の200件は層ごとに半数ずつ調整用と最終評価用に分ける．
- 発話の型と知識の統合した区分（MERGES）も定義する．
"""
import importlib.util
import os

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX13_DIR = os.path.join(PROJECT_DIR, "programs", "ex013_boke_labels")
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex014_boke_labels_v2")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex014_boke_labels_v2")


def _load(name: str, path: str):
    """ファイルからモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．
        path (str): ファイルのパス．

    Returns:
        module: 読み込んだモジュール．
    """
    import sys

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D13 = _load("ex013_data_for_v2", os.path.join(EX13_DIR, "data.py"))
VOICES, KNOWS, image_path, load_odai_labels, load_bokes = D13.VOICES, D13.KNOWS, D13.image_path, D13.load_odai_labels, D13.load_bokes
EX13_DATASET, EX13_OUTPUT, ANN_DIR = D13.DATASET_DIR, D13.OUTPUT_DIR, D13.ANN_DIR
# 統合した区分：名前 → (項目, 元の区分 → 統合後の区分)
MERGES = {
    "voice6": ("voice", {v: v for v in VOICES}),
    "voice5_SM": ("voice", {"S": "C", "M": "C", "T": "T", "N": "N", "L": "L", "A": "A"}),
    "voice3": ("voice", {"S": "C", "M": "C", "T": "X", "N": "X", "L": "X", "A": "A"}),
    "know4": ("know", {k: k for k in KNOWS}),
    "know3_0G": ("know", {"0": "B", "G": "B", "I": "I", "O": "O"}),
    "know2_O": ("know", {"0": "-", "G": "-", "I": "-", "O": "O"}),
    "know2_IO": ("know", {"0": "-", "G": "-", "I": "K", "O": "K"}),
}


def parse_labels(path: str, inappropriate_col: bool) -> pl.DataFrame:
    """目視のラベルのファイルを読み込む．

    Args:
        path (str): ファイルのパス．
        inappropriate_col (bool): 性的・不適切の列を含む書式か（ex013）．

    Returns:
        pl.DataFrame: 列 no, comp, voice, know, uses_text, mislabeled を持つ表．
    """
    rows = [ln.split() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    return pl.DataFrame({"no": [int(r[0]) for r in rows], "comp": [r[1] for r in rows], "voice": [r[2] for r in rows],
                         "know": [r[3] for r in rows], "uses_text": [r[4] == "1" for r in rows],
                         "mislabeled": [(r[6] == "m") if inappropriate_col else False for r in rows]})


def load_eval() -> pl.DataFrame:
    """検証データ（620件）を読み込む．

    Returns:
        pl.DataFrame: 列 no, stratum, odaiId, text, comp, voice, know, uses_text, mislabeled, split, part（old / rare）
        とお題のラベルを持つ表．
    """
    old = D13.load_eval().select("no", "stratum", "odaiId", "text", "split")
    old = old.join(parse_labels(os.path.join(EX13_DIR, "eval_labels.txt"), True), on="no").with_columns(pl.lit("old").alias("part"))
    new = pl.read_csv(os.path.join(DATASET_DIR, "eval2_bokes.csv")).select("no", "stratum", "odaiId", "text")
    rng = np.random.default_rng(0)
    split = {}
    for st in sorted(new["stratum"].unique().to_list()):
        ids = rng.permutation(sorted(new.filter(pl.col("stratum") == st)["odaiId"].unique().to_list()))
        for k, fid in enumerate(ids):
            split[int(fid)] = "dev" if k < len(ids) // 2 else "test"
    new = new.with_columns(pl.Series("split", [split[i] for i in new["odaiId"].to_list()]))
    new = new.join(parse_labels(os.path.join(THIS_DIR, "eval2_labels.txt"), False), on="no").with_columns(pl.lit("rare").alias("part"))
    d = pl.concat([old, new.select(old.columns)])
    return d.join(load_odai_labels(d["odaiId"].unique().to_list()), on="odaiId", how="left").sort("no")


def merge(col: str, values, scheme: str) -> list:
    """区分を統合する．

    Args:
        col (str): 項目（voice / know）．
        values (list[str]): 元の区分．
        scheme (str): MERGES のキー．

    Returns:
        list[str]: 統合後の区分．
    """
    item, mp = MERGES[scheme]
    assert item == col
    return [mp[str(v)] for v in values]
