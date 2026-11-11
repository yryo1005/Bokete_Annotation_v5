"""ex017（画像外の知識の判定の改善と，推論の高速化）で用いるデータを扱うモジュール．

ex016 の data.py を別名で読み込み，検証データ（合意ラベル）・教師のラベルの学習データ・学習用の目視のラベルをそのまま使う．
"""
import importlib.util
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX16_DIR = os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm")
EX16_OUTPUT = os.path.join(PROJECT_DIR, "outputs", "ex016_boke_small_vlm")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex017_boke_outside")
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex017_boke_outside")


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


D16 = _load("ex016_data_for_v5", os.path.join(EX16_DIR, "data.py"))
VOICES, MEDIA, image_path, load_odai_labels = D16.VOICES, D16.MEDIA, D16.image_path, D16.load_odai_labels
load_eval, load_human, load_train = D16.load_eval, D16.load_human, D16.load_train


def load_labeled() -> pl.DataFrame:
    """目視のラベルがある全ての項目（学習用931件と検証データ620件）を，教師の指示に必要な列とともに返す．

    Returns:
        pl.DataFrame: 列 key（h{hid} / e{no}），set（human / dev / test），odaiId, text, format, has_text, caption, 媒体, voice, outside．
    """
    cols = ["odaiId", "format", "has_text", "caption"] + MEDIA
    hu = load_human()
    hu = hu.drop("format").join(load_odai_labels(hu["odaiId"].unique().to_list()).select(cols), on="odaiId", how="left", maintain_order="left")
    hu = hu.select(pl.format("h{}", "hid").alias("key"), pl.lit("human").alias("set"), "text", "voice", "outside", *cols)
    ev = load_eval().select(pl.format("e{}", "no").alias("key"), pl.col("split").alias("set"), "text", "voice", "outside", *cols)
    return pl.concat([hu, ev.select(hu.columns)])
