"""ex005（定義書 v2，3軸，少ない区分の追加と重み付き損失）で用いるデータを扱うモジュール．

- 3軸（形式，属性，媒体）の項目の定義と，評価用ラベルの読み込み（ex004 の目視ラベルから事前知識を除いたもの）
- 評価用データの調整用（dev）と最終評価用（test）への分割（形式で層化，各320枚）
- 教師ラベル（基本の4万枚と，少ない区分の追加分）の読み込みと，学習用・検証用の分割
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, ".ai", "ai-dev-kit", "src"))
_spec = importlib.util.spec_from_file_location("ex004_data", os.path.join(PROJECT_DIR, "programs", "ex004_four_axis_distill",
                                                                       "data.py"))
ex004_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex004_data)
sys.modules["ex004_data"] = ex004_data

DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex005_rare_weighted_distill")
image_path, load_pool = ex004_data.image_path, ex004_data.load_pool
FORMATS = ex004_data.FORMATS
ATTRIBUTES = ex004_data.ATTRIBUTES
MEDIA = ex004_data.MEDIA
FORMAT_IMPLIES = ex004_data.FORMAT_IMPLIES
# 少ない区分（教師ラベル用の4万枚での出現率が3%未満）
RARE = {"media": ["game", "cg", "painting"], "format": ["karuta"]}
EVAL_SPLIT_SEED = 0


def load_eval_labels(split: str = "all") -> pl.DataFrame:
    """評価用ラベル（定義書 v2）を読み込む．

    Args:
        split (str): "all"（640枚），"dev"（調整用320枚），"test"（最終評価用320枚）のいずれか．

    Returns:
        pl.DataFrame: 列 fileId, pool, split, format と，属性・媒体の各項目（bool）を持つ表．
    """
    df = ex004_data.load_eval_labels().select(["fileId", "pool", "format"] + ATTRIBUTES + MEDIA)
    dev, _ = train_test_split(df["fileId"].to_numpy(), test_size=0.5, random_state=EVAL_SPLIT_SEED,
                              stratify=df["format"].to_numpy())
    df = df.with_columns(pl.when(pl.col("fileId").is_in(dev.tolist())).then(pl.lit("dev")).otherwise(pl.lit("test"))
                         .alias("split"))
    return df if split == "all" else df.filter(pl.col("split") == split)


def load_teacher_labels(name: str, extra: str = None) -> pl.DataFrame:
    """教師ラベルを読み込み，学習用・検証用の分割を付ける．追加分は学習用と検証用に9:1で分ける．

    Args:
        name (str): datasets/ex005_rare_weighted_distill/teacher_labels/ 内の基本の4万枚の CSV の名前．
        extra (str): 同じディレクトリの少ない区分の追加分の CSV の名前（None で追加しない）．

    Returns:
        pl.DataFrame: 列 fileId, split, source（"base" または "rare"）, format と各項目を持つ表．
    """
    d = os.path.join(DATASET_DIR, "teacher_labels")
    base = load_pool().join(pl.read_csv(os.path.join(d, f"{name}.csv")), on="fileId", how="inner",
                            maintain_order="left").with_columns(pl.lit("base").alias("source"))
    cols = ["fileId", "split", "source", "format"] + ATTRIBUTES + MEDIA
    if extra is None:
        return base.select(cols)
    ex = pl.read_csv(os.path.join(d, f"{extra}.csv"))
    rng = np.random.default_rng(0)
    ex = ex.with_columns(pl.Series("split", np.where(rng.random(len(ex)) < 0.9, "train", "val")),
                         pl.lit("rare").alias("source"))
    return pl.concat([base.select(cols), ex.select(cols)])


def label_arrays(df: pl.DataFrame) -> dict:
    """ラベルの表を学習用の配列に変換する．

    Args:
        df (pl.DataFrame): 列 format と各項目（bool）を持つ表（行数 N）．

    Returns:
        dict: キー format（形状 (N,) の番号），attr（形状 (N, 5)），media（形状 (N, 8)）の配列．
    """
    return {"format": np.array([FORMATS.index(f) for f in df["format"]]),
            "attr": df.select(ATTRIBUTES).to_numpy().astype(np.float32),
            "media": df.select(MEDIA).to_numpy().astype(np.float32)}


class ThreeAxisDataset(torch.utils.data.Dataset):
    """お題画像と3軸のラベルを返すデータセット（画像エンコーダの微調整用）．"""

    def __init__(self, ids: list, y: dict, transform):
        """
        Args:
            ids (list[int]): お題IDのリスト（長さ N）．
            y (dict): label_arrays の辞書．
            transform (callable): PIL 画像を形状 (3, H, W) のテンソルに変換する関数．
        """
        self.ids, self.y, self.transform = ids, y, transform

    def __len__(self) -> int:
        """データ数を返す．

        Returns:
            int: 画像の数．
        """
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple:
        """i 番目の画像とラベルを返す．

        Args:
            i (int): インデックス．

        Returns:
            tuple[torch.Tensor, dict]: 形状 (3, H, W) の画像と，キー format・attr・media のラベル．
        """
        with Image.open(image_path(self.ids[i])) as im:
            x = self.transform(im.convert("RGB"))
        return x, {"format": int(self.y["format"][i]), "attr": torch.tensor(self.y["attr"][i]),
                   "media": torch.tensor(self.y["media"][i])}
