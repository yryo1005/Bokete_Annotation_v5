"""ex007（定義書 v3）で用いるデータを扱うモジュール．

定義書 v3（programs/label_guideline/label_definition_v3.md）のラベルは，形式4クラス，属性3項目（空欄あり，カルタ記号あり，
文字あり），媒体7クラス（CG なし）である．ex005 と同じ関数名で定義し，ex005 の students.py，common.py，train.py を
そのまま用いられるようにする．

- 評価用ラベル：ex005 の目視ラベルを変換する．文字あり＝お題文あり OR その他の文字あり，合成画像は削除する．
  目視で CG とした画像は cg_relabel.txt の媒体に振り分け直す．
- 教師ラベル：教師 T_v9（形式 F_v2，属性 J_v6 の回答を変換，媒体 M_v9（CG なし），絵画は P_v8_en）の出力から作る．
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl
import torch
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX5_DIR = os.path.join(PROJECT_DIR, "programs", "ex005_rare_weighted_distill")
sys.path.insert(0, os.path.join(PROJECT_DIR, ".ai", "ai-dev-kit", "src"))


def _load(name: str, path: str):
    """ファイルからモジュールを別名で読み込む（同名のモジュールとの衝突を避ける）．

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


ex5_data = _load("ex005_data_for_v3", os.path.join(EX5_DIR, "data.py"))

# 教師ラベル・候補・埋め込みは ex005 のデータのディレクトリを共有する
DATASET_DIR = ex5_data.DATASET_DIR
image_path, load_pool = ex5_data.image_path, ex5_data.load_pool
FORMATS = ex5_data.FORMATS
ATTRIBUTES = ["has_blank", "has_karuta_mark", "has_text"]
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
FORMAT_IMPLIES = {"karuta": "has_karuta_mark", "anaume": "has_blank", "odaibun": "has_text"}
RARE = {"media": ["game", "painting"], "format": ["karuta"]}
CODE = {"P": "photo", "A": "anime", "G": "game", "I": "illustration"}


def cg_relabel() -> dict:
    """目視で CG とした画像の振り分け直した媒体を読み込む．

    Returns:
        dict[int, str]: お題ID → 媒体の名前．
    """
    out = {}
    for line in open(os.path.join(THIS_DIR, "cg_relabel.txt"), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            fid, c = line.split()
            out[int(fid)] = CODE[c]
    return out


def to_v3_media(df: pl.DataFrame) -> pl.DataFrame:
    """CG を含む媒体の表を v3 の媒体に変換する（CG の画像は cg_relabel の媒体を真にする）．

    Args:
        df (pl.DataFrame): 列 fileId と旧媒体8項目を持つ表．

    Returns:
        pl.DataFrame: 旧媒体の列を v3 の媒体7項目に置き換えた表．
    """
    rel = cg_relabel()
    cg = df["cg"].to_numpy().astype(bool)
    fids = df["fileId"].to_list()
    cols = {}
    for m in MEDIA:
        v = df[m].to_numpy().astype(bool).copy()
        for i in np.where(cg)[0]:
            if rel.get(fids[i]) == m:
                v[i] = True
        cols[m] = v
    return df.drop(["cg"]).with_columns([pl.Series(m, cols[m]) for m in MEDIA])


def load_eval_labels(split: str = "all") -> pl.DataFrame:
    """定義書 v3 の評価用ラベル（640枚）を読み込む．

    Args:
        split (str): "all"，"dev"，"test" のいずれか．

    Returns:
        pl.DataFrame: 列 fileId, pool, split, format と属性3項目・媒体7項目（bool）を持つ表．
    """
    df = ex5_data.load_eval_labels("all")
    df = df.with_columns((pl.col("has_prompt_text") | pl.col("has_other_text")).alias("has_text"))
    df = to_v3_media(df).select(["fileId", "pool", "format"] + ATTRIBUTES + MEDIA + ["split"])
    return df if split == "all" else df.filter(pl.col("split") == split)


def load_rare_labels() -> pl.DataFrame:
    """定義書 v3 の少ない区分の評価用ラベル（800枚，媒体とカルタのみ）を読み込む．

    Returns:
        pl.DataFrame: 列 fileId, set, stratum, split, karuta と媒体7項目を持つ表．
    """
    C = _load("ex005_combine_v8_for_v3", os.path.join(EX5_DIR, "combine_v8.py"))
    return to_v3_media(C.load_rare_eval())


def load_teacher_labels(name: str, extra: str = None) -> pl.DataFrame:
    """教師ラベルを読み込み，学習用・検証用の分割を付ける（ex005 と同じ規則）．

    Args:
        name (str): 基本の4万枚の教師ラベルの名前．
        extra (str): 少ない区分の追加分の教師ラベルの名前（None で追加しない）．

    Returns:
        pl.DataFrame: 列 fileId, split, source, format と各項目を持つ表．
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
        dict: キー format（形状 (N,)），attr（形状 (N, 3)），media（形状 (N, 7)）の配列．
    """
    return {"format": np.array([FORMATS.index(f) for f in df["format"]]),
            "attr": df.select(ATTRIBUTES).to_numpy().astype(np.float32),
            "media": df.select(MEDIA).to_numpy().astype(np.float32)}


class ThreeAxisDataset(torch.utils.data.Dataset):
    """お題画像と3軸のラベルを返すデータセット（画像エンコーダの微調整用）．"""

    def __init__(self, ids: list, y: dict, transform):
        """
        Args:
            ids (list[int]): お題IDのリスト．
            y (dict): label_arrays の辞書．
            transform (callable): PIL 画像をテンソルに変換する関数．
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
            tuple[torch.Tensor, dict]: 画像と，キー format・attr・media のラベル．
        """
        with Image.open(image_path(self.ids[i])) as im:
            x = self.transform(im.convert("RGB"))
        return x, {"format": int(self.y["format"][i]), "attr": torch.tensor(self.y["attr"][i]),
                   "media": torch.tensor(self.y["media"][i])}
