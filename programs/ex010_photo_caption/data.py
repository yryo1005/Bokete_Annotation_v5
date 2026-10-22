"""ex010（写真のキャプションの生成）で用いるデータを扱うモジュール．

- 対象：Bokete_Annotation_programs_v5.0 のアノテーション（読み込みのみ）で，媒体が写真のみの画像（内容が同じ画像は代表のみ）．
- 評価用データ：対象から無作為に抽出した120枚（調整用 dev 40枚，最終評価用 test 80枚）．参照キャプションは
  Claude Code が目視で付けた（reference_captions.txt）．
- 速度の測定用データ：評価用データ以外から無作為に抽出した2,000枚．
- 近傍の検索の候補（キャプションを付けておく画像の集合）：評価用データ以外から無作為に抽出した画像．
"""
import os

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ANN_DIR = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation")
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex010_photo_caption")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex010_photo_caption")
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
N_EVAL, N_DEV, N_SPEED = 120, 40, 2000


def image_path(odai_id: int) -> str:
    """お題IDから画像ファイルのパスを求める．

    Args:
        odai_id (int): お題ID．

    Returns:
        str: 画像ファイルの絶対パス．
    """
    return os.path.join(IMAGE_DIR, str(odai_id // 10000 * 10000), f"{odai_id}.jpg")


def load_photo_ids() -> list:
    """媒体が写真のみの代表の画像のお題IDを返す（昇順）．

    Returns:
        list[int]: お題IDのリスト．
    """
    ann = pl.scan_parquet(os.path.join(ANN_DIR, "annotation.parquet")).filter(
        (pl.col("odaiId") == pl.col("rep_odaiId")) & pl.col("photo")
        & ~pl.any_horizontal([pl.col(m) for m in MEDIA if m != "photo"])).select("odaiId").collect()
    return sorted(ann["odaiId"].to_list())


def make_splits(seed: int = 0) -> pl.DataFrame:
    """評価用データと速度の測定用データを抽出して保存する（既にあれば読み込む）．

    Args:
        seed (int): 乱数のシード．

    Returns:
        pl.DataFrame: 列 fileId, split（dev / test / speed）を持つ表．
    """
    path = os.path.join(DATASET_DIR, "splits.csv")
    if os.path.exists(path):
        return pl.read_csv(path)
    ids = np.array(load_photo_ids())
    rng = np.random.default_rng(seed)
    pick = rng.choice(ids, N_EVAL + N_SPEED, replace=False)
    split = ["dev"] * N_DEV + ["test"] * (N_EVAL - N_DEV) + ["speed"] * N_SPEED
    df = pl.DataFrame({"fileId": pick.tolist(), "split": split})
    os.makedirs(DATASET_DIR, exist_ok=True)
    df.write_csv(path)
    return df


def eval_ids(split: str = "all") -> list:
    """評価用データのお題IDを返す．

    Args:
        split (str): "dev"，"test"，"all"．

    Returns:
        list[int]: お題IDのリスト（抽出の順）．
    """
    df = make_splits().filter(pl.col("split").is_in(["dev", "test"] if split == "all" else [split]))
    return df["fileId"].to_list()


def load_references() -> dict:
    """参照キャプション（reference_captions.txt．書式は「お題ID<TAB>キャプション」）を読み込む．

    Returns:
        dict[int, str]: お題ID → 参照キャプション．
    """
    out = {}
    for line in open(os.path.join(THIS_DIR, "reference_captions.txt"), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            fid, cap = line.rstrip("\n").split("\t")
            out[int(fid)] = cap
    return out
