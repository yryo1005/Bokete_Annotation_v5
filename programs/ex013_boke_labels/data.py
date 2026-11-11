"""ex013（ボケのラベル）で用いるデータを扱うモジュール．

- 検証データ：datasets/ex013_boke_labels/eval_bokes.csv（420件）と eval_labels.txt（目視のラベル）．
  お題の単位で，層ごとに半数ずつ調整用（dev）と最終評価用（test）に分ける．
- お題のラベル：Bokete_Annotation_programs_v5.0 のアノテーション（読み込みのみ）．
- 学習用のボケ：検証データ以外のお題から，層ごとに抽出する（make_train_pool）．
"""
import os

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ANN_DIR = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation")
DB_DIR = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4"
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex013_boke_labels")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex013_boke_labels")
VOICES = ["S", "M", "T", "N", "L", "A"]
KNOWS = ["0", "G", "I", "O"]
STRATA = ["karuta", "anaume", "odaibun", "photo_text", "photo_notext", "anime_manga", "other"]
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]


def image_path(odai_id: int) -> str:
    """お題IDから画像ファイルのパスを求める．

    Args:
        odai_id (int): お題ID．

    Returns:
        str: 画像ファイルのパス．
    """
    return os.path.join(IMAGE_DIR, str(odai_id // 10000 * 10000), f"{odai_id}.jpg")


def load_odai_labels(ids: list = None) -> pl.DataFrame:
    """お題のラベル（形式，属性，媒体，キャプション）を読み込む．

    Args:
        ids (list[int]): お題IDのリスト（None で全て）．

    Returns:
        pl.DataFrame: 列 odaiId, rep_odaiId, format, has_blank, has_karuta_mark, has_text, 媒体7項目, caption を持つ表．
    """
    q = pl.scan_parquet(os.path.join(ANN_DIR, "annotation.parquet")).select(
        ["odaiId", "rep_odaiId", "format", "has_blank", "has_karuta_mark", "has_text"] + MEDIA + ["caption"])
    if ids is not None:
        q = q.filter(pl.col("odaiId").is_in(ids))
    return q.collect()


def load_eval() -> pl.DataFrame:
    """検証データ（420件）を，目視のラベル，お題のラベル，分割とともに読み込む．

    Returns:
        pl.DataFrame: 列 no, stratum, odaiId, kind, rateSum, text, comp, voice, know, uses_text, inappropriate, mislabeled,
        split（dev / test）とお題のラベルを持つ表．
    """
    b = pl.read_csv(os.path.join(DATASET_DIR, "eval_bokes.csv"))
    rows = [ln.split() for ln in open(os.path.join(THIS_DIR, "eval_labels.txt"), encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    lab = pl.DataFrame({"no": [int(r[0]) for r in rows], "comp": [r[1] for r in rows], "voice": [r[2] for r in rows],
                        "know": [r[3] for r in rows], "uses_text": [r[4] == "1" for r in rows],
                        "inappropriate": [r[5] == "1" for r in rows], "mislabeled": [r[6] == "m" for r in rows]})
    d = b.join(lab, on="no")
    rng = np.random.default_rng(0)
    split = {}
    for st in STRATA:
        ids = sorted(d.filter(pl.col("stratum") == st)["odaiId"].unique().to_list())
        perm = rng.permutation(ids)
        for k, fid in enumerate(perm):
            split[int(fid)] = "dev" if k < len(perm) // 2 else "test"
    d = d.with_columns(pl.Series("split", [split[i] for i in d["odaiId"].to_list()]))
    return d.join(load_odai_labels(d["odaiId"].unique().to_list()), on="odaiId", how="left")


def load_bokes(ids: list, columns: tuple = ("odaiId", "text", "rateSum")) -> pl.DataFrame:
    """お題のボケを読み込む（ボケの表はお題IDを10000で割った値＋1の番号のファイルにある）．

    Args:
        ids (list[int]): お題IDのリスト．
        columns (tuple[str]): 読み込む列．

    Returns:
        pl.DataFrame: ボケの表．
    """
    files = sorted({os.path.join(DB_DIR, "boke", f"boke_{i // 10000 + 1}.parquet") for i in ids})
    return pl.scan_parquet(files).filter(pl.col("odaiId").is_in(ids)).select(list(columns)).collect()
