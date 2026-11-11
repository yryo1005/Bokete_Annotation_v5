"""ex029（現実画像の部分集合：写真・文字なしと写真・通常の件数と判定精度）で用いるデータを扱うモジュール．

- 予測：パッケージ v5.3.0 の出力（annotation.parquet）．媒体が写真のみ（他の媒体を含まない）を「写真のみ」とする．
- 正解：Claude Code の目視のラベル（ex022 / ex023 の1,020枚．定義書 v4 の形式・属性・媒体）．
  同じ1,020枚には ex026 の定義書 v6 の文の属性もあり，規則 R1b のお題文ありで通常を導いた正解（v6）も作る．
- 集合：real（無作為抽出160枚と層別抽出220枚を重み付けした実データの推定），eval（評価用データ640枚．重みなし）．
- ボケ：パッケージのボケのアノテーション（boke_annotation/part_*.parquet）のお題ごとの件数と画像外の知識の割合．
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex029_photo_subset")
ANN_DIR = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation")
ANNOTATION = os.path.join(ANN_DIR, "annotation.parquet")
BOKE_PARTS = os.path.join(ANN_DIR, "boke_annotation", "part_*.parquet")
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
# 条件の名前と日本語名
CONDITIONS = {"photo_only": "写真のみ", "A_photo_notext": "写真・文字なし", "B_photo_normal": "写真・通常"}


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


def conditions(photo: pl.Expr, others: list, has_text: pl.Expr, normal: pl.Expr) -> list:
    """3つの条件の式を作る．

    Args:
        photo (pl.Expr): 写真か（bool）．
        others (list[pl.Expr]): 写真以外の媒体の bool の式．
        has_text (pl.Expr): 文字ありか．
        normal (pl.Expr): 形式が通常か．

    Returns:
        list[pl.Expr]: CONDITIONS の名前の列．
    """
    only = photo & ~pl.any_horizontal(others)
    return [only.alias("photo_only"), (only & ~has_text).alias("A_photo_notext"), (only & normal).alias("B_photo_normal")]


def load_pred(ids: list = None) -> pl.DataFrame:
    """v5.3.0 の出力から条件の予測を作る．

    Args:
        ids (list[int] | None): お題ID．None なら全てのお題．

    Returns:
        pl.DataFrame: 列 odaiId, rep_odaiId, format, has_text と CONDITIONS の列を持つ表．
    """
    a = pl.scan_parquet(ANNOTATION)
    if ids is not None:
        a = a.filter(pl.col("odaiId").is_in(pl.Series(ids).implode()))
    a = a.select("odaiId", "rep_odaiId", "format", "has_text", *MEDIA).collect()
    return a.with_columns(*conditions(pl.col("photo"), [pl.col(m) for m in MEDIA[1:]], pl.col("has_text"), pl.col("format") == "normal"))


def load_truth() -> pl.DataFrame:
    """1,020枚の目視の正解（v4 と v6 の通常）と重みを読み込む．

    Returns:
        pl.DataFrame: 列 fileId, set, w, stratum と CONDITIONS の列（v4 の正解），A_photo_notext_v6（文字ありを ex026 の v6 の
        文字の層で付け直した正解），B_photo_normal_v6（v6 の R1b で導いた通常），
        has_text_v4, has_text_v6 を持つ表．
    """
    e23 = _load("ex023_data_for_ex029", os.path.join(PROJECT_DIR, "programs", "ex023_image_label_pr", "data.py"))
    t = e23.load_truth()
    t = t.with_columns(*conditions(pl.col("photo"), [pl.col(m) for m in MEDIA[1:]], pl.col("has_text"), pl.col("format_normal")))
    e26 = _load("ex026_data_for_ex029", os.path.join(PROJECT_DIR, "programs", "ex026_text_attr_v6", "data.py"))
    v6 = e26.load_all().select(pl.col("rep_odaiId").alias("fileId"), "odai_R1b",
                               pl.any_horizontal([pl.col(f"L_{l}") for l in e26.LAYERS]).alias("has_text_v6"))
    t = t.join(v6, on="fileId", how="left")
    b6 = pl.col("photo_only") & ~pl.col("has_blank") & ~pl.col("has_karuta_mark") & ~pl.col("odai_R1b")
    a6 = pl.col("photo_only") & ~pl.col("has_text_v6")
    return t.select("fileId", "set", "w", "stratum", *CONDITIONS, a6.alias("A_photo_notext_v6"), b6.alias("B_photo_normal_v6"),
                    pl.col("has_text").alias("has_text_v4"), "has_text_v6")


def count_boke() -> pl.DataFrame:
    """お題ごとのボケの件数と，画像外の知識の件数を数える（初回は outputs に保存する）．

    Returns:
        pl.DataFrame: 列 odaiId, n_boke, n_outside を持つ表．
    """
    cache = os.path.join(OUTPUT_DIR, "boke_counts.parquet")
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    b = pl.scan_parquet(sorted(glob.glob(BOKE_PARTS))).group_by("odaiId").agg(
        pl.len().alias("n_boke"), pl.col("outside").fill_null(False).sum().alias("n_outside")).collect()
    b.write_parquet(cache)
    return b


def boot_groups(t: pl.DataFrame) -> list:
    """復元抽出の群（無作為抽出は1群，層別抽出は層ごと）の行番号を求める．

    Args:
        t (pl.DataFrame): load_truth の表の real の行．

    Returns:
        list[np.ndarray]: 群ごとの行番号．
    """
    key = (t["set"] + ":" + t["stratum"].fill_null("")).to_numpy()
    return [np.flatnonzero(key == k) for k in np.unique(key)]
