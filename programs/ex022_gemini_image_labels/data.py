"""ex022（Gemini と Claude Code のお題画像のラベルの比較）で用いるデータを扱うモジュール．

- Claude Code の目視のラベル（定義書 v4）：評価用データ640枚，実データの無作為抽出160枚，層別抽出220枚（ex008 の data.py で読み込む）．
- 自動のラベル：パッケージの最終版（Bokete_Annotation_programs_v5.0 の annotation.parquet）．
- Gemini のラベル：outputs/ex022_gemini_image_labels/{モデル}/raw/{お題ID}.json．
"""
import glob
import importlib.util
import json
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex022_gemini_image_labels")
ANNOTATION = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation", "annotation.parquet")
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
FORMATS = ["karuta", "anaume", "odaibun", "normal"]
ATTRIBUTES = ["has_blank", "has_karuta_mark", "has_text"]
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
LABEL_COLS = ["format"] + ATTRIBUTES + MEDIA


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


D8 = _load("ex008_data_for_ex022", os.path.join(PROJECT_DIR, "programs", "ex008_label_v4", "data.py"))


def image_path(fid: int) -> str:
    """お題IDから画像ファイルのパスを求める．

    Args:
        fid (int): お題ID．

    Returns:
        str: 画像ファイルのパス．
    """
    return os.path.join(IMAGE_DIR, str(fid // 10000 * 10000), f"{fid}.jpg")


def load_claude_labels() -> pl.DataFrame:
    """Claude Code の目視のラベル（定義書 v4）を3つの集合から読み込む．

    Returns:
        pl.DataFrame: 列 fileId, set（eval / random / strat）, stratum（strat のみ）, weight（strat のみ）と
        LABEL_COLS を持つ表（1,020行）．
    """
    ev = D8.load_eval_labels("all").select(["fileId"] + LABEL_COLS).with_columns(pl.lit("eval").alias("set"))
    rd = D8.load_real_labels("random").select(["fileId"] + LABEL_COLS).with_columns(pl.lit("random").alias("set"))
    st = D8.load_real_labels("strat").select(["fileId"] + LABEL_COLS + ["stratum", "weight"]).with_columns(pl.lit("strat").alias("set"))
    return pl.concat([ev, rd, st], how="diagonal")


def load_gemini_labels(model: str) -> pl.DataFrame:
    """Gemini のラベルを読み込む．

    Args:
        model (str): モデル名．

    Returns:
        pl.DataFrame: 列 fileId, LABEL_COLS, reason, seconds, prompt_tokens, output_tokens, thought_tokens を持つ表．
    """
    rows = []
    for p in glob.glob(os.path.join(OUTPUT_DIR, model, "raw", "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        rec = {"fileId": r["fileId"], "format": r["format"], **{a: bool(r[a]) for a in ATTRIBUTES}}
        rec.update({m: m in r["media"] for m in MEDIA})
        rec.update({k: r.get(k) for k in ["reason", "seconds", "prompt_tokens", "output_tokens", "thought_tokens"]})
        rows.append(rec)
    return pl.DataFrame(rows)


def load_auto_labels(ids: list) -> pl.DataFrame:
    """パッケージの最終版の自動のラベルを読み込む．

    Args:
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 fileId と LABEL_COLS を持つ表．
    """
    a = pl.scan_parquet(ANNOTATION).filter(pl.col("odaiId").is_in(ids)).select(["odaiId"] + LABEL_COLS).collect()
    return a.rename({"odaiId": "fileId"})
