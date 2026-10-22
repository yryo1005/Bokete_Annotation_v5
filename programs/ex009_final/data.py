"""ex009（最終版の検証）で用いるデータを扱うモジュール．

定義書 v4 のラベルと評価用データは ex008 と同じであり，ex008 の data.py を別名で読み込んで同じ名前で公開する．
"""
import importlib.util
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX8_DIR = os.path.join(PROJECT_DIR, "programs", "ex008_label_v4")


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


ex8 = _load("ex008_data_for_final", os.path.join(EX8_DIR, "data.py"))
DATASET_DIR, EX5_DIR, CHECK_DIR = ex8.DATASET_DIR, ex8.EX5_DIR, ex8.CHECK_DIR
image_path, load_pool, label_arrays, ThreeAxisDataset = ex8.image_path, ex8.load_pool, ex8.label_arrays, ex8.ThreeAxisDataset
load_rare_labels, load_teacher_labels = ex8.load_rare_labels, ex8.load_teacher_labels
FORMATS, ATTRIBUTES, MEDIA, FORMAT_IMPLIES, RARE = ex8.FORMATS, ex8.ATTRIBUTES, ex8.MEDIA, ex8.FORMAT_IMPLIES, ex8.RARE
F_CODE, A_CODE, M_CODE = ex8.F_CODE, ex8.A_CODE, ex8.M_CODE
TEACHER = "T_v10oJXk"


def format_relabel() -> dict:
    """定義書 v4 の補足で形式を見直した目視ラベルを読み込む．

    Returns:
        dict[int, str]: お題ID → 形式．
    """
    out = {}
    for line in open(os.path.join(THIS_DIR, "format_relabel.txt"), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            fid, f = line.split()
            out[int(fid)] = f
    return out


def _apply_format_relabel(df):
    """表の形式を format_relabel で上書きし，形式から導かれる属性（お題文なら文字あり）を整える．

    Args:
        df (pl.DataFrame): 列 fileId, format, has_text を持つ表．

    Returns:
        pl.DataFrame: 上書きした表．
    """
    import polars as pl

    rel = format_relabel()
    df = df.with_columns(pl.Series("format", [rel.get(f, x) for f, x in zip(df["fileId"], df["format"])]))
    return df.with_columns(pl.col("has_text") | (pl.col("format") == "odaibun"))


def load_eval_labels(split: str = "all"):
    """評価用ラベル（640枚，定義書 v4 と補足）を読み込む．

    Args:
        split (str): "all"，"dev"，"test" のいずれか．

    Returns:
        pl.DataFrame: ex008 の load_eval_labels と同じ列の表．
    """
    return _apply_format_relabel(ex8.load_eval_labels(split))


def load_real_labels(kind: str):
    """実データの目視ラベル（定義書 v4 と補足）を読み込む．

    Args:
        kind (str): "random" または "strat"．

    Returns:
        pl.DataFrame: ex008 の load_real_labels と同じ列の表．
    """
    return _apply_format_relabel(ex8.load_real_labels(kind))


def check_ids() -> list:
    """実データの目視ラベルの画像（無作為抽出160枚と層別抽出220枚）のお題IDを返す．

    Returns:
        list[int]: お題IDのリスト（380件）．
    """
    import polars as pl

    return pl.read_csv(os.path.join(DATASET_DIR, "candidates", "real_check_v4.csv"))["fileId"].to_list()
