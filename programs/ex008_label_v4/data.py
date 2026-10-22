"""ex008（定義書 v4）で用いるデータを扱うモジュール．

定義書 v4（programs/label_guideline/label_definition_v4.md）は，v3 の「文字あり」を，ロゴ・透かし・著作権表示・
看板・衣服の文字を含め，読める文字があれば真とするように変えたものである．形式・媒体と他の属性は v3 と同じである．
ex007 の data.py を別名で読み込み，同じ関数名で定義する（ex007 の students.py，common.py をそのまま用いるため）．

- 評価用ラベル（640枚）：ex007 のラベルの文字ありを text_relabel.txt で上書きする．
- 実データの目視ラベル：無作為抽出160枚（real_random_v4.txt）と，生徒の予測による層別抽出（real_strat_v4.txt）．
"""
import importlib.util
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX7_DIR = os.path.join(PROJECT_DIR, "programs", "ex007_label_v3")


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


ex7 = _load("ex007_data_for_v4", os.path.join(EX7_DIR, "data.py"))
DATASET_DIR, EX5_DIR = ex7.DATASET_DIR, ex7.EX5_DIR
image_path, load_pool, label_arrays, ThreeAxisDataset = ex7.image_path, ex7.load_pool, ex7.label_arrays, ex7.ThreeAxisDataset
load_rare_labels, load_teacher_labels = ex7.load_rare_labels, ex7.load_teacher_labels
FORMATS, ATTRIBUTES, MEDIA, FORMAT_IMPLIES, RARE = ex7.FORMATS, ex7.ATTRIBUTES, ex7.MEDIA, ex7.FORMAT_IMPLIES, ex7.RARE
CHECK_DIR = os.path.join(DATASET_DIR, "real_check_v4")
F_CODE = {"karuta": "K", "anaume": "A", "odaibun": "O", "normal": "N"}
A_CODE = {"has_blank": "b", "has_karuta_mark": "k", "has_text": "t"}
M_CODE = {"photo": "P", "anime": "A", "game": "G", "manga": "M", "text_graphic": "T", "painting": "E", "illustration": "I"}


def text_relabel() -> dict:
    """定義書 v4 で見直した評価用データの文字ありを読み込む．

    Returns:
        dict[int, bool]: お題ID → 文字あり．
    """
    out = {}
    path = os.path.join(THIS_DIR, "text_relabel.txt")
    for line in open(path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            fid, v = line.split()
            out[int(fid)] = v == "1"
    return out


def load_eval_labels(split: str = "all") -> pl.DataFrame:
    """定義書 v4 の評価用ラベル（640枚）を読み込む．

    Args:
        split (str): "all"，"dev"，"test" のいずれか．

    Returns:
        pl.DataFrame: 列 fileId, pool, format と属性3項目・媒体7項目（bool），split を持つ表．
    """
    df = ex7.load_eval_labels("all")
    rel = text_relabel()
    df = df.with_columns(pl.Series("has_text", [rel.get(f, t) for f, t in zip(df["fileId"], df["has_text"])]))
    return df if split == "all" else df.filter(pl.col("split") == split)


def parse_codes(path: str, ids: list) -> pl.DataFrame:
    """「番号 形式 属性 媒体」の書式の目視ラベルを読み込む．

    Args:
        path (str): ラベルのファイル．
        ids (list[int]): 番号に対応するお題IDのリスト．

    Returns:
        pl.DataFrame: 列 fileId, format と属性3項目・媒体7項目（bool）を持つ表（ids の順）．
    """
    inv_f = {v: k for k, v in F_CODE.items()}
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            k, f, a, m = line.split()
            r = {"fileId": ids[int(k)], "format": inv_f[f]}
            r.update({x: (c in a) for x, c in A_CODE.items()})
            r.update({x: (c in m) for x, c in M_CODE.items()})
            rows[int(k)] = r
    assert len(rows) == len(ids), f"{path}: {len(rows)} != {len(ids)}"
    return pl.DataFrame([rows[k] for k in range(len(ids))])


def load_real_labels(kind: str) -> pl.DataFrame:
    """実データの目視ラベルを読み込む．

    Args:
        kind (str): "random"（無作為抽出160枚）または "strat"（層別抽出）．

    Returns:
        pl.DataFrame: 列 fileId, format と各項目を持つ表．kind="strat" では列 stratum と weight（抽出確率の逆数）も持つ．
    """
    if kind == "random":
        ids = pl.read_csv(os.path.join(DATASET_DIR, "real_check_v3", "sample.csv"))["fileId"].to_list()
        return parse_codes(os.path.join(THIS_DIR, "real_random_v4.txt"), ids)
    s = pl.read_csv(os.path.join(CHECK_DIR, "strat_sample.csv"))
    return parse_codes(os.path.join(THIS_DIR, "real_strat_v4.txt"), s["fileId"].to_list()).with_columns(
        s["stratum"], s["weight"])
