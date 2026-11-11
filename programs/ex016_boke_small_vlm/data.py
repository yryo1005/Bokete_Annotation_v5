"""ex016（小さな VLM によるボケのラベルの判定）で用いるデータを扱うモジュール．

- 検証データ：ex015 の620件（発話の型の6区分，定義書 第3版の画像外の知識，調整用 dev と最終評価用 test の分割）．
- 検証データの正解は，既定で合意ラベル（consensus_labels.txt：620件を元のラベルを見ずに付け直し，不一致を裁定したもの）とする．
- 学習データ：ex015 の教師（Qwen3-VL-32B，one_kt_cap）が付けたラベル．ex014 の水増し用の集合（9,041件）と，
  ex013 の学習用のボケから重複と検証データのお題を除いた集合（25,593件）の計34,634件．
- ex013〜ex015 のモジュールは名前が衝突するため別名で読み込む．
"""
import importlib.util
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX13_DIR = os.path.join(PROJECT_DIR, "programs", "ex013_boke_labels")
EX14_DIR = os.path.join(PROJECT_DIR, "programs", "ex014_boke_labels_v2")
EX15_DIR = os.path.join(PROJECT_DIR, "programs", "ex015_boke_labels_v3")
EX13_DATASET = os.path.join(PROJECT_DIR, "datasets", "ex013_boke_labels")
EX14_DATASET = os.path.join(PROJECT_DIR, "datasets", "ex014_boke_labels_v2")
EX15_OUTPUT = os.path.join(PROJECT_DIR, "outputs", "ex015_boke_labels_v3")
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex016_boke_small_vlm")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex016_boke_small_vlm")
VOICES = ["S", "M", "T", "N", "L", "A"]
VOICE3 = {"S": "C", "M": "C", "T": "X", "N": "X", "L": "X", "A": "A"}
QA_FORMATS = ("anaume", "odaibun")
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]


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


D15 = _load("ex015_data_for_v4", os.path.join(EX15_DIR, "data.py"))
load_odai_labels, image_path = D15.load_odai_labels, D15.D14.image_path


def load_eval(consensus: bool = True) -> pl.DataFrame:
    """検証データ（620件）を読み込む．

    Args:
        consensus (bool): True なら発話の型と画像外の知識を合意ラベル（consensus_labels.txt）にする．False なら ex015 の元のラベル．

    Returns:
        pl.DataFrame: 列 no, odaiId, text, format, split, voice, voice3, outside, has_text, 媒体, caption などを持つ表．
    """
    e = D15.load_eval()
    if not consensus:
        return e
    rows = [ln.split()[:3] for ln in open(os.path.join(THIS_DIR, "consensus_labels.txt"), encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    c = pl.DataFrame({"no": [int(r[0]) for r in rows], "voice_c": [r[1] for r in rows], "outside_c": [r[2] == "1" for r in rows]})
    e = e.join(c, on="no", how="left")
    assert e["voice_c"].null_count() == 0
    return e.with_columns(pl.col("voice_c").alias("voice"), pl.col("outside_c").alias("outside"),
                          pl.col("voice_c").replace_strict(VOICE3).alias("voice3")).drop("voice_c", "outside_c")


def noise_transition(variant: str = "one_kt_cap", split: str = "dev") -> tuple:
    """教師のラベルの誤り方（真のラベル → 教師のラベルの遷移行列）を，検証データの一方の分割の合意ラベルで推定する．

    行列の各行は真のラベルごとの教師のラベルの割合（平滑化のため各セルに0.5を足す）である．

    Args:
        variant (str): ex015 の教師の方法．
        split (str): 推定に用いる分割（dev）．

    Returns:
        tuple[np.ndarray, np.ndarray]: 発話の型の遷移行列 (6, 6) と画像外の知識の遷移行列 (2, 2)．
    """
    import numpy as np

    e = load_eval().filter(pl.col("split") == split).select("no", "voice", "outside")
    t = pl.read_csv(os.path.join(EX15_OUTPUT, "teacher_q32", variant, "eval.csv")).select("no", pl.col("voice").alias("tv"),
                                                                                            pl.col("outside").alias("to"))
    j = e.join(t, on="no")
    mv = np.full((len(VOICES), len(VOICES)), 0.5)
    for a, b in zip(j["voice"], j["tv"]):
        mv[VOICES.index(a), VOICES.index(b)] += 1
    mo = np.full((2, 2), 0.5)
    for a, b in zip(j["outside"], j["to"]):
        mo[int(a), int(b)] += 1
    return mv / mv.sum(1, keepdims=True), mo / mo.sum(1, keepdims=True)


def load_human() -> pl.DataFrame:
    """学習用の目視（Claude）のラベル（human_labels.txt，931件．検証データのお題は含まない）を読み込む．

    Returns:
        pl.DataFrame: 列 hid, odaiId, text, format, stratum, voice, outside を持つ表．
    """
    rows = [ln.split() for ln in open(os.path.join(THIS_DIR, "human_labels.txt"), encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    lab = pl.DataFrame({"hid": [int(r[0][1:]) for r in rows], "voice": [r[1] for r in rows], "outside": [r[2] == "1" for r in rows]})
    items = pl.read_csv(os.path.join(DATASET_DIR, "human_items.csv"))
    out = items.join(lab, on="hid", how="inner")
    assert out.height == items.height
    return out


def load_train(variant: str = "one_kt_cap") -> pl.DataFrame:
    """教師（32B）のラベルを付けた学習データを読み込む．

    Args:
        variant (str): ex015 の教師の方法．

    Returns:
        pl.DataFrame: 列 odaiId, text, format, has_text, 媒体, caption, voice, outside, source（aug / extra）を持つ表．
    """
    d = os.path.join(EX15_OUTPUT, "teacher_q32", variant)
    aug = pl.read_csv(os.path.join(EX14_DATASET, "aug_pool.csv")).select("odaiId", "text")
    aug = pl.concat([aug, pl.read_csv(os.path.join(d, "aug.csv")).select("voice", "outside")], how="horizontal")
    pool = pl.read_parquet(os.path.join(EX13_DATASET, "train_pool.parquet")).with_row_index("train_index").select(
        pl.col("train_index").cast(pl.Int64), "text")
    extra = pl.read_csv(os.path.join(d, "extra.csv")).join(pool, on="train_index", how="left", maintain_order="left")
    df = pl.concat([aug.with_columns(pl.lit("aug").alias("source")),
                    extra.select("odaiId", "text", "voice", "outside").with_columns(pl.lit("extra").alias("source"))])
    df = df.filter(pl.col("voice").is_not_null() & pl.col("outside").is_not_null())
    lab = load_odai_labels(df["odaiId"].unique().to_list()).select(["odaiId", "format", "has_text", "caption"] + MEDIA)
    return df.join(lab, on="odaiId", how="left", maintain_order="left")
