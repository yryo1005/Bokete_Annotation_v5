"""ex025（お題文ありの評価用データ v5 と，パッケージへの組み込みの検証）で用いるデータを扱うモジュール．

- 新しい評価用データ：既存の目視のラベル（1,020枚）と ex024 の教師ラベル（4.5万枚）に含まれない代表の画像から，
  STRATA の層ごとに無作為に抽出した画像．実データの値は層の重み（層の枚数 / 抽出数）で推定する．
- 再現性の確認：既存の1,020枚から，お題文ありの正解（ex024）の3群ごとに REPEAT_PER_GROUP 枚を抽出した画像．
- 両者を混ぜて並べ替え，番号だけを付けた画像の一覧（sheets/）で目視のラベルを付ける（odai_labels.txt）．
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex025_odaibun_eval_v5")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex025_odaibun_eval_v5")
EX024_DIR = os.path.join(PROJECT_DIR, "programs", "ex024_odaibun_attr")
EX024_OUT = os.path.join(PROJECT_DIR, "outputs", "ex024_odaibun_attr")
TEACHER = "Qwen3-VL-32B-Instruct-FP8__802816__P3"
STUDENT_ALL = os.path.join(EX024_OUT, "pipeline", "student_all__emb+probs__lr0.001_ep40.parquet")
LABELS = os.path.join(THIS_DIR, "odai_labels.txt")
TARGETS = os.path.join(DATASET_DIR, "targets.csv")
THRESHOLD = 0.3
FORMATS = ["karuta", "anaume", "odaibun", "normal"]
# 層：現在の生徒の形式（確率が最大のもの）・確認の質問の回答・お題文ありの生徒の確率 p で分ける（名前: 抽出数）
STRATA = {"A_odaibun_yes": 100, "B_odaibun_rejected": 40, "C_blank_p_hi": 110, "D_blank_p_lo": 50, "E_normal_p_mid": 60,
          "F_normal_p_lo": 60}
NORMAL_P_MID = 0.1
REPEAT_PER_GROUP = 30
SEED = 0


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


EX024 = _load("ex024_data_for_ex025", os.path.join(EX024_DIR, "data.py"))
ANNOTATION = EX024.ANNOTATION
# パッケージ v5.2.0 の出力（v5.3.0 で finalize し直す前に複製したもの）．層と「現在のパッケージ」の構成はこの出力で求める
ANNOTATION_V52 = os.path.join(os.path.dirname(ANNOTATION), "backup_v5.2.0", "annotation.parquet")
image_path = EX024.image_path
load_truth_ex024 = EX024.load_truth


def stratum_of(fmt: np.ndarray, rejected: np.ndarray, p: np.ndarray) -> np.ndarray:
    """層の名前を求める．

    Args:
        fmt (np.ndarray): 形状 (N,) の現在の生徒の形式（確率が最大のもの．karuta / anaume / odaibun / normal）．
        rejected (np.ndarray): 形状 (N,) の確認の質問が偽と答えたか（bool）．
        p (np.ndarray): 形状 (N,) のお題文ありの生徒の確率．

    Returns:
        np.ndarray: 形状 (N,) の層の名前．
    """
    bk = np.isin(fmt, ["anaume", "karuta"])
    return np.select([(fmt == "odaibun") & ~rejected, (fmt == "odaibun") & rejected, bk & (p >= THRESHOLD), bk,
                      p >= NORMAL_P_MID], list(STRATA)[:5], list(STRATA)[5])


def load_population() -> pl.DataFrame:
    """全ての代表の画像の予測（パッケージ v5.2.0 と，お題文ありの生徒）と層を読み込む．

    Returns:
        pl.DataFrame: 列 rep_odaiId, format_argmax, format（確認の質問を反映した形式）, has_blank, has_karuta_mark,
        verify_asked, rejected, p（お題文ありの生徒の確率）, stratum を持つ表（約661万行）．
    """
    pf = [f"p_format_{c}" for c in FORMATS]
    src = ANNOTATION_V52 if os.path.exists(ANNOTATION_V52) else ANNOTATION
    a = pl.scan_parquet(src).filter(pl.col("odaiId") == pl.col("rep_odaiId")).select(
        "rep_odaiId", "format", "has_blank", "has_karuta_mark", "verify_asked", "verify_answer", *pf).collect()
    am = np.array(FORMATS)[a.select(pf).to_numpy().argmax(1)]
    a = a.with_columns(pl.Series("format_argmax", am), (pl.col("verify_asked") & ~pl.col("verify_answer").fill_null(True)).alias("rejected"))
    a = a.join(pl.read_parquet(STUDENT_ALL), on="rep_odaiId", how="left")
    st = stratum_of(a["format_argmax"].to_numpy(), a["rejected"].to_numpy(), a["prob"].to_numpy())
    return a.select("rep_odaiId", "format_argmax", "format", "has_blank", "has_karuta_mark", "verify_asked", "rejected",
                    pl.col("prob").alias("p"), pl.Series("stratum", st))


def excluded_reps() -> set:
    """新しい評価用データから除く代表の画像（既存の目視のラベルと ex024 の教師ラベルの画像）を求める．

    Returns:
        set[int]: 代表の画像のお題IDの集合．
    """
    ids = set(load_truth_ex024()["fileId"].to_list())
    for p in glob.glob(os.path.join(EX024_OUT, "teacher", TEACHER, "*.csv")):
        ids |= set(pl.read_csv(p, columns=["fileId"])["fileId"].to_list())
    rep = pl.scan_parquet(ANNOTATION).select("odaiId", "rep_odaiId").filter(pl.col("odaiId").is_in(list(ids))).collect()
    return set(rep["rep_odaiId"].to_list())


def load_labels() -> pl.DataFrame:
    """目視で付けたお題文ありのラベルを読み込む．

    書式は「番号 0/1 [メモ]」（番号は targets.csv の no）．

    Returns:
        pl.DataFrame: 列 no, label（bool）, memo を持つ表．
    """
    rows = []
    for line in open(LABELS, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            parts = line.rstrip("\n").split(maxsplit=2)
            rows.append({"no": int(parts[0]), "label": parts[1] == "1", "memo": parts[2] if len(parts) > 2 else ""})
    return pl.DataFrame(rows)


def load_eval() -> pl.DataFrame:
    """新しい評価用データ（目視のラベルと層の重み）を読み込む．

    Returns:
        pl.DataFrame: targets.csv の列（no, rep_odaiId, kind, stratum, N_h, n_h）と label, memo, w を持つ表（kind が new の行）．
    """
    t = pl.read_csv(TARGETS).join(load_labels(), on="no", how="left")
    assert t["label"].null_count() == 0, "目視のラベルが欠けている"
    return t.filter(pl.col("kind") == "new").with_columns((pl.col("N_h") / pl.col("n_h")).alias("w"))


def load_repeat() -> pl.DataFrame:
    """再現性の確認の画像の，付け直したラベルと過去のラベルを読み込む．

    Returns:
        pl.DataFrame: 列 no, rep_odaiId, group, label（付け直し）, label_old（ex024 の正解）, memo を持つ表．
    """
    t = pl.read_csv(TARGETS).filter(pl.col("kind") == "repeat").join(load_labels(), on="no", how="left")
    old = load_truth_ex024().select(pl.col("fileId").alias("rep_odaiId"), pl.col("odaibun_attr").alias("label_old"))
    return t.join(old, on="rep_odaiId", how="left").select("no", "rep_odaiId", pl.col("stratum").alias("group"), "label", "label_old", "memo")
