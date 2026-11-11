"""ex024（お題文ありの属性の判定の改善）で用いるデータを扱うモジュール．

- 正解：Claude Code の目視のラベル（定義書 v4，1,020枚）に，お題文ありの属性を加えたもの．
  形式がお題文 → お題文あり，通常 → お題文なし，穴埋め・カルタ → 目視で付け直したラベル（odaibun_relabel.txt）．
- 集合：dev（評価用データの調整用320枚），test（評価用データの最終評価用320枚），real（実データの無作為抽出160枚と層別抽出220枚．重み付き）．
"""
import importlib.util
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG_DIR = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex024_odaibun_attr")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex024_odaibun_attr")
ANNOTATION = os.path.join(PKG_DIR, "outputs", "BoketeDBv5_annotation", "annotation.parquet")
RELABEL = os.path.join(THIS_DIR, "odaibun_relabel.txt")


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


EX022 = _load("ex022_data_for_ex024", os.path.join(PROJECT_DIR, "programs", "ex022_gemini_image_labels", "data.py"))
image_path = EX022.image_path


def load_truth_v4() -> pl.DataFrame:
    """定義書 v4 の目視のラベルを，集合（dev / test / real）と重みを付けて読み込む．

    実データの重みは，無作為抽出が「母集団の枚数 / 160」，層別抽出が抽出確率の逆数である（ex009 と同じ）．評価用データの重みは1とする．

    Returns:
        pl.DataFrame: 列 fileId, set（eval / random / strat）, split（dev / test / real）, w, stratum, format と属性・媒体の列を持つ表（1,020行）．
    """
    c = EX022.load_claude_labels()
    split = EX022.D8.load_eval_labels("all").select("fileId", "split")
    c = c.join(split, on="fileId", how="left").with_columns(pl.col("split").fill_null("real"))
    n_pop = float(c.filter(pl.col("set") == "strat")["weight"].sum())
    n_rand = c.filter(pl.col("set") == "random").height
    w = pl.when(pl.col("set") == "random").then(n_pop / n_rand).when(pl.col("set") == "strat").then(pl.col("weight")).otherwise(1.0)
    return c.with_columns(w.alias("w"), pl.col("stratum").fill_null(""))


def load_relabel() -> pl.DataFrame:
    """穴埋め・カルタの画像に目視で付けたお題文ありのラベルを読み込む．

    書式は「番号 0/1 [メモ]」（番号は relabel_targets.csv の no）．

    Returns:
        pl.DataFrame: 列 fileId, odaibun_attr を持つ表．
    """
    tg = pl.read_csv(os.path.join(DATASET_DIR, "relabel_targets.csv"))
    rows = []
    for line in open(RELABEL, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            no, v = line.split()[:2]
            rows.append({"no": int(no), "odaibun_attr": v == "1"})
    return tg.join(pl.DataFrame(rows), on="no").select("fileId", "odaibun_attr")


def load_truth() -> pl.DataFrame:
    """お題文ありの属性を加えた正解を読み込む．

    Returns:
        pl.DataFrame: load_truth_v4 の列と odaibun_attr（bool）を持つ表．
    """
    t = load_truth_v4()
    r = load_relabel()
    t = t.join(r, on="fileId", how="left")
    base = pl.when(pl.col("format") == "odaibun").then(True).when(pl.col("format") == "normal").then(False).otherwise(None)
    t = t.with_columns(pl.coalesce(pl.col("odaibun_attr"), base).alias("odaibun_attr"))
    assert t["odaibun_attr"].null_count() == 0, "お題文ありの目視のラベルが欠けている"
    return t


ANN_DIR = os.path.dirname(ANNOTATION)


def load_embeddings(ids: list) -> "np.ndarray":
    """パッケージが保存した SigLIP2-so400m の画像埋め込みを，お題IDのリストの順に読み込む．

    埋め込みは代表の画像（sha256 が同じ画像の1枚）ごとに embeddings/part_XXXX.npy にあり，行の順は predict/part_XXXX.parquet と同じである．

    Args:
        ids (list[int]): お題IDのリスト．

    Returns:
        np.ndarray: 形状 (len(ids), 1152) の float32 の配列．
    """
    import glob

    import numpy as np

    rep = pl.scan_parquet(ANNOTATION).select("odaiId", "rep_odaiId").filter(pl.col("odaiId").is_in(pl.Series(ids).implode())).collect()
    rep_of = dict(zip(rep["odaiId"].to_list(), rep["rep_odaiId"].to_list()))
    want = set(rep_of.values())
    out = {}
    for path in sorted(glob.glob(os.path.join(ANN_DIR, "predict", "part_*.parquet"))):
        r = pl.read_parquet(path, columns=["rep_odaiId"]).with_row_index("row").filter(pl.col("rep_odaiId").is_in(list(want)))
        if r.height == 0:
            continue
        e = np.load(path.replace("predict", "embeddings").replace(".parquet", ".npy"), mmap_mode="r")
        for rid, row in zip(r["rep_odaiId"].to_list(), r["row"].to_list()):
            out[rid] = np.asarray(e[row], dtype=np.float32)
    return np.stack([out[rep_of[i]] for i in ids])
