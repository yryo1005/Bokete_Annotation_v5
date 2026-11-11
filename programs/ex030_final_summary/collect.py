"""最終版（パッケージ v5.3.0．ボケの段階は v5.2.0）のアノテーションの精度を，既存の目視のラベルと出力から求め直してまとめる．

新しい推論はしない．予測はパッケージの出力（annotation.parquet）と，ボケの分類器の検証データの予測（ex019 の verify_eval_pred.csv）を用いる．

求めるもの：
1. お題画像の形式・属性・媒体：1,020枚の目視のラベル（定義書 v4．ex022／ex023）での項目ごとの適合率・再現率・F1．
   実データ（無作為抽出160枚と層別抽出220枚の重み付きの推定．群ごとの2000回の復元抽出の95% 区間）と評価用データ（640枚．重みなし）．
2. 少ない区分（ゲーム・絵画・カルタ）：少ない区分の評価用データ800枚（ex005．重みなし）．
3. お題文あり：ex025（定義書 v5）と ex028 の追加の抽出を加えた推定（定義書 v6 の R1b）の値を出力から読む．
4. ボケ：検証データ620件の合意ラベルでの，発話の型（6区分・3区分）のクラスごとと画像外の知識の適合率・再現率・F1（調整用・最終評価用）．

出力：outputs/ex030_final_summary/（image_metrics.csv，rare_metrics.csv，odai_text_metrics.csv，boke_metrics.csv，collect.log）

実行例:
    .venv_annotate/bin/python programs/ex030_final_summary/collect.py
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex030_final_summary")
ANNOTATION = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation", "annotation.parquet")
BOKE_PRED = os.path.join(PROJECT_DIR, "outputs", "ex019_boke_full", "verify_eval_pred.csv")
ODAI_V6 = os.path.join(PROJECT_DIR, "outputs", "ex028_odai_holdout", "eval_df", "odai_combined.csv")
ODAI_V5 = os.path.join(PROJECT_DIR, "outputs", "ex025_odaibun_eval_v5", "metrics.csv")
N_BOOT = 2000
# 発話の型の3区分（ex016 の記号：C 登場人物の言葉，X 外からの言葉，A 回答）
VOICE3 = {"S": "C", "M": "C", "T": "X", "N": "X", "L": "X", "A": "A"}


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


def prf(y: np.ndarray, p: np.ndarray, w: np.ndarray = None) -> dict:
    """重み付きの適合率・再現率・F1 を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解（bool）．
        p (np.ndarray): 形状 (N,) の予測（bool）．
        w (np.ndarray | None): 形状 (N,) の重み．None なら1．

    Returns:
        dict[str, float]: precision, recall, f1．
    """
    w = np.ones(len(y)) if w is None else w
    tp, npred, ntrue = (w * (y & p)).sum(), (w * p).sum(), (w * y).sum()
    return {"precision": tp / npred if npred else np.nan, "recall": tp / ntrue if ntrue else np.nan,
            "f1": 2 * tp / (npred + ntrue) if npred + ntrue else np.nan}


def boot(y: np.ndarray, p: np.ndarray, w: np.ndarray, groups: list) -> dict:
    """群ごとの復元抽出で，適合率・再現率・F1 の95% 区間を求める．

    Args:
        y (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の予測．
        w (np.ndarray): 形状 (N,) の重み．
        groups (list[np.ndarray]): 群ごとの行番号．

    Returns:
        dict[str, float]: {指標}_lo と {指標}_hi．
    """
    rng = np.random.default_rng(0)
    vals = [prf(y[i], p[i], w[i]) for i in (np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(N_BOOT))]
    out = {}
    for k in ["precision", "recall", "f1"]:
        v = np.array([r[k] for r in vals], dtype=float)
        out[f"{k}_lo"], out[f"{k}_hi"] = float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
    return out


def image_metrics() -> pl.DataFrame:
    """お題画像の形式・属性・媒体の項目ごとの精度を求める．

    Returns:
        pl.DataFrame: 列 set, axis, category, n_true, n_pred, precision, recall, f1 と95% 区間（real のみ）を持つ表．
    """
    e23 = _load("ex023_data_for_ex030", os.path.join(PROJECT_DIR, "programs", "ex023_image_label_pr", "data.py"))
    t = e23.load_truth()
    a = pl.scan_parquet(ANNOTATION).filter(pl.col("odaiId").is_in(pl.Series(t["fileId"].to_list()).implode())).select(
        pl.col("odaiId").alias("fileId"), *e23.LABEL_COLS).collect()
    p = e23.to_binary(a)
    d = t.join(p, on="fileId", suffix="_pred")
    rows = []
    for s in ["real", "eval"]:
        x = d.filter(pl.col("set").is_in(["random", "strat"]) if s == "real" else pl.col("set") == "eval")
        w = x["w"].to_numpy() if s == "real" else np.ones(x.height)
        key = (x["set"] + ":" + x["stratum"]).to_numpy()
        groups = [np.flatnonzero(key == k) for k in np.unique(key)]
        for c in e23.CATEGORIES:
            y, q = x[c].to_numpy(), x[f"{c}_pred"].to_numpy()
            r = {"set": s, "axis": e23.AXIS[c], "category": c, "name": e23.JA[c], "n_true": int(y.sum()), "n_pred": int(q.sum()), **prf(y, q, w)}
            if s == "real":
                r.update(boot(y, q, w, groups))
            rows.append(r)
    return pl.DataFrame(rows)


def rare_metrics() -> pl.DataFrame:
    """少ない区分の評価用データでのゲーム・絵画・カルタの精度を求める．

    Returns:
        pl.DataFrame: 列 split, category, n_true, n_pred, precision, recall, f1 を持つ表．
    """
    e7 = _load("ex007_data_for_ex030", os.path.join(PROJECT_DIR, "programs", "ex007_label_v3", "data.py"))
    r = e7.load_rare_labels()
    a = pl.scan_parquet(ANNOTATION).filter(pl.col("odaiId").is_in(pl.Series(r["fileId"].to_list()).implode())).select(
        pl.col("odaiId").alias("fileId"), "game", "painting", (pl.col("format") == "karuta").alias("karuta")).collect()
    d = r.join(a, on="fileId", suffix="_pred")
    rows = []
    for sp in ["all", "dev", "test"]:
        x = d if sp == "all" else d.filter(pl.col("split") == sp)
        for c in ["game", "painting", "karuta"]:
            y = x[c].cast(pl.Utf8).str.to_lowercase().is_in(["true", "1"]).to_numpy()
            q = x[f"{c}_pred"].to_numpy()
            rows.append({"split": sp, "category": c, "n": x.height, "n_true": int(y.sum()), "n_pred": int(q.sum()), **prf(y, q)})
    return pl.DataFrame(rows)


def odai_text_metrics() -> pl.DataFrame:
    """お題文ありの精度を既存の出力から読む．

    Returns:
        pl.DataFrame: 列 definition, source, method, precision, recall, f1, lo, hi を持つ表．
    """
    v6 = pl.read_csv(ODAI_V6).filter(pl.col("set").is_in(["combined", "combined_v53"]))
    v6 = v6.select(pl.lit("v6（R1b）").alias("definition"), pl.col("set").alias("source"), "method", "precision", "recall", "f1", "lo", "hi")
    rows = [v6]
    if os.path.exists(ODAI_V5):
        v5 = pl.read_csv(ODAI_V5)
        rows.append(v5.with_columns(pl.lit("v5").alias("definition"), pl.lit("ex025").alias("source")).select(
            "definition", "source", *[c for c in v5.columns if c in ("method", "config", "precision", "recall", "f1", "lo", "hi")]))
    return pl.concat(rows, how="diagonal_relaxed")


def boke_metrics() -> pl.DataFrame:
    """ボケの発話の型と画像外の知識のクラスごとの精度を，検証データの合意ラベルで求める．

    Returns:
        pl.DataFrame: 列 split, label, class, n_true, n_pred, precision, recall, f1 を持つ表（class が macro の行はマクロ平均）．
    """
    saved = sys.modules.get("data")
    sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm"))
    d16 = _load("ex016_data_for_ex030", os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm", "data.py"))
    if saved is not None:
        sys.modules["data"] = saved
    e = d16.load_eval().select("no", "split", "voice", "voice3", "outside")
    p = pl.read_csv(BOKE_PRED).select(pl.col("id").alias("no"), pl.col("voice").alias("pv"), (pl.col("p_outside") >= 0.5).alias("po"))
    d = e.join(p, on="no").with_columns(pl.col("pv").replace_strict(VOICE3).alias("pv3"))
    assert d.height == 620, "検証データの予測が欠けている"
    rows = []
    for sp in ["dev", "test", "all"]:
        x = d if sp == "all" else d.filter(pl.col("split") == sp)
        for label, tcol, pcol, classes in [("voice6", "voice", "pv", list("SMTNLA")), ("voice3", "voice3", "pv3", ["C", "X", "A"])]:
            fs = []
            for c in classes:
                r = prf((x[tcol] == c).to_numpy(), (x[pcol] == c).to_numpy())
                rows.append({"split": sp, "label": label, "class": c, "n_true": int((x[tcol] == c).sum()), "n_pred": int((x[pcol] == c).sum()), **r})
                fs.append(r)
            rows.append({"split": sp, "label": label, "class": "macro", "n_true": x.height, "n_pred": x.height,
                         **{k: float(np.nanmean([f[k] for f in fs])) for k in ["precision", "recall", "f1"]}})
        y, q = x["outside"].to_numpy(), x["po"].to_numpy()
        rows.append({"split": sp, "label": "outside", "class": "yes", "n_true": int(y.sum()), "n_pred": int(q.sum()), **prf(y, q)})
    return pl.DataFrame(rows)


def main() -> None:
    """精度を求めて保存する．

    Returns:
        None
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(80)
    for name, fn in [("image_metrics", image_metrics), ("rare_metrics", rare_metrics), ("odai_text_metrics", odai_text_metrics),
                     ("boke_metrics", boke_metrics)]:
        t = fn()
        t.write_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"))
        print(name)
        print(t.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
