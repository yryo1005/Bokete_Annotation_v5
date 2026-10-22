"""order_009：教師 T_v8 の教師ラベルの作成と，生徒の少ない区分の評価用データ（第1版と第2版，計800枚）での評価．

1. --make_labels：教師 T_v8（形式 F_v2，属性 J_v6，媒体 M_v6，絵画と CG の独立の質問 P_v8_en（CG は媒体の質問が
   CG かイラストの場合のみ））で，基本の4万枚と少ない区分の追加分（rare_r00〜r04）の教師ラベルを作る．
2. --evaluate：微調整した生徒と MLP の生徒を，少ない区分の評価用データの dev/test で評価する．
   さらに第2版の層別の抽出を用いて，実データ（約29.5万枚）での適合率と再現率を推定する．区分 c について，
   実データを蒸留モデル（ex005 の seed 1）の確率 p_c で3つの区画（0.5以上，0.05以上0.5未満，0.05未満）に分け，
   区画ごとに抽出した画像（それぞれ c_pos，c_border，random の層のうち p_c が0.05未満のもの）の正例率と予測の率に
   区画の画像数を掛けて合計する．

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/rare_eval_v8.py --make_labels
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/rare_eval_v8.py --evaluate
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combine_v8 as C  # noqa: E402
from common import OUTPUT_DIR  # noqa: E402
from data import DATASET_DIR, FORMATS, MEDIA  # noqa: E402

TEACHER = "T_v8"
CONFIG = C.TEACHERS_V8["PCen_cgand"]
RARE_SETS = [f"rare_r{i:02d}" for i in range(5)]


def make_labels() -> None:
    """教師 T_v8 の教師ラベル（基本の4万枚と追加分）を保存する．

    Returns:
        None
    """
    from teacher import load_ids

    d = os.path.join(DATASET_DIR, "teacher_labels")
    ids = load_ids("pool")
    pl.concat([pl.DataFrame({"fileId": ids}), C.build_v8(CONFIG, "pool", ids)], how="horizontal").write_csv(
        os.path.join(d, f"{TEACHER}_pool.csv"))
    parts = []
    for s in RARE_SETS:
        sids = load_ids(s)
        parts.append(pl.concat([pl.DataFrame({"fileId": sids}), C.build_v8(CONFIG, s, sids)], how="horizontal"))
    rare = pl.concat(parts)
    rare.write_csv(os.path.join(d, f"{TEACHER}_rare.csv"))
    for name, df in [("pool", pl.read_csv(os.path.join(d, f"{TEACHER}_pool.csv"))), ("rare", rare)]:
        print(name, len(df), {c: int(df[c].sum()) for c in ["game", "cg", "painting"]}, int((df["format"] == "karuta").sum()))


def _module(name: str):
    """ex005 のモジュールをファイルから読み込む．

    Args:
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    import importlib.util

    if f"ex005_{name}" in sys.modules:
        return sys.modules[f"ex005_{name}"]
    spec = importlib.util.spec_from_file_location(f"ex005_{name}", os.path.join(os.path.dirname(__file__), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ex005_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def probs_to_frame(probs: dict) -> pl.DataFrame:
    """確率を形式と媒体の予測の表にする．

    Args:
        probs (dict): 出力層ごとの確率．

    Returns:
        pl.DataFrame: 列 format と媒体の各項目を持つ表．
    """
    S = _module("students")
    return S.to_frame(probs).select(["format"] + MEDIA)


def population_estimates(pred: pl.DataFrame, rare: pl.DataFrame) -> dict:
    """第2版の層別の抽出から，実データでの区分ごとの適合率と再現率を推定する．

    Args:
        pred (pl.DataFrame): 少ない区分の評価用データ全体（800枚）の予測（行の順は rare と同じ）．
        rare (pl.DataFrame): combine_v8.load_rare_eval の戻り値．

    Returns:
        dict: 区分ごとの推定（precision，recall，est_positives）．
    """
    ann = pl.read_parquet(os.path.join(DATASET_DIR, "real_annotation.parquet"))
    v2 = pl.concat([rare, pred.rename({c: f"pred_{c}" for c in pred.columns})], how="horizontal").filter(
        pl.col("set") == "rare_eval_v2").join(ann, on="fileId")
    out = {}
    for c in C.RARE_CLASSES:
        pcol = "p_format_karuta" if c == "karuta" else f"p_{c}"
        p_all = ann[pcol].to_numpy()
        cells = {"pos": (p_all >= 0.5).sum(), "border": ((p_all >= 0.05) & (p_all < 0.5)).sum(), "low": (p_all < 0.05).sum()}
        samples = {"pos": v2.filter(pl.col("stratum") == f"rare_eval_v2:{c}_pos"),
                   "border": v2.filter(pl.col("stratum") == f"rare_eval_v2:{c}_border"),
                   "low": v2.filter((pl.col("stratum") == "rare_eval_v2:random") & (pl.col(pcol) < 0.05))}
        tp = pos = pp = 0.0
        for cell, s in samples.items():
            y = s["karuta" if c == "karuta" else c].to_numpy()
            q = (s["pred_format"] == "karuta").to_numpy() if c == "karuta" else s[f"pred_{c}"].to_numpy().astype(bool)
            if len(s) == 0:
                continue
            n = cells[cell]
            tp += n * np.mean(y & q)
            pos += n * np.mean(y)
            pp += n * np.mean(q)
        out[c] = {"precision": float(tp / pp) if pp else 0.0, "recall": float(tp / pos) if pos else 0.0,
                  "est_positives": float(pos), "est_rate": float(pos / len(ann))}
    return out


def evaluate_run(run_dir: str, rare: pl.DataFrame, X_rare: np.ndarray) -> dict:
    """1つの生徒を少ない区分の評価用データで評価する（結果を run_dir/metrics_rare_v8.json に保存する）．

    Args:
        run_dir (str): 生徒の結果のディレクトリ．
        rare (pl.DataFrame): combine_v8.load_rare_eval の戻り値．
        X_rare (np.ndarray): rare の画像の SigLIP2-so400m の埋め込み（MLP 用）．

    Returns:
        dict: rare_dev，rare_test，population の指標．
    """
    path = os.path.join(run_dir, "metrics_rare_v8.json")
    if os.path.exists(path):
        return json.load(open(path))
    R = _module("rare_eval")
    if "finetune" in run_dir:
        probs = R.predict_finetune(run_dir, rare["fileId"].to_list())
    else:
        probs = R.predict_mlp(run_dir, X_rare)
        if "format" not in probs:
            raise ValueError("MLP の予測に形式がない")
    pred = probs_to_frame(probs)
    m = {}
    for sp in ["dev", "test"]:
        mask = pl.Series((rare["split"] == sp).to_numpy())
        m[f"rare_{sp}"] = C.rare_scores(pred.filter(mask), rare.filter(mask))
    m["population"] = population_estimates(pred, rare)
    json.dump(m, open(path, "w"), indent=2)
    return m


def evaluate_teacher_population(rare: pl.DataFrame) -> dict:
    """教師（base と T_v8）の実データでの推定を求める．

    Args:
        rare (pl.DataFrame): combine_v8.load_rare_eval の戻り値．

    Returns:
        dict: 教師の名前をキー，population_estimates の戻り値を値とする辞書．
    """
    out = {}
    for name, cfg in [("T_v2_JM", C.TEACHERS_V8["base"]), ("T_v8", CONFIG)]:
        parts = [C.build_v8(cfg, s, rare.filter(pl.col("set") == s)["fileId"].to_list()).select(["format"] + MEDIA)
                 for s in ["rare_eval", "rare_eval_v2"]]
        out[name] = population_estimates(pl.concat(parts), rare)
    return out


def main() -> None:
    """コマンドライン引数に従ってラベルの作成と評価を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_labels", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--patterns", nargs="+", default=["finetune_siglip2_base/*/*", "student_mlp/T_v8__*/*"])
    args = parser.parse_args()
    if args.make_labels:
        make_labels()
    if not args.evaluate:
        return
    rare = C.load_rare_eval()
    cand = pl.read_csv(os.path.join(DATASET_DIR, "candidates", "cand.csv"))["fileId"].to_list()
    index = {f: i for i, f in enumerate(cand)}
    X = np.load(os.path.join(DATASET_DIR, "embeddings", "cand_so400m.npy"), mmap_mode="r")
    X_rare = np.asarray(X[[index[f] for f in rare["fileId"]]], dtype=np.float32)
    rows = []
    for pat in args.patterns:
        for run_dir in sorted(glob.glob(os.path.join(OUTPUT_DIR, pat))):
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                continue
            m = evaluate_run(run_dir, rare, X_rare)
            rows.append({"method": run_dir.split(os.sep)[-3], "setting": run_dir.split(os.sep)[-2],
                         "seed": run_dir.split(os.sep)[-1], "rare_dev": m["rare_dev"]["macro_f1"],
                         "rare_test": m["rare_test"]["macro_f1"],
                         **{f"test_{c}_f1": m["rare_test"][c]["f1"] for c in C.RARE_CLASSES},
                         **{f"pop_{c}_precision": m["population"][c]["precision"] for c in C.RARE_CLASSES},
                         **{f"pop_{c}_recall": m["population"][c]["recall"] for c in C.RARE_CLASSES}})
    df = pl.DataFrame(rows)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    df.write_csv(os.path.join(OUTPUT_DIR, "figures", "rare_eval_v8_students.csv"))
    json.dump(evaluate_teacher_population(rare), open(os.path.join(OUTPUT_DIR, "figures", "rare_eval_v8_teachers_population.json"),
                                                      "w"), indent=2)
    print(df.group_by("method", "setting").agg(pl.len(), pl.col("rare_dev").mean(), pl.col("rare_test").mean())
          .sort("rare_test", descending=True))


if __name__ == "__main__":
    main()
