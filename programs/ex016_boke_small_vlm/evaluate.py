"""ex016 の全ての予測（outputs/ex016_boke_small_vlm/**/eval.csv）を，検証データ620件の目視のラベルと比べる．

正解は2通りとする．
- orig：ex015 の目視のラベル（発話の型6区分，定義書 第3版の画像外の知識）．
- cons：合意ラベル（consensus_labels.txt）．620件全てを元のラベルを見ずに付け直し，元のラベルとの不一致を裁定したもの．
指標：発話の型6区分のマクロ F1（voice6_f1）と正解率，3区分のマクロ F1（voice3_f1），画像外の知識の F1・適合率・再現率．
分割：調整用（dev），最終評価用（test），全体（all），最初に付け直した160件（c160）．
ex015 の主な方法（教師 32B one_kt_cap，文の埋め込みの生徒）も比較のため同じ指標で評価する．

出力：outputs/ex016_boke_small_vlm/figures/summary.csv

実行例:
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/evaluate.py
"""
import glob
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX15_OUTPUT, OUTPUT_DIR, THIS_DIR, VOICE3, VOICES, load_eval  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
EX15_METHODS = {"ex015/teacher_q32_one_kt_cap": "teacher_q32/one_kt_cap", "ex015/student_text": "student/q32-one_kt_cap_extra__ruri_fmt"}


def load_gold() -> pl.DataFrame:
    """検証データと2通りの正解を読み込む．

    Returns:
        pl.DataFrame: 列 no, split, format, voice_orig, outside_orig, voice_cons, outside_cons, c160 を持つ表．
    """
    ev = load_eval().select("no", "split", "format", pl.col("voice").alias("voice_orig"), pl.col("outside").alias("outside_orig"))
    read = lambda f: [ln.split()[:3] for ln in open(os.path.join(THIS_DIR, f), encoding="utf-8")  # noqa: E731
                      if ln.strip() and not ln.startswith("#")]
    rows = read("consensus_labels.txt")
    cons = pl.DataFrame({"no": [int(r[0]) for r in rows], "voice_cons": [r[1] for r in rows], "outside_cons": [r[2] == "1" for r in rows]})
    c160 = {int(r[0]) for r in read("consensus160_labels.txt")}
    ev = ev.join(cons, on="no", how="left")
    assert ev["voice_cons"].null_count() == 0
    return ev.with_columns(pl.col("no").is_in(list(c160)).alias("c160"))


def scores(y_voice: list, q_voice: list, y_out: np.ndarray, q_out: np.ndarray) -> dict:
    """指標を求める．

    Args:
        y_voice (list[str]): 正解の発話の型．
        q_voice (list[str]): 予測の発話の型．
        y_out (np.ndarray): 正解の画像外の知識（bool）．
        q_out (np.ndarray): 予測の画像外の知識（bool）．

    Returns:
        dict: 指標の辞書．
    """
    y3, q3 = [VOICE3[v] for v in y_voice], [VOICE3[v] for v in q_voice]
    return {"voice6_f1": f1_score(y_voice, q_voice, labels=VOICES, average="macro", zero_division=0),
            "voice6_acc": accuracy_score(y_voice, q_voice),
            "voice3_f1": f1_score(y3, q3, labels=["C", "X", "A"], average="macro", zero_division=0),
            "outside_f1": f1_score(y_out, q_out, zero_division=0), "outside_prec": precision_score(y_out, q_out, zero_division=0),
            "outside_rec": recall_score(y_out, q_out, zero_division=0)}


def collect(gold: pl.DataFrame) -> dict:
    """全ての予測を集める．

    Args:
        gold (pl.DataFrame): load_gold の結果．

    Returns:
        dict[str, pl.DataFrame]: 方法名 → 列 no, voice, outside の表（gold と同じ順）．
    """
    paths = {name: os.path.join(EX15_OUTPUT, rel) for name, rel in EX15_METHODS.items()}
    preds = {}
    for name, d in paths.items():
        for p in sorted(glob.glob(os.path.join(d, "**", "eval.csv"), recursive=True)):
            preds[name + os.path.dirname(p)[len(d):]] = p
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "eval.csv"), recursive=True)):
        preds[os.path.relpath(os.path.dirname(p), OUTPUT_DIR)] = p
    out = {}
    for name, p in preds.items():
        d = pl.read_csv(p, schema_overrides={"knowledge": pl.String})
        out[name] = gold.select("no").join(d.select("no", "voice", "outside"), on="no", how="left", maintain_order="left")
    return out


def main() -> None:
    """全ての方法を評価して保存する．

    Returns:
        None
    """
    gold = load_gold()
    masks = {"dev": (gold["split"] == "dev").to_numpy(), "test": (gold["split"] == "test").to_numpy(),
             "all": np.ones(gold.height, bool), "c160": gold["c160"].to_numpy()}
    rows = []
    for name, p in collect(gold).items():
        pv = p["voice"].fill_null("S").to_list()
        po = p["outside"].fill_null(False).to_numpy().astype(bool)
        for g in ["orig", "cons"]:
            yv, yo = gold[f"voice_{g}"].to_list(), gold[f"outside_{g}"].to_numpy()
            for split, m in masks.items():
                s = scores([v for v, k in zip(yv, m) if k], [v for v, k in zip(pv, m) if k], yo[m], po[m])
                rows.append({"method": name, "gold": g, "split": split, **s})
    os.makedirs(FIG, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(FIG, "summary.csv"))
    g = df.filter(pl.col("gold") == "cons").with_columns(pl.col("method").str.replace(r"/\d+$", "").alias("m")).group_by("m", "split").agg(
        [pl.col(c).mean() for c in ["voice6_f1", "voice3_f1", "outside_f1"]]).pivot(on="split", index="m",
                                                                                  values=["voice6_f1", "voice3_f1", "outside_f1"])
    g = g.select(["m"] + [f"{c}_{s}" for c in ["voice6_f1", "voice3_f1", "outside_f1"] for s in ["dev", "test"]])
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=50, tbl_width_chars=250, float_precision=3):
        print(g.sort("voice6_f1_dev", descending=True))


if __name__ == "__main__":
    main()
