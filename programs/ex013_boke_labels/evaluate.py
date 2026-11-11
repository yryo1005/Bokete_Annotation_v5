"""方法ごとの予測（outputs/ex013_boke_labels/{方法}/eval.csv）を検証データの目視のラベルと比べる．

項目ごとの指標：
- 発話の型・必要な知識：正解率とマクロ F1
- 画像中の文字への依存・性的・不適切：F1，適合率，再現率
- 制約の順守：形式がカルタ・穴埋め・お題文のボケでの正解率．違反は4件しかないため，参考とする
お題のラベルの誤り（mislabeled）のボケは，発話の型の評価から除かない（実際の運用と同じ条件にするため）．

出力：outputs/ex013_boke_labels/figures/summary.csv，per_class.json

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/evaluate.py
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import KNOWS, OUTPUT_DIR, VOICES, load_eval  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")


def scores(t: pl.DataFrame, p: pl.DataFrame) -> dict:
    """目視のラベル t と予測 p（同じ順）の指標を求める．

    Args:
        t (pl.DataFrame): 検証データの目視のラベル．
        p (pl.DataFrame): 予測（列 complies, voice, knowledge, uses_image_text, inappropriate）．

    Returns:
        dict: 指標．
    """
    r = {}
    for col, pc, labels in [("voice", "voice", VOICES), ("know", "knowledge", KNOWS)]:
        r[f"{col}_acc"] = accuracy_score(t[col], p[pc])
        r[f"{col}_f1"] = f1_score(t[col], p[pc], labels=labels, average="macro", zero_division=0)
        r[f"{col}_per_class"] = dict(zip(labels, f1_score(t[col], p[pc], labels=labels, average=None, zero_division=0).tolist()))
    for col, pc in [("uses_text", "uses_image_text"), ("inappropriate", "inappropriate")]:
        y, q = t[col].to_numpy(), p[pc].to_numpy().astype(bool)
        r[f"{col}_f1"] = f1_score(y, q, zero_division=0)
        r[f"{col}_p"] = precision_score(y, q, zero_division=0)
        r[f"{col}_r"] = recall_score(y, q, zero_division=0)
    m = (t["comp"] != "-").to_numpy()
    r["comp_acc"] = float(np.mean((t["comp"].to_numpy()[m] == "y") == p["complies"].to_numpy()[m]))
    r["comp_violation_found"] = int(np.sum((t["comp"].to_numpy()[m] == "n") & ~p["complies"].to_numpy()[m]))
    r["comp_false_alarm"] = int(np.sum((t["comp"].to_numpy()[m] == "y") & ~p["complies"].to_numpy()[m]))
    r["mean_f1"] = float(np.mean([r["voice_f1"], r["know_f1"], r["uses_text_f1"], r["inappropriate_f1"]]))
    return r


def main() -> None:
    """全ての方法を評価して保存する．

    Returns:
        None
    """
    ev = load_eval()
    rows, detail = [], {}
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "eval.csv"), recursive=True)):
        name = os.path.relpath(os.path.dirname(path), OUTPUT_DIR)
        if name.startswith("figures"):
            continue
        p = ev.select("no").join(pl.read_csv(path), on="no", how="left")
        sp_path = path.replace("eval.csv", "eval_speed.json")
        speed = json.load(open(sp_path))["per_sec"] if os.path.exists(sp_path) else None
        for split in ["dev", "test"]:
            m = ev["split"] == split
            s = scores(ev.filter(m), p.filter(m))
            detail[f"{name}|{split}"] = s
            rows.append({"method": name, "split": split, "per_sec": speed,
                         **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    os.makedirs(FIG, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(FIG, "summary.csv"))
    json.dump(detail, open(os.path.join(FIG, "per_class.json"), "w"), indent=1)
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250, float_precision=3):
        print(df.filter(pl.col("split") == "test").select(
            "method", "per_sec", "voice_acc", "voice_f1", "know_acc", "know_f1", "uses_text_f1", "inappropriate_f1",
            "inappropriate_p", "inappropriate_r", "comp_acc", "comp_violation_found", "comp_false_alarm", "mean_f1"))


if __name__ == "__main__":
    main()
