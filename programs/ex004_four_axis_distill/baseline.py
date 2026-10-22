"""旧定義（ex002）で学習した最良の生徒の予測を，新定義の目視ラベルの形式の軸で評価する（参考値）．

ex002 の形式（normal/text/anaume/karuta）を新定義（normal/odaibun/anaume/karuta）に読み替え，
形式の正解率とマクロ F1 のみを求める（属性・媒体・事前知識は旧定義に対応する項目がないため評価しない）．
結果は outputs/ex004_four_axis_distill/baseline_ex002/{手法}/{seed}/metrics_format.json に保存する．

実行例:
    .venv_annotate/bin/python programs/ex004_four_axis_distill/baseline.py
"""
import glob
import json
import os
import sys

import polars as pl
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR  # noqa: E402
from data import FORMATS, PROJECT_DIR, load_eval_labels  # noqa: E402

EX002_OUT = os.path.join(PROJECT_DIR, "outputs", "ex002_odai_type_distill")
RUNS = {"teacher_8B_v2": "teacher_Qwen3-VL-8B-Instruct/v2",
        "logreg_so400m": "student_logreg/Qwen3-VL-8B-Instruct__siglip2-so400m-patch14-384__nall",
        "finetune_siglip2_base": "finetune_siglip2_base/Qwen3-VL-8B-Instruct_lr3e-05_ep5"}


def main() -> None:
    """ex002 の予測を新定義の形式で評価して保存する．

    Returns:
        None
    """
    labels = load_eval_labels().select("fileId", "pool", pl.col("format").alias("true"))
    for name, run in RUNS.items():
        for seed_dir in sorted(glob.glob(os.path.join(EX002_OUT, run, "*"))):
            p = pl.read_csv(os.path.join(seed_dir, "predictions.csv")).select(
                "fileId", pl.col("pred_format").replace({"text": "odaibun"}).alias("pred"))
            d = labels.join(p, on="fileId", maintain_order="left")
            r = d.filter(pl.col("pool").str.starts_with("random"))
            m = {"format_accuracy": accuracy_score(d["true"], d["pred"]),
                 "format_macro_f1": f1_score(d["true"], d["pred"], labels=FORMATS, average="macro"),
                 "random_format_accuracy": accuracy_score(r["true"], r["pred"])}
            out = os.path.join(OUTPUT_DIR, "baseline_ex002", name, os.path.basename(seed_dir))
            os.makedirs(out, exist_ok=True)
            json.dump(m, open(os.path.join(out, "metrics_format.json"), "w"), indent=2)
            print(name, os.path.basename(seed_dir), {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    main()
