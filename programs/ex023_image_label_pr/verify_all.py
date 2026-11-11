"""目視のラベルがある全てのお題画像（1,020枚）に，Qwen3-VL-8B の確認の質問（お題文か）を行う．

パッケージ（Bokete_Annotation_programs_v5.0）の verify.py と同じモデル・指示・設定を用いる．
パッケージでは生徒がお題文と予測した画像にだけ質問するが，ここでは生徒の選別に頼らない Qwen 単独の性能を測るため，全ての画像に質問する．

出力：outputs/ex023_image_label_pr/qwen_verify_all.csv（列 fileId, qwen_verify）

実行例:
    .venv_annotate/bin/python programs/ex023_image_label_pr/verify_all.py
"""
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, PKG_DIR, image_path, load_truth  # noqa: E402

sys.path.insert(0, PKG_DIR)
from bokete_annotation import verify  # noqa: E402


def main() -> None:
    """確認の質問を行い，回答を保存する．

    Returns:
        None
    """
    out = os.path.join(OUTPUT_DIR, "qwen_verify_all.csv")
    if os.path.exists(out):
        print(f"skip: {out}")
        return
    with open(os.path.join(PKG_DIR, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)["verify"]
    ids = load_truth()["fileId"].unique().sort().to_list()
    llm = verify.build_llm(cfg["model"], cfg["max_pixels"])
    answers, sec = verify.ask(llm, [image_path(i) for i in ids], cfg["max_pixels"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pl.DataFrame({"fileId": ids, "qwen_verify": answers}).write_csv(out)
    print(f"{len(ids)} images, {sec:.1f} sec, yes={sum(answers)}")


if __name__ == "__main__":
    main()
