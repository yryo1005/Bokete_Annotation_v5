"""新しい評価用データと判断のずれの確認の画像に，ex026 と同じ指示 OCR で Qwen3-VL-32B の書き起こし（重ねた文字と余白の文字）を付ける．

後段の参考の構成（v5.3.0 の確率＋32B の書き起こし）の特徴に用いる．

出力：outputs/ex028_odai_holdout/qwen/OCR.csv（636枚），OCR_df.csv（追加の抽出の600枚．--target df）（列 rep_odaiId, ok, text_o, text_m, raw）
      と speed.json（speed_df.json）

実行例:
    .venv_annotate/bin/python programs/ex028_odai_holdout/qwen_ocr.py
    .venv_annotate/bin/python programs/ex028_odai_holdout/qwen_ocr.py --target df
"""
import argparse
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX026, OUTPUT_DIR, TARGETS, TARGETS_DF, _load  # noqa: E402

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
MAX_PIXELS = 802816
OUT = os.path.join(OUTPUT_DIR, "qwen")


def load_qwen_v6():
    """ex026 の qwen_v6 モジュールを読み込む（`from data import` のため sys.modules と sys.path を一時的に差し替える）．

    Returns:
        module: qwen_v6 モジュール．
    """
    saved = sys.modules.get("data")
    sys.modules["data"] = EX026
    ex026_dir = os.path.dirname(EX026.__file__)
    sys.path.insert(0, ex026_dir)
    try:
        return _load("ex026_qwen_for_ex028", os.path.join(ex026_dir, "qwen_v6.py"))
    finally:
        sys.path.remove(ex026_dir)
        sys.modules["data"] = saved


def main() -> None:
    """書き起こしを付けて保存する（結果があれば飛ばす）．

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["main", "df"], default="main")
    args = ap.parse_args()
    suffix = "" if args.target == "main" else "_df"
    path = os.path.join(OUT, f"OCR{suffix}.csv")
    if os.path.exists(path):
        print(f"skip: {path} がある")
        return
    os.makedirs(OUT, exist_ok=True)
    q = load_qwen_v6()
    ids = pl.read_csv(TARGETS if args.target == "main" else TARGETS_DF)["rep_odaiId"].to_list()
    llm = q.verify.build_llm(MODEL, MAX_PIXELS)
    df, sec, n_tok = q.run(llm, "OCR", ids, MAX_PIXELS)
    df.write_csv(path)
    speed = {"n": len(ids), "sec": sec, "img_per_sec": len(ids) / sec, "tokens": n_tok, "parse_fail": int((~df["ok"]).sum())}
    json.dump(speed, open(os.path.join(OUT, f"speed{suffix}.json"), "w"), ensure_ascii=False, indent=1)
    print(speed)


if __name__ == "__main__":
    main()
