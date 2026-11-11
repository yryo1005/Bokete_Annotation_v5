"""軽い日本語 OCR（YomiToku）で対象の画像の文字を書き起こし，速さを測る．

YomiToku は文字の層を区別しないため，画像中の全ての文字を読み取り順に連結した文字列を保存する．

出力：outputs/ex026_text_attr_v6/ocr/yomitoku.csv（列 rep_odaiId, text, n_words）と speed.json

実行例:
    .venv_annotate/bin/python programs/ex026_text_attr_v6/ocr_yomitoku.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, PROJECT_DIR, TARGETS, image_path  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0"))
from bokete_annotation.data import load_rgb  # noqa: E402

OUT = os.path.join(OUTPUT_DIR, "ocr")


def read_text(analyzer, path: str) -> tuple:
    """画像1枚を OCR で読み取る．

    Args:
        analyzer (yomitoku.OCR): OCR のモデル．
        path (str): 画像のパス．

    Returns:
        tuple[str, int]: 読み取った文字列（語を空白で連結）と語の数．
    """
    arr = np.asarray(load_rgb(path))[:, :, ::-1].copy()  # YomiToku は BGR の配列を受け取る
    if min(arr.shape[:2]) < 32:
        return "", 0
    result, _ = analyzer(arr)
    words = [w.content for w in result.words]
    return " ".join(words), len(words)


def main() -> None:
    """対象の1,620枚を OCR で読み取り，結果と速さを保存する（結果があれば飛ばす）．

    Returns:
        None
    """
    path = os.path.join(OUT, "yomitoku.csv")
    if os.path.exists(path):
        print("skip: yomitoku.csv がある")
        return
    from yomitoku import OCR

    os.makedirs(OUT, exist_ok=True)
    analyzer = OCR(visualize=False, device="cuda")
    ids = pl.read_csv(TARGETS)["rep_odaiId"].to_list()
    read_text(analyzer, image_path(ids[0]))  # 初回の読み込みの時間を除くための空回し
    rows = []
    start = time.perf_counter()
    for i in tqdm(ids, desc="yomitoku", leave=False):
        text, n = read_text(analyzer, image_path(i))
        rows.append({"rep_odaiId": i, "text": text, "n_words": n})
    sec = time.perf_counter() - start
    pl.DataFrame(rows).write_csv(path)
    speed = {"n": len(ids), "sec": sec, "img_per_sec": len(ids) / sec}
    json.dump(speed, open(os.path.join(OUT, "speed.json"), "w"), indent=1)
    print(speed)


if __name__ == "__main__":
    main()
