"""report_007 に載せる実際の画像の例の図を作る．

少ない区分の評価用データ（200枚）から，目視でゲーム・CG・絵画と判定した画像を選び，
目視ラベル，教師（T_v2_J6M7），微調整した SigLIP2-B の「基本のみ・重みなし」と「追加あり・重み付き」（いずれも seed 0）の
媒体の予測を並べて示す．出力は outputs/ex005_rare_weighted_distill/figures/fig_examples_rare_{区分}.png である．

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/examples.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, plot_image_grid  # noqa: E402
from data import MEDIA  # noqa: E402
import rare_eval as R  # noqa: E402

SHORT = {"photo": "写真", "anime": "アニメ", "game": "ゲーム", "cg": "CG", "manga": "マンガ", "text_graphic": "テキスト",
         "painting": "絵画", "illustration": "イラスト"}
RUNS = {"FT 基本・重みなし": "finetune_siglip2_base/T_v2_J6M7__base__unweighted/0",
        "FT 追加・重み付き": "finetune_siglip2_base/T_v2_J6M7__base+rare__weighted/0"}


def media_str(row: np.ndarray) -> str:
    """媒体の真偽値の並びを表示用の文字列にする．

    Args:
        row (np.ndarray): 形状 (8,) の真偽値．

    Returns:
        str: 例 "写真+ゲーム"．
    """
    return "+".join(SHORT[m] for m, v in zip(MEDIA, row) if v) or "なし"


def main() -> None:
    """図を作って保存する．

    Returns:
        None
    """
    labels = R.load_rare_labels()
    ids = labels["fileId"].to_list()
    combine = R._ex005_module("combine")
    teacher = combine.build(combine.TEACHERS["T_v2_J6M7"], "rare_eval", ids).select(MEDIA).to_numpy()
    preds = {}
    for name, run in RUNS.items():
        prob = R.predict_finetune(os.path.join(OUTPUT_DIR, run), ids)["media"]
        p = prob >= 0.5
        none = np.where(~p.any(1))[0]
        p[none, prob[none].argmax(1)] = True
        preds[name] = p
    true = labels.select(MEDIA).to_numpy()
    for cls in ["game", "cg", "painting"]:
        idx = np.where(true[:, MEDIA.index(cls)])[0]
        idx = np.random.default_rng(0).choice(idx, min(8, len(idx)), replace=False)
        titles = []
        for i in idx:
            t = media_str(true[i])
            lines = [f"id={ids[i]}", f"目視: {t}", f"教師: {media_str(teacher[i])}"]
            lines += [f"{k}: {media_str(v[i])}" for k, v in preds.items()]
            titles.append("\n".join(lines))
        plot_image_grid([ids[i] for i in idx], titles, f"目視で{SHORT[cls]}とした画像（少ない区分の評価用データ）",
                        os.path.join(OUTPUT_DIR, "figures", f"fig_examples_rare_{cls}.png"), ncols=4)
    print("図を保存した")


if __name__ == "__main__":
    main()
