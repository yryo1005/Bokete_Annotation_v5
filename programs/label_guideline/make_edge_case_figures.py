"""ラベル定義の改訂案（report_003）に用いる境界事例の図を作る．

評価用データ（640枚）のうち，目視ラベルと強い手法（Qwen3-VL-8B，微調整した SigLIP2-B）の判断が分かれた画像から，
論点ごとに代表例を選び，現行の目視ラベル・各手法の予測・改訂案でのラベルを併記して図示する．
出力は outputs/label_guideline/fig_edge_{論点}.png と edge_cases.csv である．

実行例:
    .venv_annotate/bin/python programs/label_guideline/make_edge_case_figures.py
"""
import os
import sys

import polars as pl

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex002_odai_type_distill"))
from analysis import plot_image_grid  # noqa: E402
from data import load_eval_labels  # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "label_guideline")
PRED_DIR = os.path.join(PROJECT_DIR, "outputs", "ex002_odai_type_distill")
PRED_RUNS = {"8B": "teacher_Qwen3-VL-8B-Instruct/v2/0", "FT": "finetune_siglip2_base/Qwen3-VL-8B-Instruct_lr3e-05_ep5/0"}

# 論点ごとの代表例．値は (お題ID, 改訂案での主形式, 改訂案での媒体, 改訂案での付加属性)
EDGE_CASES = {
    "R1_text_incidental": [
        (4021882, "normal", "photo", "has_text"), (7049832, "normal", "photo", "has_text"),
        (1200238, "normal", "photo", "has_text"), (7863302, "normal", "screen_ui", "has_text"),
        (4806748, "normal", "text_graphic", "has_text"), (3122377, "normal", "composite", "has_text"),
    ],
    "R2_prompt_with_picture": [
        (6675112, "text", "illustration", "prompt_text"), (3978430, "text", "composite", "prompt_text, known_work"),
        (6836548, "text", "illustration", "prompt_text, known_work"), (7712810, "text", "cg_3d", "prompt_text"),
    ],
    "R3_blank_precedence": [
        (7931870, "anaume", "text_graphic", "prompt_text, has_blank"), (7413650, "anaume", "photo", "has_blank"),
        (3138453, "anaume", "text_graphic", "prompt_text, has_blank"), (4414171, "karuta", "illustration", "has_blank"),
        (7040053, "anaume", "illustration", "has_blank"), (1843224, "anaume", "photo", "has_blank(empty bubble)"),
    ],
    "R4_drawing_medium": [
        (512794, "karuta", "illustration", "karuta card"), (3883800, "anaume", "illustration", "hand drawing"),
        (2559026, "anaume", "illustration", "cartoon, not manga"), (2333936, "anaume", "illustration", "known_work?"),
        (1994735, "anaume", "game_screen", "known_work"), (7332207, "normal", "game_screen", "known_work"),
    ],
    "R5_photo_vs_artwork": [
        (7798723, "normal", "photo", "photographed figures"), (544681, "normal", "painting", "flat artwork"),
        (5648669, "normal", "photo", "old photo"), (2897337, "normal", "composite", "photo + drawing"),
        (5083006, "normal", "composite", "anime + photo, known_person"), (4150487, "normal", "photo", "painted marking"),
    ],
}
GROUP_TITLES = {
    "R1_text_incidental": "R1: text in the picture but not a prompt -> normal + has_text",
    "R2_prompt_with_picture": "R2: a prompt sentence with a picture -> text (prompt_text)",
    "R3_blank_precedence": "R3: blanks take precedence over text; karuta over blanks",
    "R4_drawing_medium": "R4: manga / anime defined by source medium, not drawing style",
    "R5_photo_vs_artwork": "R5: photographed objects vs flat artworks vs composites",
}


def load_predictions() -> pl.DataFrame:
    """評価用ラベルと各手法の7クラスの予測を結合する．

    Returns:
        pl.DataFrame: 列 fileId, format, domain と，手法ごとの予測（列名は PRED_RUNS のキー）を持つ表．
    """
    df = load_eval_labels().select("fileId", "format", "domain")
    for name, run in PRED_RUNS.items():
        p = pl.read_csv(os.path.join(PRED_DIR, run, "predictions.csv"))
        p = p.select("fileId", pl.concat_str([pl.col("pred_format"), pl.col("pred_domain")], separator="/").alias(name))
        df = df.join(p, on="fileId", how="left")
    return df


def main() -> None:
    """論点ごとの境界事例の図と一覧の CSV を保存する．

    Returns:
        None
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_predictions()
    rows = []
    for group, cases in EDGE_CASES.items():
        ids, titles = [], []
        for fid, fmt, medium, attrs in cases:
            r = df.filter(pl.col("fileId") == fid).row(0, named=True)
            ids.append(fid)
            titles.append(f"id={fid}\nnow: {r['format']}/{r['domain']} | 8B: {r['8B']} | FT: {r['FT']}\n"
                          f"proposal: {fmt} / {medium}\n[{attrs}]")
            rows.append({"group": group, "fileId": fid, "hand_format": r["format"], "hand_domain": r["domain"],
                         "pred_8B": r["8B"], "pred_FT": r["FT"], "proposal_format": fmt, "proposal_medium": medium,
                         "proposal_attributes": attrs})
        plot_image_grid(ids, titles, GROUP_TITLES[group], os.path.join(OUT_DIR, f"fig_edge_{group}.png"), ncols=3)
    pl.DataFrame(rows).write_csv(os.path.join(OUT_DIR, "edge_cases.csv"))
    print(f"{len(rows)} 件の境界事例を {OUT_DIR} に保存した")


if __name__ == "__main__":
    main()
