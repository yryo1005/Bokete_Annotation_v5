"""report_013 の図と，最終版の構成の項目ごとの精度を作る．

- final_metrics.json：最終版の構成（compare.py の FINAL_SOURCE と FINAL_POST）の，通常の評価用データ（dev/test）の
  項目ごとの F1，少ない区分の評価用データの F1，実データの目視ラベルでの項目ごとの F1
- fig_examples.png：少量での動作確認（annotation_test）のアノテーションの例（無作為の20枚）
- fig_verify_examples.png：最終版の確認の質問（O_v11_en）でお題文から外した画像（annotation_test）
- fig_o10_wrong.png：O_v10_en がお題文ではないと答えたが，目視ではお題文であった画像の例（verify_reject_labels.txt）

実行例:
    .venv_annotate/bin/python programs/ex009_final/figures.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, evaluate, plot_image_grid  # noqa: E402
from data import ATTRIBUTES, F_CODE, MEDIA, M_CODE, check_ids, load_eval_labels, load_rare_labels, load_real_labels  # noqa: E402
import compare as C  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
TEST_DIR = os.path.join(OUTPUT_DIR, "annotation_test")
F_JA = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常"}
A_JA = {"has_blank": "空欄", "has_karuta_mark": "カルタ記号", "has_text": "文字"}
M_JA = {"photo": "写真", "anime": "アニメ", "game": "ゲーム", "manga": "マンガ", "text_graphic": "テキスト",
        "painting": "絵画", "illustration": "イラスト"}


def final_metrics() -> dict:
    """最終版の構成の項目ごとの精度を求める．

    Returns:
        dict: キー eval_dev，eval_test，rare，real の指標．
    """
    labels = load_eval_labels()
    rare = load_rare_labels()
    ids = {"eval": labels["fileId"].to_list(), "rare": rare["fileId"].to_list(), "check": check_ids()}
    info = {k: C.side_info(k, v) for k, v in ids.items()}
    P = {k: C.load_probs(C.FINAL_SOURCE, 0, k) for k in ids}
    pred = {k: C.apply_post(P[k], info[k], C.FINAL_POST)[0] for k in ids}
    out = {}
    for sp in ["dev", "test"]:
        m = labels["split"] == sp
        e = evaluate(labels.filter(m), pred["eval"].filter(m))
        out[f"eval_{sp}"] = {"overall": e["overall_macro_f1"], "robust": e["overall_robust_f1"],
                             "format": {k: v["f1"] for k, v in e["format_per_class"].items()},
                             "attr": {k: v["f1"] for k, v in e["attr_per_item"].items()},
                             "media": {k: (v["f1"], v["support"]) for k, v in e["media_per_item"].items()}}
    out["rare"] = C.TV.rare_scores(pred["rare"], rare)
    rnd, strat = load_real_labels("random"), load_real_labels("strat")
    pc = pl.concat([pl.DataFrame({"fileId": ids["check"]}), pred["check"]], how="horizontal")
    out["real"] = C.AN.check_scores(pc, rnd, strat, int(strat["weight"].sum()))
    return out


def label_text(r: dict) -> str:
    """予測の行を短い日本語の題にする．

    Args:
        r (dict): 列 format と各項目を持つ行．

    Returns:
        str: 題．
    """
    a = "・".join(A_JA[c] for c in ATTRIBUTES if r[c]) or "なし"
    m = "・".join(M_JA[c] for c in MEDIA if r[c])
    return f"{F_JA[r['format']]}｜{a}｜{m}"


def examples() -> None:
    """動作確認のアノテーションの例の図を作る．

    Returns:
        None
    """
    a = pl.read_parquet(os.path.join(TEST_DIR, "annotation.parquet"))
    s = a.sample(n=20, seed=1)
    plot_image_grid(s["odaiId"].to_list(), [label_text(r) for r in s.to_dicts()], "最終版のアノテーションの例（無作為の20枚）",
                    os.path.join(FIG, "fig_examples.png"), ncols=5)
    v = a.filter(pl.col("verify_asked") & ~pl.col("verify_answer").fill_null(True))
    plot_image_grid(v["odaiId"].to_list(), [f"{label_text(r)}（お題文の確率 {r['p_format_odaibun']:.2f}）" for r in v.to_dicts()],
                    "最終版の確認の質問（O_v11_en）でお題文から外した画像", os.path.join(FIG, "fig_verify_examples.png"), ncols=5)
    r = pl.read_csv(os.path.join(C.TV.DATASET_DIR, "real_check_v4", "verify_reject_sample.csv"))
    lab = [ln.split()[1] for ln in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_reject_labels.txt"),
                                         encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    w = r.with_columns(pl.Series("truth", lab)).filter(~pl.col("o10") & (pl.col("truth") == "O")).head(15)
    plot_image_grid(w["fileId"].to_list(), ["目視：お題文／O_v10：お題文ではない"] * len(w),
                    "O_v10_en が誤ってお題文から外した画像の例（少量での動作確認）", os.path.join(FIG, "fig_o10_wrong.png"), ncols=5)


def main() -> None:
    """図と指標を保存する．

    Returns:
        None
    """
    os.makedirs(FIG, exist_ok=True)
    json.dump(final_metrics(), open(os.path.join(FIG, "final_metrics.json"), "w"), indent=1, ensure_ascii=False)
    examples()


if __name__ == "__main__":
    main()
