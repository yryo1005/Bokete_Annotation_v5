"""report_006 に載せる実際の画像の例の図を作る．

1. fig_examples_{形式}.png：形式ごとに無作為に選んだ画像と，目視ラベル・最良の生徒（MLP，SigLIP2-so400m，組み合わせ教師）・
   微調整した SigLIP2-B（組み合わせ教師）の4軸の予測．
2. fig_karuta_fp_v4.png：プロンプト v4 がカルタ記号と誤った穴埋めの画像（v5 と規則の改訂の根拠）．
3. fig_prompt_text_disagree.png：「お題文あり」で目視ラベルと最良の生徒が食い違った画像．

実行例:
    .venv_annotate/bin/python programs/ex004_four_axis_distill/examples.py
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, plot_image_grid  # noqa: E402
from data import ATTRIBUTES, FORMATS, KNOWLEDGE, MEDIA, load_eval_labels  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
RUNS = {"MLP": "student_mlp/hybrid_v2format_v5rules__siglip2-so400m-patch14-384/0",
        "FT": "finetune_siglip2_base/hybrid_v2format_v5rules_lr3e-05_ep4/0"}
SHORT = {"has_blank": "空", "has_prompt_text": "題", "has_karuta_mark": "札", "has_other_text": "字", "composite": "合",
         "photo": "写真", "anime": "アニメ", "game": "ゲーム", "cg": "CG", "manga": "マンガ", "text_graphic": "テキスト",
         "painting": "絵画", "illustration": "イラスト", "known_person": "著名人", "known_work": "作品",
         "work_unknown": "作品不明"}
FMT_JA = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常"}


def describe(r: dict, prefix: str = "") -> str:
    """1行の予測（またはラベル）を短い文字列にする．

    Args:
        r (dict): 列 format と各項目を持つ行．
        prefix (str): 列名の接頭辞（予測の場合は "pred_"）．

    Returns:
        str: 例 "穴埋め [空字] 写真 / 著名人"．
    """
    attrs = "".join(SHORT[a] for a in ATTRIBUTES if r[prefix + a])
    media = "+".join(SHORT[m] for m in MEDIA if r[prefix + m])
    know = "+".join(SHORT[k] for k in KNOWLEDGE if r[prefix + k]) or "-"
    return f"{FMT_JA[r[prefix + 'format']]} [{attrs}] {media} / {know}"


def main() -> None:
    """図を作って保存する．

    Returns:
        None
    """
    os.makedirs(FIG, exist_ok=True)
    labels = load_eval_labels()
    preds = {k: pl.read_csv(os.path.join(OUTPUT_DIR, v, "predictions.csv")) for k, v in RUNS.items()}
    for fmt in FORMATS:
        idx = labels.with_row_index("i").filter(pl.col("format") == fmt).sample(n=8, seed=1)["i"].to_list()
        titles = []
        for i in idx:
            t = describe(labels.row(i, named=True))
            lines = [f"id={labels['fileId'][i]}", f"目視: {t}"]
            for k, p in preds.items():
                d = describe(p.row(i, named=True), "pred_")
                lines.append(f"{k}: {d}" + ("" if d == t else " ×"))
            titles.append("\n".join(lines))
        plot_image_grid([labels["fileId"][i] for i in idx], titles, f"目視の形式が{FMT_JA[fmt]}の画像",
                        os.path.join(FIG, f"fig_examples_{fmt}.png"), ncols=4)
    v4 = pl.read_csv(os.path.join(OUTPUT_DIR, "teacher_Qwen3-VL-8B-Instruct", "v4", "0", "predictions.csv"))
    fp = pl.DataFrame({"fileId": labels["fileId"], "t": labels["format"], "mark": v4["pred_has_karuta_mark"],
                       "ch": v4["pred_karuta_char"]}).filter(pl.col("mark") & (pl.col("t") == "anaume")).head(8)
    plot_image_grid(fp["fileId"].to_list(), [f"id={i} 記号の文字と出力: 「{c}」" for i, c in zip(fp["fileId"], fp["ch"])],
                    "プロンプト v4 がカルタ記号ありと誤った穴埋めの画像", os.path.join(FIG, "fig_karuta_fp_v4.png"), ncols=4)
    p = preds["MLP"]
    dis = labels.with_row_index("i").filter(
        pl.Series(p["pred_has_prompt_text"].to_list()) != pl.col("has_prompt_text")).sample(n=8, seed=0)
    titles = [f"id={r['fileId']} 形式（目視）: {FMT_JA[r['format']]}\nお題文あり 目視: {'あり' if r['has_prompt_text'] else 'なし'} / "
              f"MLP: {'あり' if p['pred_has_prompt_text'][r['i']] else 'なし'}" for r in dis.iter_rows(named=True)]
    plot_image_grid(dis["fileId"].to_list(), titles, "「お題文あり」で目視ラベルと最良の生徒が食い違った画像",
                    os.path.join(FIG, "fig_prompt_text_disagree.png"), ncols=4)
    print("図を保存した:", FIG)


if __name__ == "__main__":
    main()
