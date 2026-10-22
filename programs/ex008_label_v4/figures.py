"""report_012 の実例の図を作る．

- fig_odaibun_verify.png：形式の部品 F_v2 がお題文とし，確認の質問 O_v10_en が投稿者の問いではないと答えた画像
  （実データの目視ラベルの画像と評価用データ）．題に目視の形式を示す．
- fig_text_v4.png：定義書 v4 で文字ありに変えた画像（ロゴ，透かし，衣服の文字など）．

実行例:
    .venv_annotate/bin/python programs/ex008_label_v4/figures.py
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, plot_image_grid  # noqa: E402
from data import load_eval_labels, load_real_labels, text_relabel  # noqa: E402
from teacher_v4 import _comp, load_ids  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
F_NAME = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常"}


def verify_examples(n: int = 20) -> None:
    """確認の質問でお題文から外した画像の例を保存する．

    Args:
        n (int): 表示する画像の数．

    Returns:
        None
    """
    truth = pl.concat([load_real_labels("random").select("fileId", "format"),
                       load_real_labels("strat").select("fileId", "format"),
                       load_eval_labels().select("fileId", "format")])
    rows = []
    for set_name in ["real_check_v4", "eval"]:
        ids = load_ids(set_name)
        f = _comp("F_v2", set_name, ids)["format"].to_list()
        o = _comp("O_v10_en", set_name, ids)["is_poster_prompt"].to_list()
        rows += [(i, ff, oo) for i, ff, oo in zip(ids, f, o) if ff in ("odaibun", "text") and oo is False]
    df = pl.DataFrame({"fileId": [r[0] for r in rows]}).join(truth.unique("fileId"), on="fileId", how="left")
    df = pl.concat([df.filter(pl.col("format") != "odaibun").head(n - 4), df.filter(pl.col("format") == "odaibun").head(4)])
    titles = [f"{fid}：目視 {F_NAME[fm]}" + ("（確認で誤って除外）" if fm == "odaibun" else "") for fid, fm in df.iter_rows()]
    plot_image_grid(df["fileId"].to_list(), titles, "F_v2 がお題文とし，確認の質問で通常に戻した画像",
                    os.path.join(FIG, "fig_odaibun_verify.png"), ncols=5)
    print("確認で外した画像:", len(rows), "うち目視でお題文:", int((truth.join(pl.DataFrame({"fileId": [r[0] for r in rows]}),
                                                                         on="fileId")["format"] == "odaibun").sum()))


def text_examples(n: int = 15) -> None:
    """定義書 v4 で文字ありに変えた画像の例を保存する．

    Args:
        n (int): 表示する画像の数．

    Returns:
        None
    """
    ids = list(text_relabel())[:n]
    plot_image_grid(ids, [str(i) for i in ids], "定義書 v4 で文字ありに変えた画像の例（評価用データ）",
                    os.path.join(FIG, "fig_text_v4.png"), ncols=5)


def main() -> None:
    """図を作る．

    Returns:
        None
    """
    os.makedirs(FIG, exist_ok=True)
    verify_examples()
    text_examples()


if __name__ == "__main__":
    main()
