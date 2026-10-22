"""形式を2軸用の短いプロンプト v2（ex002）の判定から，属性・媒体・事前知識を v5 と規則の結果から取る組み合わせ教師を作る．

v2 の形式（normal/text/anaume/karuta）は新定義（normal/odaibun/anaume/karuta）に読み替える．
属性は形式と矛盾しないように整える（カルタ記号ありは形式がカルタの場合のみ，空欄あり・お題文ありは形式から補う）．
カルタはボケの書き出しで照合する（rules.py と同じ規則）．

- 評価：outputs/ex004_four_axis_distill/teacher_hybrid/v2format_v5rules/0/
- ラベル：datasets/ex004_four_axis_distill/teacher_labels/hybrid_v2format_v5rules.csv

実行例:
    .venv_annotate/bin/python programs/ex004_four_axis_distill/hybrid.py
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, save_run  # noqa: E402
from data import ATTRIBUTES, DATASET_DIR, KNOWLEDGE, MEDIA, PROJECT_DIR, load_eval_labels  # noqa: E402
from rules import MIN_KANA, VETO_RATE, boke_kana_stats, karuta_rate  # noqa: E402

EX002 = os.path.join(PROJECT_DIR, "outputs", "ex002_odai_type_distill", "teacher_Qwen3-VL-8B-Instruct", "v2", "0",
                     "predictions.csv")
EX002_POOL = os.path.join(PROJECT_DIR, "datasets", "ex002_odai_type_distill", "teacher_labels",
                          "Qwen3-VL-8B-Instruct.csv")


def combine(fmt: pl.Series, rest: pl.DataFrame, ids: list) -> pl.DataFrame:
    """v2 の形式と v5 の他の軸を組み合わせ，属性を形式と整合させる．

    Args:
        fmt (pl.Series): v2 の形式（旧定義の値）．
        rest (pl.DataFrame): v5 と規則の予測（列 karuta_char と各項目）．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 format と各項目を持つ表．
    """
    df = rest.with_columns(fmt.replace({"text": "odaibun"}).alias("format"))
    stats = boke_kana_stats(ids)
    # カルタは v5 のひらがなでボケと照合する（ひらがながない場合は照合しない）
    rate = [karuta_rate(c or "", n or 0, ch or "") if ch else float("nan") for c, n, ch in
            zip(stats["kana_counts"], stats["n_kana"], df["karuta_char"].fill_null(""))]
    veto = (pl.col("format") == "karuta") & pl.Series(stats["n_kana"].fill_null(0) >= MIN_KANA) & \
        pl.Series(rate, dtype=pl.Float64).lt(VETO_RATE)
    df = df.with_columns(pl.when(veto).then(pl.when(pl.col("has_blank")).then(pl.lit("anaume"))
                                            .when(pl.col("has_prompt_text")).then(pl.lit("odaibun"))
                                            .otherwise(pl.lit("normal"))).otherwise(pl.col("format")).alias("format"))
    return df.with_columns((pl.col("format") == "karuta").alias("has_karuta_mark"),
                           (pl.col("has_blank") | (pl.col("format") == "anaume")).alias("has_blank"),
                           (pl.col("has_prompt_text") | (pl.col("format") == "odaibun")).alias("has_prompt_text"))


OUT_COLS = ["format", "karuta_char"] + ATTRIBUTES + MEDIA + KNOWLEDGE


def main() -> None:
    """評価用データで組み合わせ教師を評価し，教師ラベル用の画像のラベルを作る．

    Returns:
        None
    """
    labels = load_eval_labels()
    v5 = pl.read_csv(os.path.join(OUTPUT_DIR, "teacher_Qwen3-VL-8B-Instruct", "v5_rule_wu", "0", "predictions.csv"))
    v5 = v5.select([c for c in v5.columns if c.startswith("pred_")]).rename(lambda c: c[5:])
    v2 = labels.select("fileId").join(pl.read_csv(EX002).select("fileId", "pred_format"), on="fileId",
                                      maintain_order="left")
    out = combine(v2["pred_format"], v5.drop("format"), labels["fileId"].to_list())
    save_run(os.path.join(OUTPUT_DIR, "teacher_hybrid", "v2format_v5rules", "0"), labels,
             out.select(OUT_COLS), {"method": "teacher_hybrid", "format_from": "ex002 prompt v2",
                                      "others_from": "prompt v5 + rules (work_unknown)", "boke_karuta_veto": True})
    pool_v5 = os.path.join(DATASET_DIR, "teacher_labels", "Qwen3-VL-8B-Instruct_v5_rules.csv")
    if os.path.exists(pool_v5):
        p5 = pl.read_csv(pool_v5)
        p2 = p5.select("fileId").join(pl.read_csv(EX002_POOL), on="fileId", maintain_order="left")
        pool = combine(p2["format"], p5.drop("format", "fileId", "boke_karuta_rate"), p5["fileId"].to_list())
        pl.concat([p5.select("fileId"), pool.select(OUT_COLS)], how="horizontal").write_csv(
            os.path.join(DATASET_DIR, "teacher_labels", "hybrid_v2format_v5rules.csv"))
        print("教師ラベル用の画像のラベルを保存した:", pool["format"].value_counts(sort=True).to_dicts())


if __name__ == "__main__":
    main()
