"""定義書 v4 の目視ラベルを付けるための画像を抽出し，一覧画像を作る．

1. --strat：ex007 の最良の生徒（T_v8m・base+rare・重みなし，seed 0）の実データへの予測に基づき，各画像を予測された
   クラスのうち最も少ないものの層に割り当て（優先順は STRATA），各層から N_PER_STRATUM 枚ずつ無作為に抽出する．
   各クラスに予測された画像が少なくとも N_PER_STRATUM 枚含まれ，クラスの分布がおおむね一様になる．
   各画像には抽出確率の逆数（層の画像数 / 抽出数）を重みとして保存し，実際の分布での精度の推定に用いる．
   一覧は層が分からないように順番を無作為に並べ替え，予測は表示しない．
2. --text_recheck：定義書 v4 の文字あり（ロゴ等を含む）に合わせて既存の目視ラベルを見直すための一覧を作る．
   - 実データの無作為抽出160枚のうち，文字なしとした全ての画像
   - 評価用データ640枚のうち，文字なしとし，教師（J_v6）または ex007 の生徒が文字ありと予測した画像
3. また，教師の部品を実データの目視ラベルの画像に適用するため，両方の画像のお題IDを
   datasets/ex005_rare_weighted_distill/candidates/real_check_v4.csv に保存する．

実行例:
    .venv_annotate/bin/python programs/ex008_label_v4/sample.py --strat --text_recheck
"""
import argparse
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import plot_image_grid  # noqa: E402
from data import CHECK_DIR, DATASET_DIR, EX7_DIR, PROJECT_DIR, ex7, load_rare_labels  # noqa: E402

N_PER_STRATUM = 20
BEST_RUN = os.path.join(PROJECT_DIR, "outputs", "ex007_label_v3", "student_mlp", "T_v8m__base+rare__unweighted", "0")
# 層の優先順（実データで予測された割合の少ない順）．画像は条件を満たす最初の層に割り当てる
STRATA = [("painting", pl.col("painting")), ("karuta", pl.col("format") == "karuta"), ("game", pl.col("game")),
          ("anime", pl.col("anime")), ("odaibun", pl.col("format") == "odaibun"), ("illustration", pl.col("illustration")),
          ("text_graphic", pl.col("text_graphic")), ("manga", pl.col("manga")), ("anaume", pl.col("format") == "anaume"),
          ("text", pl.col("has_text")), ("photo", pl.lit(True))]


def random160_ids() -> list:
    """order_011 の無作為抽出160枚のお題IDを読み込む．

    Returns:
        list[int]: お題IDのリスト．
    """
    return pl.read_csv(os.path.join(DATASET_DIR, "real_check_v3", "sample.csv"))["fileId"].to_list()


def assign_strata(ann: pl.DataFrame) -> pl.DataFrame:
    """予測に基づき各画像を層に割り当てる．

    Args:
        ann (pl.DataFrame): 実データへのアノテーション（列 fileId, format と各項目）．

    Returns:
        pl.DataFrame: 列 stratum を加えた表．
    """
    expr = pl.when(STRATA[0][1]).then(pl.lit(STRATA[0][0]))
    for name, cond in STRATA[1:]:
        expr = expr.when(cond).then(pl.lit(name))
    return ann.with_columns(expr.alias("stratum"))


def make_strat(seed: int = 0) -> list:
    """層別抽出を行い，一覧画像と抽出の表を保存する．

    Args:
        seed (int): 乱数のシード．

    Returns:
        list[int]: 抽出したお題IDのリスト（一覧の順）．
    """
    ann = pl.read_parquet(os.path.join(BEST_RUN, "real_annotation_v3.parquet"))
    used = set(ex7.load_eval_labels()["fileId"].to_list()) | set(load_rare_labels()["fileId"].to_list()) | set(random160_ids())
    ann = assign_strata(ann.filter(~pl.col("fileId").is_in(list(used))))
    parts = []
    for name, _ in STRATA:
        g = ann.filter(pl.col("stratum") == name)
        n = min(N_PER_STRATUM, len(g))
        parts.append(g.sample(n=n, seed=seed).select("fileId", "stratum").with_columns(
            pl.lit(len(g) / n).alias("weight"), pl.lit(len(g)).alias("n_stratum")))
    s = pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=seed)
    os.makedirs(CHECK_DIR, exist_ok=True)
    s.write_csv(os.path.join(CHECK_DIR, "strat_sample.csv"))
    ids = s["fileId"].to_list()
    for k in range(0, len(ids), 20):
        part = ids[k:k + 20]
        plot_image_grid(part, [f"{k + i:03d}:{f}" for i, f in enumerate(part)], f"層別抽出 {k // 20}",
                        os.path.join(CHECK_DIR, f"strat_{k // 20:02d}.png"), ncols=5)
    print(s.group_by("stratum").agg(pl.len(), pl.col("n_stratum").first()).sort("n_stratum"))
    return ids


def make_text_recheck() -> None:
    """文字ありの見直し用の一覧を作る（大きめの画像で12枚ずつ）．

    Returns:
        None
    """
    rnd = pl.read_csv(os.path.join(DATASET_DIR, "real_check_v3", "sample.csv"))["fileId"].to_list()
    lines = [ln.split() for ln in open(os.path.join(EX7_DIR, "real_check_v3.txt"), encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
    real_ids = [rnd[int(k)] for k, _, a, _ in lines if "t" not in a]
    ev = ex7.load_eval_labels()
    j6 = ev.select("fileId").join(pl.read_csv(os.path.join(PROJECT_DIR, "outputs", "ex005_rare_weighted_distill",
                                                          "teacher_components", "J_v6", "eval.csv")),
                                  on="fileId", how="left", maintain_order="left")
    teacher_text = (j6["has_prompt_text"].fill_null(False) | j6["has_other_text"].fill_null(False)).to_numpy()
    st = pl.read_csv(os.path.join(BEST_RUN, "predictions.csv"))
    st = ev.select("fileId").join(st.select("fileId", pl.col("pred_has_text").alias("has_text")), on="fileId", how="left", maintain_order="left")
    cand = (~ev["has_text"].to_numpy()) & (teacher_text | st["has_text"].fill_null(False).to_numpy().astype(bool))
    eval_ids = ev["fileId"].to_numpy()[cand].tolist()
    ids = [("r", f) for f in real_ids] + [("e", f) for f in eval_ids]
    pl.DataFrame({"set": [s for s, _ in ids], "fileId": [f for _, f in ids]}).write_csv(
        os.path.join(CHECK_DIR, "text_recheck.csv"))
    for k in range(0, len(ids), 12):
        part = ids[k:k + 12]
        plot_image_grid([f for _, f in part], [f"{s}{k + i:03d}:{f}" for i, (s, f) in enumerate(part)],
                        f"文字ありの見直し {k // 12}", os.path.join(CHECK_DIR, f"text_{k // 12:02d}.png"), ncols=4)
    print(f"実データ {len(real_ids)} 枚，評価用データ {len(eval_ids)} 枚の一覧を作成した")


def main() -> None:
    """コマンドライン引数に従って抽出を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--strat", action="store_true")
    parser.add_argument("--text_recheck", action="store_true")
    args = parser.parse_args()
    os.makedirs(CHECK_DIR, exist_ok=True)
    if args.strat:
        ids = make_strat()
        pl.DataFrame({"fileId": random160_ids() + ids}).write_csv(os.path.join(DATASET_DIR, "candidates", "real_check_v4.csv"))
    if args.text_recheck:
        make_text_recheck()


if __name__ == "__main__":
    main()
