"""新しい評価用データと再現性の確認の画像を抽出し，目視のラベルを付けるための画像の一覧（番号付き）を作る．

予測や過去のラベルの影響を避けるため，2種類の画像を混ぜて並べ替え，一覧には番号だけを表示する．

出力：datasets/ex025_odaibun_eval_v5/targets.csv（列 no, rep_odaiId, kind（new / repeat）, stratum, N_h, n_h），
      datasets/ex025_odaibun_eval_v5/population_strata.csv（層ごとの枚数），sheets/sheet_{k:03}.png

実行例:
    .venv_annotate/bin/python programs/ex025_odaibun_eval_v5/make_sheets.py
"""
import os
import sys

import matplotlib
import numpy as np
import polars as pl
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, REPEAT_PER_GROUP, SEED, STRATA, TARGETS, excluded_reps, image_path, load_population, load_truth_ex024  # noqa: E402

PER_SHEET = 9
TILE = 520


def sample_new(pop: pl.DataFrame, excluded: set) -> pl.DataFrame:
    """層ごとに新しい評価用データの画像を無作為に抽出する．

    Args:
        pop (pl.DataFrame): load_population の戻り値．
        excluded (set[int]): 除く代表の画像．

    Returns:
        pl.DataFrame: 列 rep_odaiId, kind, stratum, N_h, n_h を持つ表．
    """
    pop = pop.filter(~pl.col("rep_odaiId").is_in(list(excluded)))
    parts = []
    for k, (name, n) in enumerate(STRATA.items()):
        s = pop.filter(pl.col("stratum") == name)
        parts.append(s.sample(n, seed=SEED + k).select("rep_odaiId", pl.lit("new").alias("kind"), "stratum",
                                                       pl.lit(s.height, pl.Int64).alias("N_h"), pl.lit(n, pl.Int64).alias("n_h")))
    return pl.concat(parts)


def sample_repeat() -> pl.DataFrame:
    """既存の1,020枚から，再現性の確認の画像を群ごとに抽出する．

    群：お題文（形式がお題文．お題文あり），blank_karuta（形式が穴埋め・カルタ．付け直したラベル），normal（形式が通常．お題文なし）．

    Returns:
        pl.DataFrame: 列 rep_odaiId, kind, stratum（群）, N_h, n_h を持つ表（N_h と n_h は null）．
    """
    t = load_truth_ex024().with_columns(
        pl.when(pl.col("format").is_in(["anaume", "karuta"])).then(pl.lit("blank_karuta")).otherwise(pl.col("format")).alias("group"))
    parts = [t.filter(pl.col("group") == g).sample(REPEAT_PER_GROUP, seed=SEED) for g in ["odaibun", "blank_karuta", "normal"]]
    return pl.concat(parts).select(pl.col("fileId").alias("rep_odaiId"), pl.lit("repeat").alias("kind"), pl.col("group").alias("stratum"),
                                   pl.lit(None, pl.Int64).alias("N_h"), pl.lit(None, pl.Int64).alias("n_h"))


def draw(rows: list, out: str) -> None:
    """画像の一覧を1枚の図にする．

    Args:
        rows (list[dict]): 列 no, rep_odaiId を持つ行のリスト（最大 PER_SHEET 行）．
        out (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        im = Image.open(image_path(r["rep_odaiId"])).convert("RGB")
        im.thumbnail((TILE, TILE))
        ax.imshow(im)
        ax.set_title(f"{r['no']}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=60)
    plt.close(fig)


def main() -> None:
    """抽出して一覧を保存する（targets.csv があれば抽出し直さない）．

    Returns:
        None
    """
    os.makedirs(os.path.join(DATASET_DIR, "sheets"), exist_ok=True)
    if not os.path.exists(TARGETS):
        pop = load_population()
        pop.group_by("stratum").agg(pl.len().alias("N"), pl.col("p").mean().alias("mean_p")).sort("stratum").write_csv(
            os.path.join(DATASET_DIR, "population_strata.csv"))
        t = pl.concat([sample_new(pop, excluded_reps()), sample_repeat()])
        order = np.random.default_rng(SEED).permutation(t.height)
        t = t[order].with_row_index("no")
        t.write_csv(TARGETS)
    rows = pl.read_csv(TARGETS).to_dicts()
    for k in tqdm(range(0, len(rows), PER_SHEET), desc="sheets", leave=False):
        draw(rows[k:k + PER_SHEET], os.path.join(DATASET_DIR, "sheets", f"sheet_{k // PER_SHEET:03d}.png"))
    print(pl.read_csv(TARGETS).group_by("kind", "stratum").len().sort("kind", "stratum"))


if __name__ == "__main__":
    main()
