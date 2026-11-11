"""新しい評価用データ（600枚）と判断のずれの確認の画像（36枚）を抽出し，目視のラベルを付けるための画像の一覧を作る．

予測や過去のラベルの影響を避けるため，2種類の画像を混ぜて並べ替え，一覧には番号だけを表示する．

出力：datasets/ex028_odai_holdout/targets.csv（列 no, rep_odaiId, kind（new / repeat）, stratum2, N_h, n_h），
      population_strata.csv（層ごとの枚数），sheets/sheet_{k:03}.png

実行例:
    .venv_annotate/bin/python programs/ex028_odai_holdout/make_sheets.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import polars as pl
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, EX026, REPEAT, SEED, STRATA, TARGETS, excluded_reps, image_path, load_population  # noqa: E402

PER_SHEET = 9
TILE = 640


def draw(rows: list, out: str) -> None:
    """画像の一覧を1枚の図にする（ex026 と同じ大きさ）．

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
        ax.set_title(f"{r['no']}", fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def sample_new(pop: pl.DataFrame, excluded: set) -> pl.DataFrame:
    """層ごとに新しい評価用データの画像を無作為に抽出する（画像のファイルがない画像は除く）．

    Args:
        pop (pl.DataFrame): load_population の戻り値．
        excluded (set[int]): 除く代表の画像．

    Returns:
        pl.DataFrame: 列 rep_odaiId, kind, stratum2, N_h, n_h を持つ表．
    """
    cand = pop.filter(~pl.col("rep_odaiId").is_in(list(excluded)))
    parts = []
    for k, (name, n) in enumerate(STRATA.items()):
        n_h = pop.filter(pl.col("stratum2") == name).height
        s = cand.filter(pl.col("stratum2") == name).sample(n * 2, seed=SEED + k)
        s = s.filter(pl.Series([os.path.exists(image_path(i)) for i in s["rep_odaiId"].to_list()])).head(n)
        assert s.height == n, f"{name} の画像が足りない"
        parts.append(s.select("rep_odaiId", pl.lit("new").alias("kind"), "stratum2", pl.lit(n_h, pl.Int64).alias("N_h"),
                              pl.lit(n, pl.Int64).alias("n_h")))
    return pl.concat(parts)


def sample_repeat() -> pl.DataFrame:
    """ex026 の1,620枚から判断のずれの確認の画像を無作為に選ぶ．

    Returns:
        pl.DataFrame: 列 rep_odaiId, kind, stratum2, N_h, n_h を持つ表（stratum2 は "repeat"，N_h と n_h は null）．
    """
    t = pl.read_csv(EX026.TARGETS).sample(REPEAT, seed=SEED)
    return t.select("rep_odaiId", pl.lit("repeat").alias("kind"), pl.lit("repeat").alias("stratum2"),
                    pl.lit(None, pl.Int64).alias("N_h"), pl.lit(None, pl.Int64).alias("n_h"))


def main() -> None:
    """抽出して一覧を保存する（targets.csv があれば抽出し直さない）．

    Returns:
        None
    """
    os.makedirs(os.path.join(DATASET_DIR, "sheets"), exist_ok=True)
    if not os.path.exists(TARGETS):
        pop = load_population()
        pop.group_by("stratum2").agg(pl.len().alias("N"), pl.col("p").mean().alias("mean_p")).sort("stratum2").write_csv(
            os.path.join(DATASET_DIR, "population_strata.csv"))
        t = pl.concat([sample_new(pop, excluded_reps()), sample_repeat()])
        t = t[np.random.default_rng(SEED).permutation(t.height)].with_row_index("no")
        t.write_csv(TARGETS)
    rows = pl.read_csv(TARGETS).select("no", "rep_odaiId").to_dicts()
    jobs = [(rows[k:k + PER_SHEET], os.path.join(DATASET_DIR, "sheets", f"sheet_{k // PER_SHEET:03d}.png")) for k in range(0, len(rows), PER_SHEET)]
    with ProcessPoolExecutor(8) as ex:
        list(tqdm(ex.map(draw, *zip(*jobs)), total=len(jobs), desc="sheets", leave=False))
    print(pl.read_csv(TARGETS).group_by("kind", "stratum2").len().sort("kind", "stratum2"))


if __name__ == "__main__":
    main()
