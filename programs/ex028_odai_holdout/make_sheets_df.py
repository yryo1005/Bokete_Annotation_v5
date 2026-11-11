"""追加の抽出（order_049）：層 D・F をそれぞれ300枚抽出し，目視のラベルを付けるための番号だけの画像の一覧を作る．

これまでの全ての目視・教師の画像（ex024〜ex027 と ex028 の636枚）に含まれない代表の画像から抽出する．

出力：datasets/ex028_odai_holdout/targets_df.csv（列 no, rep_odaiId, stratum2, N_h, n_h），sheets_df/sheet_{k:03}.png

実行例:
    .venv_annotate/bin/python programs/ex028_odai_holdout/make_sheets_df.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, DF_N, DF_STRATA, SEED_DF, TARGETS, TARGETS_DF, excluded_reps, image_path, load_population  # noqa: E402
from make_sheets import PER_SHEET, draw  # noqa: E402


def sample_df(pop: pl.DataFrame, excluded: set) -> pl.DataFrame:
    """層 D・F から無作為に抽出する（画像のファイルがない画像は除く）．

    Args:
        pop (pl.DataFrame): load_population の戻り値．
        excluded (set[int]): 除く代表の画像．

    Returns:
        pl.DataFrame: 列 rep_odaiId, stratum2, N_h, n_h を持つ表．
    """
    cand = pop.filter(~pl.col("rep_odaiId").is_in(list(excluded)))
    parts = []
    for k, name in enumerate(DF_STRATA):
        n_h = pop.filter(pl.col("stratum2") == name).height
        s = cand.filter(pl.col("stratum2") == name).sample(DF_N * 2, seed=SEED_DF + k)
        s = s.filter(pl.Series([os.path.exists(image_path(i)) for i in s["rep_odaiId"].to_list()])).head(DF_N)
        assert s.height == DF_N, f"{name} の画像が足りない"
        parts.append(s.select("rep_odaiId", "stratum2", pl.lit(n_h, pl.Int64).alias("N_h"), pl.lit(DF_N, pl.Int64).alias("n_h")))
    return pl.concat(parts)


def main() -> None:
    """抽出して一覧を保存する（targets_df.csv があれば抽出し直さない）．

    Returns:
        None
    """
    out_dir = os.path.join(DATASET_DIR, "sheets_df")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(TARGETS_DF):
        excluded = excluded_reps() | set(pl.read_csv(TARGETS)["rep_odaiId"].to_list())
        t = sample_df(load_population(), excluded)
        t = t[np.random.default_rng(SEED_DF).permutation(t.height)].with_row_index("no")
        t.write_csv(TARGETS_DF)
    rows = pl.read_csv(TARGETS_DF).select("no", "rep_odaiId").to_dicts()
    jobs = [(rows[k:k + PER_SHEET], os.path.join(out_dir, f"sheet_{k // PER_SHEET:03d}.png")) for k in range(0, len(rows), PER_SHEET)]
    with ProcessPoolExecutor(8) as ex:
        list(tqdm(ex.map(draw, *zip(*jobs)), total=len(jobs), desc="sheets", leave=False))
    print(pl.read_csv(TARGETS_DF).group_by("stratum2").len())


if __name__ == "__main__":
    main()
