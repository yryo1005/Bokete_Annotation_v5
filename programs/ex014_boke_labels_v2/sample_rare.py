"""発話の型の少ない型を多く含む検証データの追加の抽出（ex013 の学習用のボケ3.5万件から）．

形式が通常かカルタのボケについて，規則（ex013 rules.py）か教師（30B-A3B 第2版）の少なくとも一方がその型と予測したボケを候補とし，
心の声・ツッコミ・状況の説明・名付け・見立て・セリフから各 N_PER 件を，お題1件につき1件まで無作為に抽出する．
抽出したボケは学習用から除く（生徒の学習で除外する）．一覧には予測を表示しない．

出力：datasets/ex014_boke_labels_v2/eval2_bokes.csv（列 no（420〜），odaiId, stratum, format, text, pool_index, cand），sheets/sheet_XX.png

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/sample_rare.py
"""
import importlib.util
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX13 = os.path.join(PROJECT_DIR, "programs", "ex013_boke_labels")
sys.path.insert(0, EX13)
import data as D13  # noqa: E402
import rules as R13  # noqa: E402

OUT = os.path.join(PROJECT_DIR, "datasets", "ex014_boke_labels_v2")
N_PER = 40
TARGETS = ["M", "T", "N", "L", "S"]


def plot_sheet(rows: list, path: str) -> None:
    """画像とボケを2列×4行に並べた一覧を保存する．

    Args:
        rows (list[tuple]): (番号, お題ID, 形式, ボケ) のリスト．
        path (str): 保存先．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={"width_ratios": [1, 1.1, 1, 1.1]})
    for ax in axes.ravel():
        ax.axis("off")
    for k, (no, fid, fmt, text) in enumerate(rows):
        r, c = k // 2, (k % 2) * 2
        im = Image.open(D13.image_path(fid)).convert("RGB")
        im.thumbnail((400, 400))
        axes[r, c].imshow(im)
        axes[r, c].set_title(f"{no}（{fmt}）", fontsize=9)
        axes[r, c + 1].text(0, 0.95, "\n".join(textwrap.wrap((text or "").replace("\n", " "), 18)), va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    """抽出と一覧の作成を行う．

    Returns:
        None
    """
    pool = pl.read_parquet(os.path.join(D13.DATASET_DIR, "train_pool.parquet")).with_row_index("pool_index")
    t = pl.read_csv(os.path.join(D13.OUTPUT_DIR, "teacher_q30a3_v2", "train.csv"))
    pool = pl.concat([pool, t.select(pl.col("voice").alias("t_voice"))], how="horizontal")
    pool = pool.filter(pl.col("format").is_in(["normal", "karuta"]))
    pool = pool.with_columns(pl.Series("r_voice", [R13.rule_voice(x, f) for x, f in zip(pool["text"], pool["format"])]))
    used = set(D13.load_eval()["odaiId"].to_list())
    rng = np.random.default_rng(5)
    picked, recs = set(), []
    for v in TARGETS:
        cand = pool.filter(((pl.col("t_voice") == v) | (pl.col("r_voice") == v)) & ~pl.col("odaiId").is_in(list(used | picked)))
        cand = cand.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1e6))).unique("odaiId", keep="first", maintain_order=True)
        part = cand.head(N_PER)
        picked |= set(part["odaiId"].to_list())
        recs.append(part.with_columns(pl.lit(v).alias("cand")))
    s = pl.concat(recs).sample(fraction=1.0, shuffle=True, seed=6)
    s = s.with_columns((pl.int_range(pl.len()) + 420).alias("no")).select("no", "odaiId", "stratum", "format", "text", "pool_index", "cand")
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    s.write_csv(os.path.join(OUT, "eval2_bokes.csv"))
    rows = s.select("no", "odaiId", "format", "text").rows()
    for k in range(0, len(rows), 8):
        plot_sheet(rows[k:k + 8], os.path.join(OUT, "sheets", f"sheet_{k // 8:02d}.png"))
    print(s.height, s.group_by("cand").len().rows())


if __name__ == "__main__":
    main()
