"""全データ（BoketeDBv4 の 6,823,270 お題）に対する生徒の予測（確認の質問の前）から，ラベルの割合と年ごとの推移を求める．

Bokete_Annotation_programs_v5.0 の predict 段階の出力（代表画像ごとの確率）を，同じ後処理（確認の質問なし）でラベルにし，
images.parquet でお題に展開し，BoketeDBv4 の odai の表の投稿日時と結合する．

出力:
    outputs/ex010_paper/full_distribution.json：全体の割合，年ごとの割合，年ごとのお題数
    paper/figures/fig_full_year_format.pdf：年ごとの形式の割合
    paper/figures/fig_full_year_media.pdf：年ごとの媒体の割合（小分け）

実行例:
    .venv_annotate/bin/python programs/ex010_paper/full_distribution.py
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, "Bokete_Annotation_programs_v5.0")
sys.path.insert(0, PKG)
from bokete_annotation.labels import ATTRIBUTES, FORMATS, MEDIA  # noqa: E402
from bokete_annotation.postprocess import to_labels  # noqa: E402

ANN_DIR = os.path.join(PKG, "outputs", "BoketeDBv5_annotation")
DB_DIR = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4"
OUT_DIR = os.path.join(ROOT, "outputs", "ex010_paper")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
F_JA = {"karuta": "Karuta", "anaume": "Anaume", "odaibun": "Odaibun", "normal": "Normal"}
M_EN = {"photo": "Photo", "anime": "Anime", "game": "Game", "manga": "Manga", "text_graphic": "Text graphic",
        "painting": "Painting", "illustration": "Illustration"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_rep_labels() -> pl.DataFrame:
    """predict 段階の確率から代表画像ごとのラベルを作る（確認の質問は適用しない）．

    Returns:
        pl.DataFrame: 列 rep_odaiId，format，属性3項目，媒体7項目（行数は代表画像の数）．
    """
    df = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(ANN_DIR, "predict", "part_*.parquet")))])
    p_format = df.select([f"p_format_{f}" for f in FORMATS]).to_numpy()
    p_attr = df.select([f"p_{a}" for a in ATTRIBUTES]).to_numpy()
    p_media = df.select([f"p_{m}" for m in MEDIA]).to_numpy()
    lab = to_labels(p_format.argmax(1), p_attr, p_media)
    return pl.concat([df.select("rep_odaiId"), lab], how="horizontal")


def load_odai_year() -> pl.DataFrame:
    """お題の投稿年を読み込む．

    Returns:
        pl.DataFrame: 列 odaiId（Int64），year（Int32）．
    """
    files = glob.glob(os.path.join(DB_DIR, "odai", "odai_*.parquet"))
    o = pl.concat([pl.read_parquet(f, columns=["id", "date", "dateIsEmpty"]) for f in files])
    o = o.filter(~pl.col("dateIsEmpty")).unique("id")
    return o.select(pl.col("id").alias("odaiId"), pl.col("date").dt.year().alias("year"))


def rates(df: pl.DataFrame) -> dict:
    """ラベルの割合を求める．

    Args:
        df (pl.DataFrame): format，属性，媒体の列を持つ表．

    Returns:
        dict: 形式の割合，属性・媒体の割合，媒体の複数選択の割合．
    """
    n = df.height
    out = {"n": n, "format": {f: df.filter(pl.col("format") == f).height / n for f in FORMATS}}
    out["item"] = {c: float(df[c].mean()) for c in ATTRIBUTES + MEDIA}
    out["media_multi"] = float((df.select(pl.sum_horizontal(MEDIA)).to_series() >= 2).mean())
    return out


def plot_format(yr: dict, years: list) -> None:
    """年ごとの形式の割合（通常を除く3形式）を折れ線で描く．

    Args:
        yr (dict): 年ごとの rates の結果．
        years (list): 描く年の一覧．
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for c, f in zip(COLORS, ["anaume", "odaibun", "karuta"]):
        y = [100 * yr[y]["format"][f] for y in years]
        ax.plot(years, y, color=c, lw=2, marker="o", ms=4, label=F_JA[f])
        ax.annotate(F_JA[f], (years[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#333333")
    ax.set_xlabel("Year of posting")
    ax.set_ylabel("Share of odai [%]")
    ax.set_xlim(years[0] - 0.5, years[-1] + 2.2)
    ax.set_xticks(list(range(2008, years[-1] + 1, 2)))
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#e0e0e0", lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_full_year_format.pdf"))
    plt.close(fig)


def plot_media(yr: dict, years: list) -> None:
    """年ごとの媒体の割合を，媒体ごとの小分けの折れ線で描く．

    Args:
        yr (dict): 年ごとの rates の結果．
        years (list): 描く年の一覧．
    """
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
    for ax, m in zip(axes.flat, MEDIA + ["has_text"]):
        y = [100 * yr[y]["item"][m] for y in years]
        ax.plot(years, y, color=COLORS[0], lw=2)
        ax.set_title(M_EN.get(m, "Has text (attr.)"), fontsize=9)
        ax.set_ylim(0, max(y) * 1.25)
        ax.grid(axis="y", color="#e0e0e0", lw=0.8)
        ax.tick_params(labelsize=7)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xticks([2010, 2015, 2020, 2025])
    fig.supxlabel("Year of posting", fontsize=9)
    fig.supylabel("Share of odai [%]", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_full_year_media.pdf"))
    plt.close(fig)


def main() -> None:
    """全データのラベルの割合を集計し，JSON と図を保存する．"""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    rep = load_rep_labels()
    images = pl.read_parquet(os.path.join(ANN_DIR, "images.parquet"), columns=["odaiId", "rep_odaiId"])
    odai = images.join(rep, on="rep_odaiId", how="inner")
    res = {"n_rep_images": rep.height, "n_odai": odai.height, "all_odai": rates(odai), "rep_images": rates(rep)}
    odai = odai.join(load_odai_year(), on="odaiId", how="left")
    counts = odai.group_by("year").len().sort("year")
    res["year_count"] = {str(r["year"]): r["len"] for r in counts.iter_rows(named=True)}
    years = [r["year"] for r in counts.iter_rows(named=True) if r["year"] is not None and r["len"] >= 10000]
    yr = {y: rates(odai.filter(pl.col("year") == y)) for y in years}
    res["by_year"] = {str(y): v for y, v in yr.items()}
    with open(os.path.join(OUT_DIR, "full_distribution.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    plot_format(yr, years)
    plot_media(yr, years)
    print(json.dumps({k: res[k] for k in ["n_rep_images", "n_odai", "all_odai", "year_count"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
