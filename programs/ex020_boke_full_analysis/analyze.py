"""ボケの全件のアノテーション（Bokete_Annotation_programs_v5.0 の boke_annotation/）の確認と集計．

1. 件数の確認：BoketeDBv4 の全てのボケ ID が，出力にちょうど1回ずつ含まれるか．予測しなかったボケの内訳．
2. ラベルの分布：全体，形式ごと，投稿年ごとの発話の型（6区分・3区分）・画像外の知識・制約の順守の割合．
3. 図：投稿年ごとの3区分と画像外の知識の割合，形式ごとの6区分の割合，順守の得点の分布．
4. 目視の確認用の一覧：ラベルごとに無作為の8件（お題画像とボケ）を並べた画像．

出力：outputs/ex020_boke_full_analysis/（check.json，dist.json，by_year.csv，by_format.csv，fig_*.png，sheets/）

実行例:
    .venv_annotate/bin/python programs/ex020_boke_full_analysis/analyze.py
"""
import argparse
import glob
import json
import os
import textwrap

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ANN = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation")
DB = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4"
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex020_boke_full_analysis")
VOICES = ["S", "M", "T", "N", "L", "A"]
VOICE_JA = {"S": "セリフ", "M": "心の声", "T": "ツッコミ", "N": "状況の説明", "L": "名付け", "A": "回答"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a7a7a"]
FONT = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"


def setup_font() -> None:
    """図の日本語のフォントを設定する．

    Returns:
        None
    """
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def load() -> pl.LazyFrame:
    """ボケのラベルと，投稿日時（BoketeDBv4 の boke の date）を結合した表を返す．

    Returns:
        pl.LazyFrame: ボケのラベルの列と year を持つ表．
    """
    ann = pl.scan_parquet(os.path.join(ANN, "boke_annotation", "*.parquet"))
    date = pl.scan_parquet(glob.glob(os.path.join(DB, "boke", "boke_*.parquet"))).select("id", pl.col("date").dt.year().alias("year"))
    return ann.join(date, on="id", how="left")


def check() -> dict:
    """件数を確かめる．

    Returns:
        dict: 確認の結果．
    """
    ann = pl.scan_parquet(os.path.join(ANN, "boke_annotation", "*.parquet"))
    # 出力に含まれるシャード（ボケの表のファイル）だけを対象にする（全件の実行では814ファイル全て）
    nums = [int(os.path.basename(p)[5:9]) for p in glob.glob(os.path.join(ANN, "boke_annotation", "part_*.parquet"))]
    ids_db = pl.scan_parquet([os.path.join(DB, "boke", f"boke_{n}.parquet") for n in nums]).select("id")
    n_db = ids_db.select(pl.len()).collect().item()
    r = ann.select(pl.len().alias("n"), pl.col("id").n_unique().alias("n_unique"), pl.col("predicted").sum().alias("n_predicted"),
                   pl.col("format").is_null().sum().alias("n_no_format"), pl.col("empty_text").sum().alias("n_empty"),
                   (~pl.col("predicted") & pl.col("format").is_not_null()).sum().alias("n_unpredicted_with_format")).collect().row(0, named=True)
    missing = ids_db.join(ann.select("id"), on="id", how="anti").select(pl.len()).collect().item()
    return {"n_db": n_db, **r, "n_missing_in_output": missing, "all_present_once": n_db == r["n"] == r["n_unique"] and missing == 0}


def distributions(df: pl.LazyFrame) -> dict:
    """全体・形式ごと・投稿年ごとの分布を求めて保存する．

    Args:
        df (pl.LazyFrame): load の戻り値．

    Returns:
        dict: 全体の分布．
    """
    p = df.filter(pl.col("predicted"))
    agg = [pl.len().alias("n"), pl.col("outside").mean().alias("outside")] + \
        [(pl.col("voice6") == v).mean().alias(f"v6_{v}") for v in VOICES] + \
        [(pl.col("voice3") == v).mean().alias(f"v3_{v}") for v in ["character", "outside", "answer"]]
    total = p.select(agg).collect().row(0, named=True)
    by_fmt = p.group_by("format").agg(agg + [pl.col("karuta_complies").mean().alias("karuta_complies"),
                                             pl.col("compliance_score").mean().alias("compliance_score_mean"),
                                             pl.col("compliance_score").quantile(0.05).alias("compliance_score_p05")]).collect().sort("format")
    by_year = p.group_by("year").agg(agg).collect().sort("year")
    by_fmt.write_csv(os.path.join(OUTPUT_DIR, "by_format.csv"))
    by_year.write_csv(os.path.join(OUTPUT_DIR, "by_year.csv"))
    out = {"total": total, "by_format": by_fmt.to_dicts()}
    json.dump(out, open(os.path.join(OUTPUT_DIR, "dist.json"), "w"), indent=1, ensure_ascii=False)
    return out


def figures() -> None:
    """図を作る．

    Returns:
        None
    """
    setup_font()
    y = pl.read_csv(os.path.join(OUTPUT_DIR, "by_year.csv")).filter(pl.col("n") >= 10000)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for col, name, c in [("v3_character", "登場人物の言葉", COLORS[0]), ("v3_outside", "外からの言葉", COLORS[1]),
                         ("v3_answer", "回答", COLORS[2]), ("outside", "画像外の知識", COLORS[3])]:
        ax.plot(y["year"], y[col], marker="o", color=c, label=name, ls="--" if col == "outside" else "-")
    ax.set_xlabel("ボケの投稿年")
    ax.set_xticks(y["year"].to_list()[::2])
    ax.set_ylabel("割合")
    ax.set_ylim(0, 0.7)
    ax.grid(color="#dddddd", lw=0.6)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_by_year.png"), dpi=150)
    plt.close(fig)
    f = pl.read_csv(os.path.join(OUTPUT_DIR, "by_format.csv")).filter(pl.col("format").is_not_null())
    fig, ax = plt.subplots(figsize=(7, 3.2))
    left = [0.0] * f.height
    for v, c in zip(VOICES, COLORS):
        vals = f[f"v6_{v}"].to_list()
        ax.barh(f["format"].to_list(), vals, left=left, color=c, label=VOICE_JA[v])
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("発話の型（6区分）の割合")
    ax.legend(fontsize=7, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_by_format.png"), dpi=150)
    plt.close(fig)


def sheets(df: pl.LazyFrame) -> None:
    """ラベルごとに無作為の8件の一覧を作る（目視の確認用）．

    Args:
        df (pl.LazyFrame): load の戻り値．

    Returns:
        None
    """
    setup_font()
    d = os.path.join(OUTPUT_DIR, "sheets")
    os.makedirs(d, exist_ok=True)
    img = pl.read_parquet(os.path.join(ANN, "images.parquet"), columns=["odaiId", "path"])
    base = pl.scan_parquet(glob.glob(os.path.join(DB, "boke", "boke_*.parquet"))).select("id", "text")
    cats = {f"voice_{v}": pl.col("voice6") == v for v in VOICES}
    cats.update({"outside_yes": pl.col("outside"), "outside_no": ~pl.col("outside")})
    for name, cond in cats.items():
        s = df.filter(pl.col("predicted") & cond).select("id", "odaiId", "format", "voice6", "p_outside").collect().sample(n=8, seed=0)
        s = s.join(base.filter(pl.col("id").is_in(s["id"].to_list())).collect(), on="id").join(img, on="odaiId", how="left")
        fig, axes = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={"width_ratios": [1, 1.1, 1, 1.1]})
        for ax in axes.ravel():
            ax.axis("off")
        for k, r in enumerate(s.iter_rows(named=True)):
            row, col = k // 2, (k % 2) * 2
            if r["path"]:
                im = Image.open(os.path.join(IMAGE_DIR, r["path"])).convert("RGB")
                im.thumbnail((400, 400))
                axes[row, col].imshow(im)
            axes[row, col].set_title(f"{r['format']} / {r['voice6']} / 外{r['p_outside']:.2f}", fontsize=9)
            axes[row, col + 1].text(0, 0.95, "\n".join(textwrap.wrap((r["text"] or "").replace("\n", " "), 18)), va="top", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=70)
        plt.close(fig)


def main() -> None:
    """確認・集計・図・一覧を作る．

    Returns:
        None
    """
    global ANN, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default=ANN, help="アノテーションの出力先（boke_annotation/ を含む）")
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--figures_only", action="store_true", help="集計済みの CSV から図だけを作り直す")
    args = parser.parse_args()
    ANN, OUTPUT_DIR = args.ann, args.out
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.figures_only:
        figures()
        return
    c = check()
    json.dump(c, open(os.path.join(OUTPUT_DIR, "check.json"), "w"), indent=1)
    print(json.dumps(c, indent=1))
    df = load()
    dist = distributions(df)
    print(json.dumps(dist["total"], indent=1))
    figures()
    if not args.figures_only:
        sheets(df)


if __name__ == "__main__":
    main()
