"""大喜利（ボケ）の区分の設計のため，お題を層別に抽出し，ボケと画像を並べた一覧を作る．

お題の層（BoketeDBv5 のアノテーション．読み込みのみ）：
karuta，anaume，odaibun（形式），photo_text（通常・写真のみ・文字あり），photo_notext（通常・写真のみ・文字なし），
anime_manga（通常・アニメかマンガを含む），other（通常・イラスト，ゲーム，絵画，テキストグラフィックのいずれかを含む）．
各層から，ボケが N_MIN 件以上あるお題を N_ODAI 件無作為に抽出し，評価（rateSum）の上位3件と，それ以外から無作為に2件のボケを取る．

出力：datasets/ex012_boke_scheme/sample_bokes.csv（列 no, stratum, odaiId, boke_no, kind（top / random）, rateSum, text），
      sheets/sheet_XX.png（お題1件ごとに画像とボケを並べた一覧）

実行例:
    .venv_annotate/bin/python programs/ex012_boke_scheme/sample.py
"""
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ANN = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation", "annotation.parquet")
DB_DIR = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4"
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
OUT = os.path.join(PROJECT_DIR, "datasets", "ex012_boke_scheme")
N_ODAI, N_MIN = 4, 30


def strata() -> dict:
    """層の名前と，お題の条件（polars の式）を返す．

    Returns:
        dict[str, pl.Expr]: 層の名前 → 条件．
    """
    only_photo = pl.col("photo") & ~pl.any_horizontal(["anime", "game", "manga", "text_graphic", "painting", "illustration"])
    normal = pl.col("format") == "normal"
    return {"karuta": pl.col("format") == "karuta", "anaume": pl.col("format") == "anaume",
            "odaibun": pl.col("format") == "odaibun",
            "photo_text": normal & only_photo & pl.col("has_text"), "photo_notext": normal & only_photo & ~pl.col("has_text"),
            "anime_manga": normal & (pl.col("anime") | pl.col("manga")),
            "other": normal & pl.any_horizontal(["illustration", "game", "painting", "text_graphic"])}


def load_bokes(ids: list) -> pl.DataFrame:
    """お題のボケを読み込む（ボケの表はお題IDを10000で割った値＋1の番号のファイルにある）．

    Args:
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 odaiId, text, rateSum を持つ表．
    """
    files = sorted({os.path.join(DB_DIR, "boke", f"boke_{i // 10000 + 1}.parquet") for i in ids})
    return pl.scan_parquet(files).filter(pl.col("odaiId").is_in(ids)).select("odaiId", "text", "rateSum").collect()


def plot_sheet(rows: list, path: str) -> None:
    """お題ごとに画像とボケを並べた一覧を保存する．

    Args:
        rows (list[tuple]): (お題ID, 層, [(番号, 種類, 評価, 文)]) のリスト．
        path (str): 保存先．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.2 * len(rows)), gridspec_kw={"width_ratios": [1, 1.6]})
    for (fid, st, bokes), (a1, a2) in zip(rows, np.atleast_2d(axes)):
        im = Image.open(os.path.join(IMAGE_DIR, str(fid // 10000 * 10000), f"{fid}.jpg")).convert("RGB")
        im.thumbnail((420, 420))
        a1.imshow(im)
        a1.set_title(f"{fid}（{st}）", fontsize=9)
        a1.axis("off")
        a2.axis("off")
        lines = [f"{n:03d} [{k}:{r}] " + "\n      ".join(textwrap.wrap(t.replace("\n", " "), 44)) for n, k, r, t in bokes]
        a2.text(0, 1, "\n".join(lines), va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    """抽出と一覧の作成を行う．

    Returns:
        None
    """
    ann = pl.scan_parquet(ANN).filter(pl.col("odaiId") == pl.col("rep_odaiId")).collect()
    rng = np.random.default_rng(0)
    recs, sheets = [], []
    no = 0
    for st, cond in strata().items():
        cand = ann.filter(cond)["odaiId"].to_numpy()
        picked = 0
        for fid in rng.permutation(cand):
            if picked == N_ODAI:
                break
            b = load_bokes([int(fid)]).filter(pl.col("text").fill_null("").str.len_chars() > 0)
            if b.height < N_MIN:
                continue
            b = b.sort("rateSum", descending=True)
            top = b.head(3)
            rest = b.slice(3).sample(n=2, seed=int(fid))
            bokes = []
            for kind, part in [("top", top), ("random", rest)]:
                for t, r in zip(part["text"], part["rateSum"]):
                    recs.append({"no": no, "stratum": st, "odaiId": int(fid), "n_bokes": b.height, "kind": kind,
                                 "rateSum": int(r), "text": t})
                    bokes.append((no, kind, int(r), t))
                    no += 1
            sheets.append((int(fid), st, bokes))
            picked += 1
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    pl.DataFrame(recs).write_csv(os.path.join(OUT, "sample_bokes.csv"))
    for k in range(0, len(sheets), 4):
        plot_sheet(sheets[k:k + 4], os.path.join(OUT, "sheets", f"sheet_{k // 4:02d}.png"))
    print(len(recs), len(sheets))


if __name__ == "__main__":
    main()
