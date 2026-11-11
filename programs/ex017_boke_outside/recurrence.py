"""ボケの文が，別のお題にどれだけ繰り返し投稿されているか（再出現の回数）を数え，画像外の知識の手がかりになるかを調べる．

有名な台詞・歌詞・ネットの定型などを引用するボケは，同じ文が多くのお題に投稿されやすいと考えられる．
BoketeDBv4 の全てのボケ（814ファイル）について，文を正規化（NFKC，空白・記号を除く，小文字化）し，
正規化した文ごとに「投稿されたお題の数」を数える．目視のラベルのある1,551件（学習用931件と検証データ620件）について，
自分のお題を除いた回数（other_odai）を求める．

出力：outputs/ex017_boke_outside/recurrence/labeled.csv（列 key, norm, other_odai），counts.parquet（正規化した文 → お題の数．2以上のみ）

実行例:
    .venv_annotate/bin/python programs/ex017_boke_outside/recurrence.py
"""
import glob
import os
import re
import sys
import unicodedata

import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, load_labeled  # noqa: E402

BOKE_DIR = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4/boke"
PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """ボケの文を正規化する．

    Args:
        text (str): ボケの文．

    Returns:
        str: NFKC・小文字化し，空白と記号を除いた文．
    """
    return PUNCT.sub("", unicodedata.normalize("NFKC", text or "").lower())


def norm_expr(col: str) -> pl.Expr:
    """polars で normalize と同じ正規化を行う式（NFKC は map_elements で行う）．

    Args:
        col (str): 列名．

    Returns:
        pl.Expr: 正規化した文の式．
    """
    return pl.col(col).fill_null("").str.normalize("NFKC").str.to_lowercase().str.replace_all(r"[\s\W_]+", "")


def main() -> None:
    """全てのボケの正規化した文ごとのお題の数を数え，目視のラベルのある項目の再出現の回数を求める．

    Returns:
        None
    """
    out = os.path.join(OUTPUT_DIR, "recurrence")
    os.makedirs(out, exist_ok=True)
    parts = []
    for f in tqdm(sorted(glob.glob(os.path.join(BOKE_DIR, "boke_*.parquet"))), leave=False):
        d = pl.read_parquet(f, columns=["odaiId", "text"]).select("odaiId", norm_expr("text").alias("norm"))
        d = d.filter(pl.col("norm").str.len_chars() >= 4).unique(["odaiId", "norm"])
        parts.append(d.group_by("norm").agg(pl.len().alias("n")))
    cnt = pl.concat(parts).group_by("norm").agg(pl.col("n").sum()).filter(pl.col("n") >= 2)
    cnt.write_parquet(os.path.join(out, "counts.parquet"))
    lab = load_labeled().select("key", "text").with_columns(norm_expr("text").alias("norm"))
    lab = lab.join(cnt, on="norm", how="left").with_columns((pl.col("n").fill_null(1) - 1).alias("other_odai"))
    lab.select("key", "norm", "other_odai").write_csv(os.path.join(out, "labeled.csv"))
    print("distinct recurring texts", cnt.height, "labeled with other_odai>0", int((lab["other_odai"] > 0).sum()), flush=True)


if __name__ == "__main__":
    main()
