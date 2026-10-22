"""VLM の出力に定義書の規則と補助の証拠（ボケの本文）を適用する後処理．

1. 作品不明：媒体がアニメ・マンガ・ゲームで，作品名を特定できなかった（known_work が偽の）画像は work_unknown とする．
2. カルタのボケによる照合：カルタ記号ありと判定された画像で，かなで始まるボケが MIN_KANA 件以上あり，
   そのうち karuta_char で始まるボケの割合が VETO_RATE 未満であれば，カルタ記号なしとして形式を導き直す．

定義書（label_definition_v1.md）の「ボケの傾向は判定の証拠として用いる」に従う処理である．
"""
import os
import sys
from collections import Counter

import polars as pl

from data import DATASET_DIR, FORMAT_IMPLIES, PROJECT_DIR

sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex003_user_scheme_pilot"))
from pilot import first_char, load_bokes  # noqa: E402

MIN_KANA = 5
VETO_RATE = 0.3


def boke_kana_stats(ids: list) -> pl.DataFrame:
    """お題ごとに，かなで始まるボケの書き出しの文字の数え上げを求める（キャッシュを用いる）．

    Args:
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 fileId, n_kana（かなで始まるボケの数），kana_counts（"文字:数" を ";" で連結した文字列）を持つ表．
    """
    cache = os.path.join(DATASET_DIR, "boke_kana_stats.parquet")
    have = pl.read_parquet(cache) if os.path.exists(cache) else pl.DataFrame(
        schema={"fileId": pl.Int64, "n_kana": pl.Int64, "kana_counts": pl.String})
    need = sorted(set(ids) - set(have["fileId"].to_list()))
    if need:
        bokes = load_bokes(need)
        rows = []
        for (fid,), g in bokes.group_by("odaiId"):
            chars = [c for c in (first_char(t) for t in g["text"].fill_null("").to_list()) if "ぁ" <= c <= "ゖ"]
            cnt = Counter(chars)
            rows.append({"fileId": fid, "n_kana": len(chars), "kana_counts": ";".join(f"{k}:{v}" for k, v in cnt.items())})
        found = {r["fileId"] for r in rows}
        rows += [{"fileId": f, "n_kana": 0, "kana_counts": ""} for f in need if f not in found]
        have = pl.concat([have, pl.DataFrame(rows, schema=have.schema)])
        have.write_parquet(cache)
    return pl.DataFrame({"fileId": ids}).join(have, on="fileId", how="left", maintain_order="left")


def karuta_rate(kana_counts: str, n_kana: int, ch: str) -> float:
    """かなで始まるボケのうち，指定したひらがなで始まるものの割合を求める．

    Args:
        kana_counts (str): "文字:数" を ";" で連結した文字列．
        n_kana (int): かなで始まるボケの数．
        ch (str): ひらがな1文字．

    Returns:
        float: 割合（n_kana が0の場合は NaN）．
    """
    if not n_kana:
        return float("nan")
    counts = dict(kv.split(":") for kv in kana_counts.split(";") if kv)
    return int(counts.get(ch, 0)) / n_kana


def apply_rules(pred: pl.DataFrame, ids: list, use_work_unknown: bool = True, use_boke: bool = True) -> pl.DataFrame:
    """VLM の予測に規則を適用する．

    Args:
        pred (pl.DataFrame): 列 format, karuta_char と各項目を持つ予測の表（行の順は ids と同じ）．
        ids (list[int]): お題IDのリスト．
        use_work_unknown (bool): 規則1を適用するか．
        use_boke (bool): 規則2を適用するか．

    Returns:
        pl.DataFrame: 規則を適用した予測の表（列 boke_karuta_rate を追加）．
    """
    df = pred.clone()
    if use_work_unknown:
        df = df.with_columns(((pl.col("anime") | pl.col("manga") | pl.col("game")) & ~pl.col("known_work"))
                             .alias("work_unknown"))
    stats = boke_kana_stats(ids)
    rate = [karuta_rate(c or "", n or 0, ch or "") for c, n, ch in
            zip(stats["kana_counts"], stats["n_kana"], df["karuta_char"].fill_null(""))]
    df = df.with_columns(pl.Series("boke_karuta_rate", rate, dtype=pl.Float64))
    if use_boke:
        veto = (pl.col("has_karuta_mark") & pl.Series(stats["n_kana"].fill_null(0) >= MIN_KANA)
                & (pl.col("boke_karuta_rate") < VETO_RATE))
        df = df.with_columns(pl.when(veto).then(False).otherwise(pl.col("has_karuta_mark")).alias("has_karuta_mark"))
        df = df.with_columns(
            pl.when(pl.col("has_karuta_mark")).then(pl.lit("karuta"))
            .when(pl.col("has_blank")).then(pl.lit("anaume"))
            .when(pl.col("has_prompt_text")).then(pl.lit("odaibun"))
            .otherwise(pl.lit("normal")).alias("format"))
        for fmt, attr in FORMAT_IMPLIES.items():
            df = df.with_columns((pl.col(attr) | (pl.col("format") == fmt)).alias(attr))
    return df
