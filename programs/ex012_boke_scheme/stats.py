"""ボケの区分の設計のための集計（試験的なラベルの集計と，大規模な自動の集計）．

1. 試験的なラベル（pilot_labels.txt，140件）の区分ごとの件数．
2. お題の形式・媒体ごとに無作為に N_ODAI 件のお題を抽出し，そのボケの全件について次を求める．
   - ボケの文字数の分布（形式ごと）
   - カルタのお題で，かなで始まるボケのうち最も多い書き出しのかなの割合（制約への従い方）
   - 評価（rateSum）が0のボケの割合と，評価の件数の分布（判定の教師に使えるかの目安）

出力：outputs/ex012_boke_scheme/stats.json

実行例:
    .venv_annotate/bin/python programs/ex012_boke_scheme/stats.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from sample import ANN, DB_DIR, PROJECT_DIR, strata  # noqa: E402

OUT = os.path.join(PROJECT_DIR, "outputs", "ex012_boke_scheme")
N_ODAI = 300
SKIP = set("「『（(【［[ 　\"'“”…・")


def pilot() -> dict:
    """試験的なラベルを集計する．

    Returns:
        dict: 区分ごとの件数（全体，層ごと，評価の上位と無作為）．
    """
    s = pl.read_csv(os.path.join(PROJECT_DIR, "datasets", "ex012_boke_scheme", "sample_bokes.csv"))
    rows = [ln.split() for ln in open(os.path.join(THIS_DIR, "pilot_labels.txt"), encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    lab = pl.DataFrame({"no": [int(r[0]) for r in rows], "voice": [r[1] for r in rows], "know": [r[2] for r in rows],
                        "uses_text": [int(r[3]) for r in rows], "flag": [r[4] for r in rows]})
    d = s.join(lab, on="no")
    out = {"n": d.height}
    for c in ["voice", "know", "uses_text"]:
        out[c] = dict(d.group_by(c).len().iter_rows())
        out[f"{c}_by_stratum"] = {st: dict(g.group_by(c).len().iter_rows()) for (st,), g in d.group_by("stratum")}
        out[f"{c}_by_kind"] = {k: dict(g.group_by(c).len().iter_rows()) for (k,), g in d.group_by("kind")}
    out["flags"] = dict(d.filter(pl.col("flag") != "-").group_by("flag").len().iter_rows())
    return out


def first_kana(text: str) -> str:
    """ボケの書き出しのかな（括弧や記号を飛ばし，カタカナはひらがなにする）を求める．

    Args:
        text (str): ボケの文．

    Returns:
        str: ひらがな1文字（かなで始まらなければ空文字）．
    """
    for ch in text:
        if ch in SKIP:
            continue
        if "ァ" <= ch <= "ヶ":
            ch = chr(ord(ch) - 0x60)
        return ch if "ぁ" <= ch <= "ゖ" else ""
    return ""


def large_scale(seed: int = 0) -> dict:
    """層ごとに無作為のお題のボケの全件を集計する．

    Args:
        seed (int): 乱数のシード．

    Returns:
        dict: 層ごとの文字数，評価，カルタの書き出しの集計．
    """
    ann = pl.scan_parquet(ANN).filter(pl.col("odaiId") == pl.col("rep_odaiId")).collect()
    rng = np.random.default_rng(seed)
    out = {}
    for st, cond in strata().items():
        ids = rng.choice(ann.filter(cond)["odaiId"].to_numpy(), N_ODAI, replace=False).tolist()
        files = sorted({os.path.join(DB_DIR, "boke", f"boke_{i // 10000 + 1}.parquet") for i in ids})
        b = pl.scan_parquet(files).filter(pl.col("odaiId").is_in(ids)).select(
            "odaiId", "text", "rateSum", "rateCount").collect().filter(pl.col("text").fill_null("") != "")
        n = b["text"].str.len_chars()
        r = {"n_odai": len(ids), "n_bokes": b.height, "bokes_per_odai_median": float(b.group_by("odaiId").len()["len"].median()),
             "chars_median": float(n.median()), "chars_p25": float(n.quantile(0.25)), "chars_p75": float(n.quantile(0.75)),
             "rate0_rate": float((b["rateSum"] == 0).mean()), "ratecount_median": float(b["rateCount"].median()),
             "ratecount_ge5_rate": float((b["rateCount"] >= 5).mean())}
        if st == "karuta":
            b = b.with_columns(pl.col("text").map_elements(first_kana, return_dtype=pl.String).alias("k"))
            shares = []
            for (_,), g in b.filter(pl.col("k") != "").group_by("odaiId"):
                if g.height >= 5:
                    shares.append(g.group_by("k").len()["len"].max() / g.height)
            r["karuta_top_kana_share_median"] = float(np.median(shares))
            r["karuta_top_kana_share_ge08_rate"] = float(np.mean(np.array(shares) >= 0.8))
            r["karuta_n_odai_used"] = len(shares)
        out[st] = r
    return out


def main() -> None:
    """集計して保存する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    res = {"pilot": pilot(), "large_scale": large_scale()}
    json.dump(res, open(os.path.join(OUT, "stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["pilot"]["uses_text_by_stratum"], ensure_ascii=False))
    for st, r in res["large_scale"].items():
        print(st, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
