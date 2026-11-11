"""方法ごとの予測（outputs/ex014_boke_labels_v2/**/eval.csv）を，検証データ620件の目視のラベルと比べる．

区分の統合（data.MERGES）ごとに，発話の型と必要な知識の正解率とマクロ F1 を求める．画像中の文字への依存は F1．
最終評価用（test）全体と，そのうち元の420件（old）・少ない型の追加分（rare）の内訳も求める．
規則（ex013 rules.py の発話の型，形式による文字への依存）も同じ検証データで評価する．

出力：outputs/ex014_boke_labels_v2/figures/summary.csv，per_class.json

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/evaluate.py
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX13_DIR, MERGES, OUTPUT_DIR, _load, load_eval, merge  # noqa: E402

R13 = _load("ex013_rules_for_v2", os.path.join(EX13_DIR, "rules.py"))
FIG = os.path.join(OUTPUT_DIR, "figures")


def rules_pred(ev: pl.DataFrame) -> pl.DataFrame:
    """規則による予測を作る（知識は最も多い区分の「見た目だけ」とする）．

    Args:
        ev (pl.DataFrame): 検証データ．

    Returns:
        pl.DataFrame: 予測の表．
    """
    return pl.DataFrame({"no": ev["no"], "voice": [R13.rule_voice(t, f) for t, f in zip(ev["text"], ev["format"])],
                         "knowledge": ["0"] * ev.height, "uses_image_text": ev["format"].is_in(["anaume", "odaibun"])})


def scores(t: pl.DataFrame, p: pl.DataFrame) -> dict:
    """指標を求める．

    Args:
        t (pl.DataFrame): 目視のラベル．
        p (pl.DataFrame): 予測（t と同じ順）．

    Returns:
        dict: 区分ごとの正解率・マクロ F1，クラスごとの F1，文字への依存の F1．
    """
    r = {}
    for scheme, (item, mp) in MERGES.items():
        tc, pc = ("voice", "voice") if item == "voice" else ("know", "knowledge")
        y, q = merge(item, t[tc].to_list(), scheme), merge(item, p[pc].cast(pl.String).to_list(), scheme)
        labels = sorted(set(mp.values()))
        r[f"{scheme}_acc"] = accuracy_score(y, q)
        r[f"{scheme}_f1"] = f1_score(y, q, labels=labels, average="macro", zero_division=0)
        r[f"{scheme}_per_class"] = dict(zip(labels, f1_score(y, q, labels=labels, average=None, zero_division=0).round(3).tolist()))
    r["uses_text_f1"] = f1_score(t["uses_text"].to_numpy(), p["uses_image_text"].to_numpy().astype(bool), zero_division=0)
    return r


def main() -> None:
    """全ての方法を評価して保存する．

    Returns:
        None
    """
    ev = load_eval()
    preds = {"rules": rules_pred(ev)}
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "eval.csv"), recursive=True)):
        name = os.path.relpath(os.path.dirname(path), OUTPUT_DIR)
        preds[name] = ev.select("no").join(pl.read_csv(path, schema_overrides={"knowledge": pl.String}), on="no", how="left")
    rows, detail = [], {}
    for name, p in preds.items():
        for split, part in [("test", None), ("test", "old"), ("test", "rare"), ("dev", None)]:
            m = (ev["split"] == split) & (pl.lit(True) if part is None else (ev["part"] == part))
            m = ev.select(m.alias("m"))["m"]
            s = scores(ev.filter(m), p.filter(m))
            key = f"{split}{'' if part is None else '_' + part}"
            detail[f"{name}|{key}"] = s
            rows.append({"method": name, "split": key, **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    os.makedirs(FIG, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(FIG, "summary.csv"))
    json.dump(detail, open(os.path.join(FIG, "per_class.json"), "w"), indent=1)
    g = df.filter(pl.col("split") == "test").with_columns(pl.col("method").str.replace(r"/\d+$", "").alias("m")).group_by("m").agg(
        [pl.col(c).mean() for c in ["voice6_f1", "voice5_SM_f1", "voice3_f1", "know4_f1", "know3_0G_f1", "know2_O_f1", "know2_IO_f1",
                                    "uses_text_f1"]])
    with pl.Config(tbl_rows=80, tbl_cols=12, tbl_width_chars=250, float_precision=3):
        print(g.sort("voice6_f1", descending=True))


if __name__ == "__main__":
    main()
