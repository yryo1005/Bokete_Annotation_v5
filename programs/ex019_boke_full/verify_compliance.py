"""パッケージの穴埋め・お題文の制約の順守の得点（boke_rules.knn_scores）が，ex014 の検証と同じ判別の性能を持つかを確かめる．

検証データの穴埋め・お題文のボケ（正）と，同じ形式の別のお題に付け替えた組（負．datasets/ex014_boke_labels_v2/swap_pairs.csv）について，
パッケージと同じ埋め込み（ruri-v3-310m，boke_pipeline.embed_texts）と得点の関数で，お題の他のボケとの類似度を求め，AUC を比べる．
カルタの規則（boke_rules.karuta_compliance）も，検証データのカルタのボケの目視ラベルと比べる．

出力：outputs/ex019_boke_full/verify_compliance.json

実行例:
    .venv_annotate/bin/python programs/ex019_boke_full/verify_compliance.py
"""
import json
import os
import sys

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex019_boke_full")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm"))
import data as D16  # noqa: E402
from bokete_annotation.boke_pipeline import embed_texts  # noqa: E402
from bokete_annotation.boke_rules import karuta_compliance, knn_scores  # noqa: E402


def score_one(text: str, others: list, odai: int, tok_model: tuple) -> float:
    """1件のボケについて，お題の他のボケとの得点を求める（パッケージと同じく自分を含めた群で計算し，自分の行を取る）．

    Args:
        text (str): ボケの文．
        others (list[str]): そのお題の他のボケの文．
        odai (int): お題 ID（乱数の種）．
        tok_model (tuple): 埋め込みのモデル．

    Returns:
        float: 得点．
    """
    emb = embed_texts([text] + others, 1024, tok_model).astype(np.float32)
    return float(knn_scores(emb, odai)[0])


def main() -> None:
    """検証を行い，結果を保存する．

    Returns:
        None
    """
    from transformers import AutoModel, AutoTokenizer

    cfg = json.load(open(os.path.join(PKG, "config.json"), encoding="utf-8"))["boke"]
    tm = (AutoTokenizer.from_pretrained(cfg["embedding"]), AutoModel.from_pretrained(cfg["embedding"], dtype=torch.float16).cuda().eval())
    ev = D16.load_eval(consensus=False)
    pos = ev.filter(pl.col("format").is_in(["anaume", "odaibun"])).sort("no")
    sw = pl.read_csv(os.path.join(PROJECT_DIR, "datasets", "ex014_boke_labels_v2", "swap_pairs.csv")).sort("no")
    ids = sorted(set(pos["odaiId"]) | set(sw["odaiId"]))
    bokes = D16.D15.load_bokes(ids, ("odaiId", "text")).with_columns(pl.col("text").fill_null(""))
    by = {int(k[0]): g["text"].to_list() for k, g in bokes.group_by("odaiId")}
    sp = [score_one(t, [x for x in by[f] if x != t], f, tm) for t, f in zip(pos["text"], pos["odaiId"])]
    sn = [score_one(t, [x for x in by[f] if x != t], f, tm) for t, f in zip(sw["text"], sw["odaiId"])]
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    s = np.r_[sp, sn]
    res = {"n_pos": len(sp), "n_neg": len(sn), "auc": roc_auc_score(y, s),
           "detected_at_95": float(np.mean(np.array(sn) < np.quantile(sp, 0.05)))}
    for fmt in ["anaume", "odaibun"]:
        m = (pos["format"] == fmt).to_numpy()
        mn = (sw["format"] == fmt).to_numpy() if "format" in sw.columns else m
        res[f"auc_{fmt}"] = roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(mn.sum())], np.r_[np.array(sp)[m], np.array(sn)[mn]])
    kar = ev.filter(pl.col("format") == "karuta")
    kb = D16.D15.load_bokes(sorted(set(kar["odaiId"])), ("odaiId", "text")).with_columns(pl.col("text").fill_null(""))
    pred = {}
    for (f,), g in kb.group_by("odaiId"):
        _, comp = karuta_compliance(g["text"].to_list())
        pred.update({(int(f), t): c for t, c in zip(g["text"].to_list(), comp)})
    gold = (kar["comp"] == "y").to_list()
    got = [pred.get((int(f), t), None) for f, t in zip(kar["odaiId"], kar["text"])]
    res["karuta_n"] = len(gold)
    res["karuta_accuracy"] = float(np.mean([g == p for g, p in zip(gold, got) if p is not None]))
    res["karuta_gold_violations"] = int(sum(not g for g in gold))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUTPUT_DIR, "verify_compliance.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
