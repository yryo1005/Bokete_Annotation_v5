"""パッケージ（Bokete_Annotation_programs_v5.0）のボケの推論のコードで，検証データ620件の精度が再現されるかを確かめる．

パッケージの BokeClassifier（LoRA を統合）・ChunkDataset・predict_chunks（プレフィックスのキャッシュと長さによるまとめ）で
検証データを予測し，研究用のリポジトリの ex016 の予測（LoRA を統合しない通常の推論）と比べる．
また，合意ラベルでの F1 を求め，同じ入力を2回推論して出力が一致するか（再現性）も確かめる．

出力：outputs/ex019_boke_full/verify_eval.json，verify_eval_pred.csv

実行例:
    .venv_annotate/bin/python programs/ex019_boke_full/verify_eval.py
"""
import json
import os
import sys

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex019_boke_full")
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm"))
import data as D16  # noqa: E402
from bokete_annotation.boke import VOICES, BokeClassifier, ChunkDataset, collate, make_batches, predict_chunks  # noqa: E402
from bokete_annotation.boke_pipeline import chunk_table  # noqa: E402

REF = os.path.join(PROJECT_DIR, "outputs", "ex016_boke_small_vlm", "ft", "q4", "teacher_lr0.0001_ep2_bs16_r16__stage2", "0", "eval.csv")


def make_list(ev: pl.DataFrame, path: str) -> None:
    """検証データからパッケージの一覧（boke/list）と同じ形式の表を作る．

    Args:
        ev (pl.DataFrame): 検証データ．
        path (str): 保存先．

    Returns:
        None
    """
    ann = pl.read_parquet(os.path.join(PKG, "outputs", "BoketeDBv5_annotation", "annotation.parquet"), columns=["odaiId", "rep_odaiId"])
    img = pl.read_parquet(os.path.join(PKG, "outputs", "BoketeDBv5_annotation", "images.parquet"), columns=["odaiId", "path"])
    t = ev.select(pl.col("no").alias("id"), "odaiId", "format", "text").join(ann, on="odaiId", how="left")
    t = t.join(img.rename({"odaiId": "rep_odaiId"}), on="rep_odaiId", how="left").sort(["rep_odaiId", "format", "id"])
    t.select("id", "odaiId", "rep_odaiId", "format", "path", "text").write_parquet(path)


def run(model, list_path: str, cfg: dict) -> pl.DataFrame:
    """一覧の全てのボケを予測する．

    Args:
        model (BokeClassifier): 分類器．
        list_path (str): 一覧の parquet．
        cfg (dict): パッケージの config.json の boke の項目．

    Returns:
        pl.DataFrame: 列 id, voice, p_outside, p_voice_* の表．
    """
    ch = chunk_table(list_path, cfg["max_group"])
    chunks = np.c_[np.zeros(len(ch), dtype=np.int64), ch]
    batches = make_batches(chunks, chunks[:, 2] - chunks[:, 1], cfg["bokes_per_batch"])
    loader = torch.utils.data.DataLoader(ChunkDataset(chunks, [list_path], IMAGE_DIR, cfg["model"]), batch_sampler=batches,
                                         num_workers=4, collate_fn=collate, multiprocessing_context="spawn")
    ids, pvs, pos = [], [], []
    for batch in loader:
        i, v, o = predict_chunks(model, [g for g in batch if g["ok"]], sub_batch=cfg["sub_batch"])
        ids += i
        pvs.append(v)
        pos.append(o)
    pv, po = np.concatenate(pvs), np.concatenate(pos)
    return pl.DataFrame({"id": ids, "voice": np.array(VOICES)[pv.argmax(1)], "p_outside": po,
                         **{f"p_voice_{v}": pv[:, k] for k, v in enumerate(VOICES)}})


def main() -> None:
    """検証を行い，結果を保存する．

    Returns:
        None
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = json.load(open(os.path.join(PKG, "config.json"), encoding="utf-8"))["boke"]
    ev = D16.load_eval()
    list_path = os.path.join(OUTPUT_DIR, "eval_list.parquet")
    make_list(ev, list_path)
    model = BokeClassifier(cfg["model"], os.path.join(PKG, cfg["weights"])).cuda()
    p1 = run(model, list_path, cfg)
    p2 = run(model, list_path, cfg)
    p1.write_csv(os.path.join(OUTPUT_DIR, "verify_eval_pred.csv"))
    ref = pl.read_csv(REF).select(pl.col("no").alias("id"), pl.col("voice").alias("ref_voice"), pl.col("p_outside").alias("ref_po"))
    j = ev.select(pl.col("no").alias("id"), "split", pl.col("voice").alias("gold_voice"), pl.col("outside").alias("gold_outside"))
    j = j.join(p1.select("id", pl.col("voice").alias("pv"), "p_outside"), on="id").join(ref, on="id")
    d = np.abs(j["p_outside"].to_numpy() - j["ref_po"].to_numpy())
    res = {"n": j.height, "voice_agree_with_ex016": float((j["pv"] == j["ref_voice"]).mean()),
           "outside_agree_with_ex016": float(((j["p_outside"] > 0.5) == (j["ref_po"] > 0.5)).mean()),
           "p_outside_mean_abs_diff": float(d.mean()), "p_outside_max_abs_diff": float(d.max()),
           "rerun_identical": bool(p1.sort("id").equals(p2.sort("id"))),
           "rerun_max_abs_diff": float(np.abs(p1.sort("id")["p_outside"].to_numpy() - p2.sort("id")["p_outside"].to_numpy()).max())}
    for split in ["dev", "test"]:
        m = j.filter(pl.col("split") == split)
        res[f"{split}_voice6_f1"] = f1_score(m["gold_voice"], m["pv"], labels=VOICES, average="macro")
        res[f"{split}_outside_f1"] = f1_score(m["gold_outside"], m["p_outside"] > 0.5)
        res[f"{split}_voice6_f1_ex016"] = f1_score(m["gold_voice"], m["ref_voice"], labels=VOICES, average="macro")
        res[f"{split}_outside_f1_ex016"] = f1_score(m["gold_outside"], m["ref_po"] > 0.5)
    json.dump(res, open(os.path.join(OUTPUT_DIR, "verify_eval.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
