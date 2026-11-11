"""パッケージの predict_chunks の GPU の時間を，プレフィックスとボケの部分に分けて測る（最適化の判断のため）．

1シャードの先頭のバッチを予めワーカーで作っておき，GPU の処理だけを時間を区切って測る．

出力：標準出力

実行例:
    .venv_annotate/bin/python programs/ex019_boke_full/profile_predict.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "Bokete_Annotation_programs_v5.0"))
sys.path.insert(0, PKG)
import bokete_annotation.boke as B  # noqa: E402
from bokete_annotation.boke_pipeline import chunk_table  # noqa: E402

LIST = os.path.join(PKG, "outputs", "test_boke", "boke", "list", "part_0005.parquet")


def main() -> None:
    """GPU の時間の内訳を測る．

    Returns:
        None
    """
    cfg = json.load(open(os.path.join(PKG, "config.json"), encoding="utf-8"))["boke"]
    ch = chunk_table(LIST, cfg["max_group"])
    chunks = np.c_[np.zeros(len(ch), dtype=np.int64), ch]
    batches = B.make_batches(chunks, chunks[:, 2] - chunks[:, 1], cfg["bokes_per_batch"])[:40]
    loader = torch.utils.data.DataLoader(B.ChunkDataset(chunks, [LIST], "/mnt/d/Bokete_Scraping_v3.2/datasets/images", cfg["model"]),
                                         batch_sampler=batches, num_workers=8, collate_fn=B.collate, multiprocessing_context="spawn")
    data = [[g for g in b if g["ok"]] for b in loader]
    model = B.BokeClassifier(cfg["model"], os.path.join(PKG, cfg["weights"])).cuda()
    B.predict_chunks(model, data[0])
    timing = {"prefix": 0.0, "suffix": 0.0}
    orig_sub = B._sub_cache

    n = sum(len(g["ids"]) for b in data for g in b)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for b in data:
        B.predict_chunks(model, b, sub_batch=cfg["sub_batch"])
    torch.cuda.synchronize()
    total = time.perf_counter() - t
    # プレフィックスだけの時間（ボケの部分を0件にして測る）
    for b in data:
        for g in b:
            g["_suf"] = g["suffix_ids"]
            g["suffix_ids"] = g["suffix_ids"][:1]
    torch.cuda.synchronize()
    t = time.perf_counter()
    for b in data:
        B.predict_chunks(model, b, sub_batch=cfg["sub_batch"])
    torch.cuda.synchronize()
    pre_only = time.perf_counter() - t
    # get_rope_index（Python のループを含む）の時間
    t_rope = 0.0
    for b in data:
        plen = max(len(g["prefix_ids"]) for g in b)
        pre = torch.zeros((len(b), plen), dtype=torch.long, device="cuda")
        pm = torch.zeros_like(pre)
        mm = torch.zeros_like(pre)
        for i, g in enumerate(b):
            L = len(g["prefix_ids"])
            pre[i, plen - L:] = g["prefix_ids"].cuda()
            pm[i, plen - L:] = 1
            mm[i, plen - L:] = g["mm_token_type_ids"].cuda()
        thw = torch.cat([g["image_grid_thw"] for g in b]).cuda()
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.backbone.get_rope_index(pre, image_grid_thw=thw, attention_mask=pm, mm_token_type_ids=mm)
        torch.cuda.synchronize()
        t_rope += time.perf_counter() - t
    # 画像の符号化（視覚の部分）だけの時間
    t_vis = 0.0
    for b in data:
        pix = torch.cat([g["pixel_values"] for g in b]).cuda().to(torch.bfloat16)
        thw = torch.cat([g["image_grid_thw"] for g in b]).cuda()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.backbone.get_image_features(pix, thw)
        torch.cuda.synchronize()
        t_vis += time.perf_counter() - t
    plens = [len(g["prefix_ids"]) for b in data for g in b]
    pad = np.mean([1 - np.mean([len(g["prefix_ids"]) for g in b]) / max(len(g["prefix_ids"]) for g in b) for b in data])
    print(json.dumps({"n_bokes": n, "total_sec": total, "per_sec": n / total, "prefix_only_sec_approx": pre_only, "rope_index_sec": t_rope, "vision_sec": t_vis,
                      "prefix_len_mean": float(np.mean(plens)), "prefix_len_min": int(min(plens)), "prefix_len_max": int(max(plens)),
                      "prefix_pad_rate": float(pad), "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}, indent=1))


if __name__ == "__main__":
    main()
