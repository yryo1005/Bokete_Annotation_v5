"""実データの無作為の標本で，パッケージの高速な推論（LoRA の統合＋プレフィックスのキャッシュ）と，
研究用のリポジトリの通常の推論（ex016．LoRA を統合しない1件ごとの推論）の一致を確かめる．

パッケージで予測済みのシャード（outputs/test_boke/boke/predict/part_0001〜0002）から無作為に3,000件を選び，
ex016 の通常の推論で予測し直して，発話の型・画像外の知識の判定の一致率と確率の差を求める．

出力：outputs/ex019_boke_full/verify_sample.json

実行例:
    .venv_annotate/bin/python programs/ex019_boke_full/verify_sample.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex019_boke_full")
W = os.path.join(PROJECT_DIR, "outputs", "ex016_boke_small_vlm", "ft", "q4", "teacher_lr0.0001_ep2_bs16_r16__stage2", "0", "best_model.pt")
VOICES = ["S", "M", "T", "N", "L", "A"]


def main() -> None:
    """一致を確かめて保存する．

    Returns:
        None
    """
    sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm"))
    import model as M16  # noqa: E402
    import train as T16  # noqa: E402

    src = os.path.join(PKG, "outputs", "test_boke", "boke")
    lst = pl.concat([pl.read_parquet(os.path.join(src, "list", f"part_{i:04d}.parquet")) for i in (1, 2)])
    pred = pl.concat([pl.read_parquet(os.path.join(src, "predict", f"part_{i:04d}.parquet")) for i in (1, 2)])
    df = lst.join(pred, on="id").filter(pl.col("image_ok")).sample(n=3000, seed=0).sort("id")
    model = M16.load_model("q4", W, seed=0, gradient_checkpointing=False).cuda()
    inp = df.select("odaiId", "text", "format", pl.lit("S").alias("voice"), pl.lit(False).alias("outside"))
    pv, po, _ = T16.predict(model, T16.make_loader(inp, "q4", 32, False, 0))
    fast_v = np.array(VOICES)[df.select([f"p_voice_{v}" for v in VOICES]).to_numpy().argmax(1)]
    ref_v = np.array(VOICES)[pv.argmax(1)]
    fo = df["p_outside"].to_numpy()
    d = np.abs(fo - po)
    res = {"n": df.height, "voice_agree": float((fast_v == ref_v).mean()), "outside_agree": float(((fo > 0.5) == (po > 0.5)).mean()),
           "p_outside_mean_abs_diff": float(d.mean()), "p_outside_p99_abs_diff": float(np.quantile(d, 0.99)),
           "p_outside_max_abs_diff": float(d.max()),
           "p_voice_mean_abs_diff": float(np.abs(df.select([f"p_voice_{v}" for v in VOICES]).to_numpy() - pv).mean())}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUTPUT_DIR, "verify_sample.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
