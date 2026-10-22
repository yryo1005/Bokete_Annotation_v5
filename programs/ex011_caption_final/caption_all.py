"""写真のお題画像のキャプションの本番用のスクリプト（Qwen3-VL-30B-A3B-Instruct（FP8），短い指示）．

対象は Bokete_Annotation_programs_v5.0 のアノテーション（読み込みのみ）で，媒体が写真のみの代表の画像（sha256 の重複を除く）．
SHARD 枚ごとに parts/part_XXXX.parquet に保存し，既にある部分は飛ばすため，中断しても同じコマンドで再開できる．
画像の読み込みは別のスレッドで次の束を先読みし，推論と重ねる．最後に全ての部分を captions.parquet にまとめ，
出力の検査（空，長すぎる，日本語以外，改行，打ち切り）の結果と処理速度を summary.json に保存する．

出力の列（captions.parquet）：odaiId（代表のお題ID），caption（日本語の1文．句読点を「、」「。」にそろえたもの），
caption_raw（モデルの出力そのもの），n_tokens（出力トークン数），finish（stop / length）
内容が同じ画像（sha256 が同じお題）には，annotation.parquet の rep_odaiId で結合して同じキャプションを付ける．

実行例:
    # 動作確認（無作為の2万枚）
    .venv_annotate/bin/python programs/ex011_caption_final/caption_all.py --sample 20000 --out_dir outputs/ex011_caption_final/test_20k
    # 本番（約337万枚）
    .venv_annotate/bin/python programs/ex011_caption_final/caption_all.py --out_dir outputs/ex011_caption_final/photo_captions
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ANN_PATH = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0", "outputs", "BoketeDBv5_annotation", "annotation.parquet")
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]

CONFIG = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "max_pixels": 448 * 448,
    "prompt": "この写真の内容を，日本語で30字程度の1文で簡潔に説明してください．人物の名前は書かないでください．説明の文のみを出力してください．",
    "max_tokens": 64,
    "temperature": 0.0,
    "seed": 0,
    # バッチの構成によらず同じ出力にする（vLLM の VLLM_BATCH_INVARIANT）．温度0でも，これがないと再実行で約半数の文が変わる
    "batch_invariant": True,
}
SHARD = 50_000
CHUNK = 2048
MAX_CHARS = 80


def image_path(odai_id: int) -> str:
    """お題IDから画像ファイルのパスを求める．

    Args:
        odai_id (int): お題ID．

    Returns:
        str: 画像ファイルの絶対パス．
    """
    return os.path.join(IMAGE_DIR, str(odai_id // 10000 * 10000), f"{odai_id}.jpg")


def load_image(fid: int) -> Image.Image:
    """画像を RGB で読み込む（極端に細い画像は白い余白を足して32画素以上にする）．

    Args:
        fid (int): お題ID．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(fid)) as im:
        im = im.convert("RGB")
    if min(im.size) < 32:
        canvas = Image.new("RGB", (max(im.width, 32), max(im.height, 32)), "white")
        canvas.paste(im, (0, 0))
        im = canvas
    return im


def target_ids(out_dir: str, sample: int = None) -> list:
    """対象の代表のお題IDを求めて targets.parquet に保存する（既にあれば読み込む）．

    Args:
        out_dir (str): 出力先のディレクトリ．
        sample (int): 動作確認用に無作為に抽出する枚数（None で全て）．

    Returns:
        list[int]: お題IDのリスト（昇順）．
    """
    path = os.path.join(out_dir, "targets.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)["odaiId"].to_list()
    ann = pl.scan_parquet(ANN_PATH).filter(
        (pl.col("odaiId") == pl.col("rep_odaiId")) & pl.col("photo")
        & ~pl.any_horizontal([pl.col(m) for m in MEDIA if m != "photo"])).select("odaiId").collect()
    if sample:
        ann = ann.sample(n=sample, seed=0)
    ann = ann.sort("odaiId")
    ann.write_parquet(path)
    return ann["odaiId"].to_list()


def normalize(text: str) -> str:
    """句読点を「、」「。」にそろえる（モデルは指示文の「，」「．」と「、」「。」を混ぜて出力するため）．

    Args:
        text (str): モデルの出力．

    Returns:
        str: 句読点をそろえた文．
    """
    return text.strip().replace("，", "、").replace("．", "。")


def check(caps: pl.DataFrame) -> dict:
    """出力の異常を数える．

    Args:
        caps (pl.DataFrame): 列 caption，finish を持つ表．

    Returns:
        dict: 項目ごとの件数と割合．
    """
    c = caps["caption"].fill_null("")
    n = caps.height
    ja = c.map_elements(lambda s: bool(re.search(r"[ぁ-んァ-ン一-龥]", s)), return_dtype=pl.Boolean)
    items = {"empty": int((c.str.len_chars() == 0).sum()), "too_long": int((c.str.len_chars() > MAX_CHARS).sum()),
             "not_japanese": int((~ja).sum()), "newline": int(c.str.contains("\n").sum()),
             "truncated": int((caps["finish"] == "length").sum()),
             "multi_sentence": int(c.str.count_matches("。").gt(1).sum())}
    return {"n": n, **{k: {"count": v, "rate": v / n} for k, v in items.items()},
            "mean_chars": float(c.str.len_chars().mean()), "p95_chars": float(c.str.len_chars().quantile(0.95))}


def run(out_dir: str, ids: list, shard: int = SHARD) -> None:
    """未処理のシャードにキャプションを付けて保存する．

    Args:
        out_dir (str): 出力先のディレクトリ．
        ids (list[int]): 対象のお題IDのリスト．
        shard (int): 1つの部分の枚数（再開の単位）．

    Returns:
        None
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if CONFIG["batch_invariant"] else "0"
    from vllm import LLM, SamplingParams

    d = os.path.join(out_dir, "parts")
    os.makedirs(d, exist_ok=True)
    todo = [k for k in range(0, len(ids), shard) if not os.path.exists(os.path.join(d, f"part_{k // shard:04d}.parquet"))]
    if not todo:
        return
    llm = LLM(model=CONFIG["model"], max_model_len=4096, gpu_memory_utilization=0.88, seed=CONFIG["seed"],
              limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs={"max_pixels": CONFIG["max_pixels"]})
    params = SamplingParams(temperature=CONFIG["temperature"], max_tokens=CONFIG["max_tokens"])
    pool = ThreadPoolExecutor(32)

    def load_chunk(chunk_ids: list) -> list:
        """束の画像を並列に読み込む．

        Args:
            chunk_ids (list[int]): お題IDのリスト．

        Returns:
            list[PIL.Image.Image]: 画像のリスト．
        """
        return list(pool.map(load_image, chunk_ids))

    loader = ThreadPoolExecutor(1)
    for k in tqdm(todo, desc="shard", leave=False):
        part = ids[k:k + shard]
        chunks = [part[s:s + CHUNK] for s in range(0, len(part), CHUNK)]
        recs = []
        start = time.perf_counter()
        fut = loader.submit(load_chunk, chunks[0])
        for j, chunk in enumerate(chunks):
            images = fut.result()
            if j + 1 < len(chunks):
                fut = loader.submit(load_chunk, chunks[j + 1])
            convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                                   {"type": "text", "text": CONFIG["prompt"]}]}] for im in images]
            outs = llm.chat(convs, sampling_params=params, use_tqdm=False,
                            mm_processor_kwargs={"max_pixels": CONFIG["max_pixels"]})
            for fid, o in zip(chunk, outs):
                out = o.outputs[0]
                recs.append({"odaiId": fid, "caption": out.text.strip(), "n_tokens": len(out.token_ids),
                             "finish": out.finish_reason})
        sec = time.perf_counter() - start
        pl.DataFrame(recs).write_parquet(os.path.join(d, f"part_{k // shard:04d}.parquet"))
        log = os.path.join(out_dir, "speed.json")
        sp = json.load(open(log)) if os.path.exists(log) else {"n": 0, "sec": 0.0}
        sp["n"] += len(part)
        sp["sec"] += sec
        sp["images_per_sec"] = sp["n"] / sp["sec"]
        json.dump(sp, open(log, "w"), indent=2)


def finalize(out_dir: str, ids: list) -> None:
    """部分をまとめ，検査の結果を保存する．

    Args:
        out_dir (str): 出力先のディレクトリ．
        ids (list[int]): 対象のお題IDのリスト．

    Returns:
        None
    """
    caps = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(out_dir, "parts", "part_*.parquet")))])
    assert caps["odaiId"].to_list() == ids, "部分の並びが対象と一致しない"
    caps = caps.rename({"caption": "caption_raw"}).with_columns(
        pl.col("caption_raw").map_elements(normalize, return_dtype=pl.String).alias("caption")).select(
        "odaiId", "caption", "caption_raw", "n_tokens", "finish")
    caps.write_parquet(os.path.join(out_dir, "captions.parquet"))
    sp = json.load(open(os.path.join(out_dir, "speed.json")))
    summary = {"config": CONFIG, "n_targets": len(ids), "check": check(caps), "speed": sp,
               "est_hours_all_photos": 3_371_140 / sp["images_per_sec"] / 3600}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main() -> None:
    """対象を求め，キャプションを付け，まとめる．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--shard", type=int, default=SHARD, help="1つの部分の枚数（再開の単位．同じ出力先では変えないこと）")
    parser.add_argument("--stages", nargs="+", default=["run", "finalize"], choices=["run", "finalize"])
    parser.add_argument("--no_batch_invariant", action="store_true", help="再現性を捨てて約2割速くする")
    args = parser.parse_args()
    CONFIG["batch_invariant"] = not args.no_batch_invariant
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ids = target_ids(out_dir, args.sample)
    if "run" in args.stages:
        run(out_dir, ids, args.shard)
    if "finalize" in args.stages:
        finalize(out_dir, ids)


if __name__ == "__main__":
    main()
