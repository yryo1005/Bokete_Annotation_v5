"""VLM でお題画像（写真）のキャプションを生成する（方法ごとに1プロセスで実行する）．

方法（METHODS）は，モデル，入力の最大画素数，指示文の組み合わせである．評価用データ120枚と，速度の測定用の2,000枚に
キャプションを付け，outputs/ex010_photo_caption/{方法}/captions_{eval,speed}.csv と meta.json（処理速度）に保存する．

実行例:
    .venv_annotate/bin/python programs/ex010_photo_caption/generate.py --method q8_448_ja
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import throughput  # noqa: E402
from data import OUTPUT_DIR, eval_ids, image_path, make_splits  # noqa: E402

PROMPT_JA = ("この写真の概要を，日本語の1文（30〜60字）で説明してください．誰・何が，どこで，何をしているかを中心に書き，"
             "細部の描写や推測は省いてください．人物の名前や画像中の文字の内容は書かず，「男性」「女性」などと書いてください．"
             "説明の文のみを出力してください．")
PROMPT_JA_SHORT = "この写真の内容を，日本語で30字程度の1文で簡潔に説明してください．人物の名前は書かないでください．説明の文のみを出力してください．"
PROMPT_JA_STRICT = ("この写真に写っているものを，日本語で30字程度の1文で簡潔に説明してください．画像から確実に分かることだけを書き，"
                    "場所や状況を推測で補わないでください．人物の名前は書かないでください．説明の文のみを出力してください．")
PROMPT_EN = ("Describe this photo in one short English sentence (10-20 words): who or what is in it, where, and what they are doing. "
             "Do not name people and do not quote text in the image. Output only the sentence.")
MODELS = {"q8": "Qwen/Qwen3-VL-8B-Instruct", "q4": "Qwen/Qwen3-VL-4B-Instruct", "q2": "Qwen/Qwen3-VL-2B-Instruct",
          "q30a3": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "q32": "Qwen/Qwen3-VL-32B-Instruct-FP8"}
# 方法の名前 → (モデル，入力の最大画素数，指示文，最大の出力トークン数)
METHODS = {
    "q8_448_ja": ("q8", 448 * 448, PROMPT_JA, 96),
    "q8_256_ja": ("q8", 256 * 256, PROMPT_JA, 96),
    "q8_448_short": ("q8", 448 * 448, PROMPT_JA_SHORT, 64),
    "q8_448_en": ("q8", 448 * 448, PROMPT_EN, 48),
    "q4_448_ja": ("q4", 448 * 448, PROMPT_JA, 96),
    "q2_448_ja": ("q2", 448 * 448, PROMPT_JA, 96),
    "q30a3_448_ja": ("q30a3", 448 * 448, PROMPT_JA, 96),
    "q32_448_ja": ("q32", 448 * 448, PROMPT_JA, 96),
    # 詳しい指示は場所や行為の推測を誘い，誤りが増えた（採点）．短い指示を他のモデルでも確かめる
    "q4_448_short": ("q4", 448 * 448, PROMPT_JA_SHORT, 64),
    "q2_448_short": ("q2", 448 * 448, PROMPT_JA_SHORT, 64),
    "q30a3_448_short": ("q30a3", 448 * 448, PROMPT_JA_SHORT, 64),
    "q8_256_short": ("q8", 256 * 256, PROMPT_JA_SHORT, 64),
    # order_016：本番の方式（30B-A3B の短い指示）の最終確認のための条件
    "q30a3_256_short": ("q30a3", 256 * 256, PROMPT_JA_SHORT, 64),
    "q30a3_448_strict": ("q30a3", 448 * 448, PROMPT_JA_STRICT, 64),
    # 環境変数 VLLM_BATCH_INVARIANT=1 で実行し，バッチの構成によらず同じ出力にする条件（本番の候補）
    "q30a3_448_short_bi": ("q30a3", 448 * 448, PROMPT_JA_SHORT, 64),
}


def build_llm(model: str, max_pixels: int):
    """vLLM のモデルを読み込む．

    Args:
        model (str): Hugging Face のモデル名．
        max_pixels (int): 入力画像の最大画素数．

    Returns:
        vllm.LLM: 読み込んだモデル．
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM

    return LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.88, seed=0,
               limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs={"max_pixels": max_pixels})


def _load(fid: int) -> Image.Image:
    """画像を RGB で読み込む．

    Args:
        fid (int): お題ID．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(fid)) as im:
        im = im.convert("RGB")
    # 極端に細い画像（例：400×1 画素）は VLM の前処理で扱えないため，白い余白を足して32画素以上にする
    if min(im.size) < 32:
        canvas = Image.new("RGB", (max(im.width, 32), max(im.height, 32)), "white")
        canvas.paste(im, (0, 0))
        im = canvas
    return im


def caption(llm, ids: list, prompt: str, max_pixels: int, max_tokens: int, chunk: int = 2048) -> tuple:
    """画像のリストにキャプションを付ける．

    Args:
        llm (vllm.LLM): モデル．
        ids (list[int]): お題IDのリスト．
        prompt (str): 指示文．
        max_pixels (int): 入力画像の最大画素数．
        max_tokens (int): 最大の出力トークン数．
        chunk (int): 一度に推論する枚数．

    Returns:
        tuple[list[str], float, int]: キャプションのリスト，推論の秒数，出力トークン数の合計．
    """
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    caps, sec, n_tok = [], 0.0, 0
    for s in range(0, len(ids), chunk):
        with ThreadPoolExecutor(32) as ex:
            images = list(ex.map(_load, ids[s:s + chunk]))
        convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im}, {"type": "text", "text": prompt}]}]
                 for im in images]
        start = time.perf_counter()
        outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": max_pixels})
        sec += time.perf_counter() - start
        caps += [o.outputs[0].text.strip().replace("\n", " ") for o in outs]
        n_tok += sum(len(o.outputs[0].token_ids) for o in outs)
    return caps, sec, n_tok


def main() -> None:
    """指定した方法で評価用データと速度の測定用データにキャプションを付ける．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--no_speed", action="store_true", help="速度の測定用データを省く")
    parser.add_argument("--extra_set", default=None, help="追加でキャプションを付ける集合（datasets/.../{名前}.csv の fileId）")
    args = parser.parse_args()
    key, max_pixels, prompt, max_tokens = METHODS[args.method]
    out_dir = os.path.join(OUTPUT_DIR, args.method)
    os.makedirs(out_dir, exist_ok=True)
    llm = build_llm(MODELS[key], max_pixels)
    caption(llm, eval_ids()[:16], prompt, max_pixels, max_tokens)  # 準備運転
    sets = {"eval": eval_ids()}
    if not args.no_speed:
        sets["speed"] = make_splits().filter(pl.col("split") == "speed")["fileId"].to_list()
    if args.extra_set:
        from data import DATASET_DIR

        sets[args.extra_set] = pl.read_csv(os.path.join(DATASET_DIR, f"{args.extra_set}.csv"))["fileId"].to_list()
    meta = {"method": args.method, "model": MODELS[key], "max_pixels": max_pixels, "prompt": prompt, "max_tokens": max_tokens}
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = {**json.load(open(meta_path)), **meta}
    for name, ids in sets.items():
        caps, sec, n_tok = caption(llm, ids, prompt, max_pixels, max_tokens)
        pl.DataFrame({"fileId": ids, "caption": caps}).write_csv(os.path.join(out_dir, f"captions_{name}.csv"))
        meta[f"speed_{name}"] = {**throughput(len(ids), sec), "mean_output_tokens": n_tok / len(ids)}
        print(args.method, name, f"{len(ids) / sec:.1f} 枚/秒", flush=True)
    json.dump(meta, open(meta_path, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
