"""英語のキャプション（q8_448_en）を，Qwen3-VL-8B の文章のみの入力で日本語に翻訳する（方法 q8_448_en_ja）．

出力：outputs/ex010_photo_caption/q8_448_en_ja/captions_{eval,speed}.csv，meta.json
速度は，英語のキャプションの生成と翻訳の合計の時間で求める．

実行例:
    .venv_annotate/bin/python programs/ex010_photo_caption/translate.py
"""
import json
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import throughput  # noqa: E402
from data import OUTPUT_DIR  # noqa: E402
from generate import MODELS, build_llm  # noqa: E402

PROMPT = "次の英文を，自然な日本語の1文に翻訳してください．翻訳文のみを出力してください．\n\n{en}"


def main() -> None:
    """英語のキャプションを翻訳して保存する．

    Returns:
        None
    """
    from vllm import SamplingParams

    src = os.path.join(OUTPUT_DIR, "q8_448_en")
    meta_src = json.load(open(os.path.join(src, "meta.json")))
    out = os.path.join(OUTPUT_DIR, "q8_448_en_ja")
    os.makedirs(out, exist_ok=True)
    llm = build_llm(MODELS["q8"], 448 * 448)
    params = SamplingParams(temperature=0, max_tokens=96)
    meta = {"method": "q8_448_en_ja", "source": "q8_448_en", "prompt": PROMPT}
    for name in ["eval", "speed"]:
        df = pl.read_csv(os.path.join(src, f"captions_{name}.csv"))
        convs = [[{"role": "user", "content": PROMPT.format(en=c)}] for c in df["caption"].fill_null("").to_list()]
        start = time.perf_counter()
        outs = llm.chat(convs, sampling_params=params, use_tqdm=False)
        sec = time.perf_counter() - start
        df.with_columns(pl.Series("caption", [o.outputs[0].text.strip().replace("\n", " ") for o in outs])).write_csv(
            os.path.join(out, f"captions_{name}.csv"))
        total = meta_src[f"speed_{name}"]["sec"] + sec
        meta[f"speed_{name}"] = {**throughput(len(df), total), "translate_sec": sec}
        print(name, f"翻訳 {len(df) / sec:.1f} 枚/秒，合計 {len(df) / total:.1f} 枚/秒", flush=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
