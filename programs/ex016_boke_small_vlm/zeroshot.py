"""小さな VLM と推論（thinking）モデルに，指示だけでボケのラベル（発話の型6区分，画像外の知識）を判定させ，精度と速度を測る．

指示と入力は ex015 の one_kt_cap と同じとする（お題画像＋お題の形式・媒体・画像中の文字の有無・写真のキャプション＋ボケの文）．
お題画像は長辺 448 画素以下に縮小して全モデルに同じ画像を与える．
- json モード：JSON スキーマで出力を制約する（Instruct のモデル）．
- think モード：推論の後に JSON を書かせ，最後の JSON を読み取る（Thinking のモデル）．構造化出力の制約は推論を妨げるため用いない．

速度は vLLM（1 GPU，gpu_memory_utilization 0.85）で620件をまとめて推論したときの1秒あたりの件数とする．

出力：outputs/ex016_boke_small_vlm/zeroshot/{モデル}/eval.csv（列 no, voice, outside, ok），speed.json

実行例:
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/zeroshot.py --models q2 q4 iv2
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX15_DIR, OUTPUT_DIR, _load, image_path, load_eval  # noqa: E402

T15 = _load("ex015_teacher_for_v4", os.path.join(EX15_DIR, "teacher.py"))
# モデル名 → (Hugging Face の ID, モード, 追加の vLLM の引数)
MODELS = {
    "q2": ("Qwen/Qwen3-VL-2B-Instruct", "json", {}),
    "q4": ("Qwen/Qwen3-VL-4B-Instruct", "json", {}),
    "q8": ("Qwen/Qwen3-VL-8B-Instruct", "json", {}),
    "iv1": ("OpenGVLab/InternVL3_5-1B-HF", "json", {}),
    "iv2": ("OpenGVLab/InternVL3_5-2B-HF", "json", {}),
    "iv4": ("OpenGVLab/InternVL3_5-4B-HF", "json", {}),
    "smol2": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "json", {}),
    "lfm16": ("LiquidAI/LFM2-VL-1.6B", "json", {}),
    "lfm3": ("LiquidAI/LFM2-VL-3B", "json", {}),
    "mcpm4": ("openbmb/MiniCPM-V-4", "json", {"trust_remote_code": True}),
    "gemma4": ("unsloth/gemma-3-4b-it", "json", {}),
    "heron2": ("turing-motors/Heron-NVILA-Lite-2B-hf", "json", {"trust_remote_code": True}),
    "q32": ("Qwen/Qwen3-VL-32B-Instruct-FP8", "json", {}),
    "q32t": ("Qwen/Qwen3-VL-32B-Thinking-FP8", "think", {}),
    "q30t": ("Qwen/Qwen3-VL-30B-A3B-Thinking-FP8", "think", {}),
}
VARIANT = "one_kt_cap"
THINK_TAIL = ("\n考えた後，最後に次のキーを持つ JSON を1つだけ出力してください："
              "voice（S/M/T/N/L），referent，referent_type（work/person/product/place/event/bokete/general/none），in_image，necessary．")


def load_small_image(fid: int, side: int = 448) -> Image.Image:
    """お題画像を読み込み，長辺を side 画素以下に縮小する．

    Args:
        fid (int): お題ID．
        side (int): 長辺の上限．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(fid)) as im:
        im = im.convert("RGB")
    im.thumbnail((side, side))
    if min(im.size) < 32:
        canvas = Image.new("RGB", (max(im.width, 32), max(im.height, 32)), "white")
        canvas.paste(im, (0, 0))
        im = canvas
    return im


def parse_last_json(text: str) -> dict:
    """文の最後の JSON を読み取る（推論の部分は除く）．

    Args:
        text (str): モデルの出力．

    Returns:
        dict: 読み取った辞書（失敗したら空）．
    """
    text = text.split("</think>")[-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def build_llm(key: str):
    """vLLM のモデルを読み込む．

    Args:
        key (str): MODELS のキー．

    Returns:
        vllm.LLM: 読み込んだモデル．
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM

    hf_id, mode, extra = MODELS[key]
    return LLM(model=hf_id, max_model_len=8192 if mode == "think" else 4096, gpu_memory_utilization=0.85, seed=0,
               limit_mm_per_prompt={"image": 1, "video": 0}, **extra)


def run(key: str, ev: pl.DataFrame) -> tuple:
    """検証データにラベルを付ける．

    Args:
        key (str): MODELS のキー．
        ev (pl.DataFrame): 検証データ．

    Returns:
        tuple[pl.DataFrame, float]: 予測の表と推論の秒数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    mode = MODELS[key][1]
    llm = build_llm(key)
    if mode == "json":
        params = SamplingParams(temperature=0, max_tokens=256, structured_outputs=StructuredOutputsParams(json=T15.SCHEMA_KT))
    else:
        params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, seed=0)
    rows = ev.to_dicts()
    with ThreadPoolExecutor(32) as ex:
        imgs = dict(zip([r["odaiId"] for r in rows], ex.map(load_small_image, [r["odaiId"] for r in rows])))
    convs = [[{"role": "user", "content": [
        {"type": "image_pil", "image_pil": imgs[r["odaiId"]]},
        {"type": "text", "text": T15.build_prompt(VARIANT, r["format"], r["text"], r) + (THINK_TAIL if mode == "think" else "")}]}]
        for r in rows]
    start = time.perf_counter()
    outs = llm.chat(convs, sampling_params=params, use_tqdm=False)
    sec = time.perf_counter() - start
    recs = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        rec = parse_last_json(text)
        d = T15.derive(VARIANT, rec, r["format"])
        recs.append({"no": r["no"], "voice": d["voice"], "outside": d["outside"], "ok": bool(rec), "n_tokens": len(o.outputs[0].token_ids)})
    return pl.DataFrame(recs), sec


def main() -> None:
    """指定したモデルで判定し，結果と速度を保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    args = parser.parse_args()
    ev = load_eval()
    for key in args.models:
        out = os.path.join(OUTPUT_DIR, "zeroshot", key)
        if os.path.exists(os.path.join(out, "eval.csv")):
            continue
        os.makedirs(out, exist_ok=True)
        pred, sec = run(key, ev)
        pred.write_csv(os.path.join(out, "eval.csv"))
        json.dump({"model": MODELS[key][0], "mode": MODELS[key][1], "n": ev.height, "sec": sec, "per_sec": ev.height / sec,
                   "parse_ok": float(pred["ok"].mean()), "mean_tokens": float(pred["n_tokens"].mean())},
                  open(os.path.join(out, "speed.json"), "w"), indent=1)
        print(key, f"{ev.height / sec:.1f} 件/秒", "parse_ok", round(float(pred["ok"].mean()), 3), flush=True)


if __name__ == "__main__":
    main()
