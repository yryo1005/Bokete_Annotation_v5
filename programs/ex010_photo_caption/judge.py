"""キャプションを Qwen3-VL-32B（FP8）で採点する（画像，参照キャプション，候補のキャプションを与える）．

採点の項目（JSON の構造化出力，温度0）：
- subject：主な被写体（人・動物・物）が正しいか
- scene：場所・行為・状況が正しいか（大きな誤りがないか）
- hallucination：画像にない物や誤った内容を含むか
- score：概要の説明としての総合点（1〜5）
参照キャプションそのもの（"reference"）も採点し，採点者の基準を確かめる．

出力：outputs/ex010_photo_caption/{方法}/judge_eval.csv

実行例:
    .venv_annotate/bin/python programs/ex010_photo_caption/judge.py
"""
import glob
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, eval_ids, load_references  # noqa: E402
from generate import _load, build_llm  # noqa: E402

JUDGE_MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
JUDGE_PROMPT = """You are evaluating a short caption of a photo. The caption is meant to give only an overview of the image
(who/what, where, doing what); missing small details is fine. A human-written reference caption is given as a hint, but judge
against the image itself.

Reference caption: {ref}
Candidate caption: {cand}

Answer with JSON only:
- subject: true if the main subject(s) (people, animals, objects) are correctly identified
- scene: true if the place, action or situation is correct (no major error)
- hallucination: true if the candidate states something that is not in the image or is wrong
- score: overall quality as an overview caption, 1 (wrong) to 5 (correct and sufficient). 5 = correct overview with no errors,
  4 = correct but missing an important element or with a minor inaccuracy, 3 = partly correct, 2 = mostly wrong, 1 = unrelated"""
SCHEMA = {"type": "object", "properties": {"subject": {"type": "boolean"}, "scene": {"type": "boolean"},
                                           "hallucination": {"type": "boolean"},
                                           "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
          "required": ["subject", "scene", "hallucination", "score"], "additionalProperties": False}


def targets() -> dict:
    """採点する方法と，評価用データのキャプションを集める．

    Returns:
        dict[str, dict[int, str]]: 方法の名前 → お題ID → キャプション（参照キャプション "reference" を含む）．
    """
    out = {"reference": load_references()}
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "captions_eval.csv"))):
        m = os.path.basename(os.path.dirname(p))
        if os.path.exists(os.path.join(os.path.dirname(p), "judge_eval.csv")):
            continue
        df = pl.read_csv(p)
        out[m] = dict(zip(df["fileId"].to_list(), df["caption"].fill_null("").to_list()))
    return out


def main() -> None:
    """未採点の方法を全て採点する．

    Returns:
        None
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    todo = targets()
    if os.path.exists(os.path.join(OUTPUT_DIR, "reference", "judge_eval.csv")):
        todo.pop("reference")
    if not todo:
        return
    llm = build_llm(JUDGE_MODEL, 448 * 448)
    params = SamplingParams(temperature=0, max_tokens=100, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    ids = eval_ids()
    refs = load_references()
    images = {i: _load(i) for i in ids}
    for method, caps in todo.items():
        convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": images[i]},
                                               {"type": "text", "text": JUDGE_PROMPT.format(ref=refs[i], cand=caps[i])}]}]
                 for i in ids]
        outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": 448 * 448})
        recs = []
        for i, o in zip(ids, outs):
            try:
                r = json.loads(o.outputs[0].text)
            except json.JSONDecodeError:
                r = {"subject": False, "scene": False, "hallucination": True, "score": 1}
            recs.append({"fileId": i, **r})
        d = os.path.join(OUTPUT_DIR, method)
        os.makedirs(d, exist_ok=True)
        pl.DataFrame(recs).write_csv(os.path.join(d, "judge_eval.csv"))
        df = pl.DataFrame(recs)
        print(method, f"score={df['score'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
