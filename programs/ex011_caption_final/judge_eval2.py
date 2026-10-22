"""評価用データ240枚（ex010 の120枚と追加の120枚）で，候補の方法のキャプションを Qwen3-VL-32B で採点する．

採点の指示と項目は ex010 の judge.py と同じである．ex010 で採点済みの120枚の結果は再利用し，追加の120枚のみ採点する
（ex010 で未採点の方法は120枚も採点する）．

出力：outputs/ex011_caption_final/{方法}/judge_eval2.csv（追加の120枚），judge_eval.csv（ex010 の120枚．未採点の場合のみ）

実行例:
    .venv_annotate/bin/python programs/ex011_caption_final/judge_eval2.py
"""
import importlib.util
import json
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX10 = os.path.join(PROJECT_DIR, "programs", "ex010_photo_caption")
sys.path.insert(0, EX10)
from data import OUTPUT_DIR as EX10_OUT, eval_ids, load_references  # noqa: E402
from generate import _load, build_llm  # noqa: E402

J = importlib.util.spec_from_file_location("ex010_judge", os.path.join(EX10, "judge.py"))
judge_mod = importlib.util.module_from_spec(J)
J.loader.exec_module(judge_mod)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex011_caption_final")
METHODS = ["q30a3_448_short", "q8_448_short", "q30a3_256_short", "q30a3_448_strict", "q30a3_448_short_bi"]


def load_refs2() -> dict:
    """追加の評価用データの参照キャプションを読み込む．

    Returns:
        dict[int, str]: お題ID → 参照キャプション．
    """
    out = {}
    for line in open(os.path.join(THIS_DIR, "reference_captions_eval2.txt"), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            fid, cap = line.rstrip("\n").split("\t")
            out[int(fid)] = cap
    return out


def main() -> None:
    """未採点の（方法，集合）を全て採点する．

    Returns:
        None
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    sets = {"eval2": load_refs2(), "eval": load_references()}
    todo = []
    for m in METHODS:
        for name in ["eval2", "eval"]:
            done = os.path.join(EX10_OUT, m, f"judge_{name}.csv")
            if not os.path.exists(done) and not os.path.exists(os.path.join(OUTPUT_DIR, m, f"judge_{name}.csv")):
                todo.append((m, name))
    if not todo:
        return
    llm = build_llm(judge_mod.JUDGE_MODEL, 448 * 448)
    params = SamplingParams(temperature=0, max_tokens=100, structured_outputs=StructuredOutputsParams(json=judge_mod.SCHEMA))
    for m, name in todo:
        refs = sets[name]
        caps = pl.read_csv(os.path.join(EX10_OUT, m, f"captions_{name}.csv"))
        cap_of = dict(zip(caps["fileId"].to_list(), caps["caption"].fill_null("").to_list()))
        ids = list(refs)
        convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": _load(i)}, {"type": "text",
                  "text": judge_mod.JUDGE_PROMPT.format(ref=refs[i], cand=cap_of[i])}]}] for i in ids]
        outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": 448 * 448})
        recs = []
        for i, o in zip(ids, outs):
            try:
                r = json.loads(o.outputs[0].text)
            except json.JSONDecodeError:
                r = {"subject": False, "scene": False, "hallucination": True, "score": 1}
            recs.append({"fileId": i, **r})
        d = os.path.join(OUTPUT_DIR, m)
        os.makedirs(d, exist_ok=True)
        pl.DataFrame(recs).write_csv(os.path.join(d, f"judge_{name}.csv"))
        print(m, name, f"score={pl.DataFrame(recs)['score'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
