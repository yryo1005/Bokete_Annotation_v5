"""プロンプトの部品（prompts.py）を Qwen3-VL-8B で画像の集合に適用し，出力を保存する．

モデルは1回だけ読み込み，指定した部品を順に適用する．出力は部品と画像の集合ごとに CSV に保存し，既にあれば省略する．
- 評価用データ（eval，640枚）：outputs/ex005_rare_weighted_distill/teacher_components/{部品}/eval.csv
- その他の集合（pool，rare_{n} など）：datasets/ex005_rare_weighted_distill/component_labels/{部品}_{集合}.csv
処理速度は同じ場所の _timing.json に保存する．

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/teacher.py --components F_v6 J_v6 --sets eval
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/teacher.py --components F_v6 J_v6 --sets pool
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, throughput  # noqa: E402
from data import DATASET_DIR, image_path, load_eval_labels, load_pool  # noqa: E402
from prompts import COMPONENTS, json_schema, parse  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
CHUNK = 2048


def output_path(component: str, set_name: str) -> str:
    """部品と集合の出力の CSV のパスを返す．

    Args:
        component (str): 部品の名前．
        set_name (str): 集合の名前．

    Returns:
        str: CSV のパス．
    """
    if set_name == "eval":
        return os.path.join(OUTPUT_DIR, "teacher_components", component, "eval.csv")
    return os.path.join(DATASET_DIR, "component_labels", f"{component}_{set_name}.csv")


def load_ids(set_name: str) -> list:
    """集合の名前からお題IDのリストを得る．

    Args:
        set_name (str): "eval"，"pool"，または datasets/.../candidates/{集合}.csv の名前．

    Returns:
        list[int]: お題IDのリスト．
    """
    if set_name == "eval":
        return load_eval_labels()["fileId"].to_list()
    if set_name == "pool":
        return load_pool()["fileId"].to_list()
    return pl.read_csv(os.path.join(DATASET_DIR, "candidates", f"{set_name}.csv"))["fileId"].to_list()


def build_llm(max_pixels: int):
    """vLLM のモデルを読み込む．

    Args:
        max_pixels (int): 入力画像の最大画素数（部品の中の最大値）．

    Returns:
        vllm.LLM: 読み込んだモデル．
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM

    return LLM(model=MODEL_NAME, max_model_len=6144, gpu_memory_utilization=0.88, seed=0,
               limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs={"max_pixels": max_pixels})


def _load(fid: int) -> Image.Image:
    """画像を RGB で読み込む．

    Args:
        fid (int): お題ID．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(fid)) as im:
        return im.convert("RGB")


def run_component(llm, component: str, ids: list, desc: str) -> tuple:
    """1つの部品を画像のリストに適用する．

    Args:
        llm (vllm.LLM): モデル．
        component (str): 部品の名前．
        ids (list[int]): お題IDのリスト．
        desc (str): 進捗バーの説明．

    Returns:
        tuple[pl.DataFrame, float]: 列 fileId と部品の項目を持つ表と，推論の秒数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    spec = COMPONENTS[component]
    params = SamplingParams(temperature=0, max_tokens=600,
                            structured_outputs=StructuredOutputsParams(json=json_schema(component)))
    mm = {"max_pixels": spec["max_pixels"]}
    recs, sec = [], 0.0
    for s in tqdm(range(0, len(ids), CHUNK), desc=desc, leave=False):
        with ThreadPoolExecutor(32) as ex:
            images = list(ex.map(_load, ids[s:s + CHUNK]))
        convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                                {"type": "text", "text": spec["prompt"]}]}] for im in images]
        start = time.perf_counter()
        outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs=mm)
        sec += time.perf_counter() - start
        for o in outs:
            try:
                recs.append(parse(component, json.loads(o.outputs[0].text)))
            except json.JSONDecodeError:
                recs.append(parse(component, {}))
    return pl.DataFrame(recs).with_columns(pl.Series("fileId", ids)), sec


def main() -> None:
    """指定した部品を指定した集合に適用して保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", nargs="+", default=list(COMPONENTS))
    parser.add_argument("--sets", nargs="+", default=["eval"])
    args = parser.parse_args()
    todo = [(c, s) for s in args.sets for c in args.components if not os.path.exists(output_path(c, s))]
    if not todo:
        return
    llm = build_llm(max(COMPONENTS[c]["max_pixels"] for c, _ in todo))
    run_component(llm, todo[0][0], load_ids("eval")[:16], "warmup")
    for component, set_name in todo:
        ids = load_ids(set_name)
        df, sec = run_component(llm, component, ids, f"{component}:{set_name}")
        path = output_path(component, set_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.write_csv(path)
        json.dump(throughput(len(ids), sec), open(path.replace(".csv", "_timing.json"), "w"), indent=2)
        print(f"{component} {set_name}: {len(ids) / sec:.1f} 枚/秒")


if __name__ == "__main__":
    main()
