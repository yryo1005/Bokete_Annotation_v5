"""教師となる VLM を比較し，教師ラベル用の画像（40,000枚）にラベルを付与する．

vLLM の chat API を用いるため，モデルごとのチャットテンプレートの違いを意識せずに同じプロンプト（ex001 の v2）で比較できる．
出力は正規表現による構造化出力で {"format": ..., "domain": ...} に制約する．

- 評価：outputs/ex002_odai_type_distill/teacher_{モデル}/v2/0/
- ラベル：datasets/ex002_odai_type_distill/teacher_labels/{モデル}.csv（と処理速度の _timing.json）
- 多数決：複数の教師のラベルを軸ごとに多数決したラベル（同数の場合は先頭の教師に従う）

実行例:
    .venv_annotate/bin/python programs/ex002_odai_type_distill/teacher.py --models Qwen/Qwen3-VL-8B-Instruct --label_pool
    .venv_annotate/bin/python programs/ex002_odai_type_distill/teacher.py --vote Qwen3-VL-8B-Instruct Qwen3-VL-30B-A3B-Instruct-FP8 InternVL3_5-8B
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, is_done, save_run, throughput  # noqa: E402
from data import DATASET_DIR, DOMAIN_LABELS, EX001_DIR, FORMAT_LABELS, image_path, load_eval_labels, load_pool  # noqa: E402

# ex001 の model.py からプロンプトと出力の解釈の関数を読み込む
_spec = importlib.util.spec_from_file_location("ex001_model", os.path.join(EX001_DIR, "model.py"))
ex001_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex001_model)

LABEL_DIR = os.path.join(DATASET_DIR, "teacher_labels")
MAX_PIXELS = 448 * 448
CHUNK = 2048
# モデルごとの vLLM の追加の設定
MODEL_KWARGS = {
    "OpenGVLab/InternVL3_5-8B": {"trust_remote_code": True},
    "openbmb/MiniCPM-V-4_5": {"trust_remote_code": True},
}


def build_llm(model_name: str, gpu_memory_utilization: float = 0.88):
    """vLLM のモデルを読み込む．

    Args:
        model_name (str): Hugging Face Hub 上のモデル名．
        gpu_memory_utilization (float): vLLM が確保する VRAM の割合．

    Returns:
        vllm.LLM: 読み込んだモデル．
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM

    kwargs = {"mm_processor_kwargs": {"max_pixels": MAX_PIXELS}} if "Qwen" in model_name else {}
    kwargs.update(MODEL_KWARGS.get(model_name, {}))
    return LLM(model=model_name, max_model_len=4096, gpu_memory_utilization=gpu_memory_utilization,
               limit_mm_per_prompt={"image": 1, "video": 0}, seed=0, **kwargs)


def sampling_params():
    """出力を2軸のラベルの JSON に制約する生成の設定を作る．

    Returns:
        vllm.SamplingParams: 温度0，最大32トークン，正規表現による構造化出力の設定．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    pattern = (r'\{"format": "(' + "|".join(FORMAT_LABELS) + r')", "domain": "(' + "|".join(DOMAIN_LABELS) + r')"\}')
    return SamplingParams(temperature=0, max_tokens=32, structured_outputs=StructuredOutputsParams(regex=pattern))


def _load(odai_id: int) -> Image.Image:
    """画像を RGB で読み込む．

    Args:
        odai_id (int): お題ID．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(odai_id)) as im:
        return im.convert("RGB")


def predict(llm, ids: list, desc: str) -> tuple:
    """画像の種別を VLM で予測する．読み込みと推論の時間を分けて測る．

    Args:
        llm (vllm.LLM): モデル．
        ids (list[int]): お題IDのリスト（長さ N）．
        desc (str): 進捗バーの説明．

    Returns:
        tuple[list[dict], float]: 長さ N の予測（キー format，domain，raw）と，推論に要した秒数．
    """
    params = sampling_params()
    preds, sec = [], 0.0
    for s in tqdm(range(0, len(ids), CHUNK), desc=desc, leave=False):
        with ThreadPoolExecutor(32) as ex:
            images = list(ex.map(_load, ids[s:s + CHUNK]))
        convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                                {"type": "text", "text": ex001_model.VLM_PROMPTS["v2"]}]}]
                 for im in images]
        start = time.perf_counter()
        outs = llm.chat(convs, sampling_params=params, use_tqdm=False)
        sec += time.perf_counter() - start
        preds += [ex001_model.parse_vlm_output(o.outputs[0].text) for o in outs]
    return preds, sec


def run_teacher(model_name: str, label_pool: bool) -> None:
    """1つの VLM で評価用データを予測し，必要に応じて教師ラベル用の画像にもラベルを付ける．

    Args:
        model_name (str): モデル名．
        label_pool (bool): 教師ラベル用の画像（40,000枚）にラベルを付けるか．

    Returns:
        None
    """
    name = model_name.split("/")[-1]
    run_dir = os.path.join(OUTPUT_DIR, f"teacher_{name}", "v2", "0")
    label_path = os.path.join(LABEL_DIR, f"{name}.csv")
    if is_done(run_dir) and (not label_pool or os.path.exists(label_path)):
        return
    labels = load_eval_labels()
    llm = build_llm(model_name)
    predict(llm, labels["fileId"].to_list()[:32], "warmup")
    preds, sec = predict(llm, labels["fileId"].to_list(), f"{name}:eval")
    meta = {"method": "teacher_vlm", "model": model_name, "prompt": "v2", "max_pixels": MAX_PIXELS,
            "throughput_eval": throughput(len(preds), sec)}
    if label_pool:
        ids = load_pool()["fileId"].to_list()
        pool_preds, pool_sec = predict(llm, ids, f"{name}:pool")
        os.makedirs(LABEL_DIR, exist_ok=True)
        pl.DataFrame({"fileId": ids, "format": [p["format"] for p in pool_preds],
                      "domain": [p["domain"] for p in pool_preds]}).write_csv(label_path)
        meta["throughput_pool"] = throughput(len(ids), pool_sec)
        json.dump(meta["throughput_pool"], open(label_path.replace(".csv", "_timing.json"), "w"), indent=4)
    save_run(run_dir, labels, [p["format"] for p in preds], [p["domain"] for p in preds], meta,
             extra_columns={"raw": [p["raw"] for p in preds]})


def majority(values: list) -> str:
    """多数決の結果を返す．同数の場合はリストの先頭に近い値を優先する．

    Args:
        values (list[str]): 教師の順に並んだラベル．

    Returns:
        str: 多数決のラベル．
    """
    counts = Counter(values)
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)


def run_vote(teachers: list) -> None:
    """複数の教師の評価用データの予測とラベルを軸ごとに多数決する．

    Args:
        teachers (list[str]): 教師の短い名前のリスト（先頭ほど同数の際に優先する）．

    Returns:
        None
    """
    name = "vote_" + "+".join(t.split("-Instruct")[0] for t in teachers)
    run_dir = os.path.join(OUTPUT_DIR, "teacher_vote", name, "0")
    labels = load_eval_labels()
    preds = [pl.read_csv(os.path.join(OUTPUT_DIR, f"teacher_{t}", "v2", "0", "predictions.csv")) for t in teachers]
    fmt = [majority(list(v)) for v in zip(*[p["pred_format"].to_list() for p in preds])]
    dom = [majority(list(v)) for v in zip(*[p["pred_domain"].to_list() for p in preds])]
    meta = {"method": "teacher_vote", "teachers": teachers}
    save_run(run_dir, labels, fmt, dom, meta)
    paths = [os.path.join(LABEL_DIR, f"{t}.csv") for t in teachers]
    if all(os.path.exists(p) for p in paths):
        labs = [pl.read_csv(p) for p in paths]
        out = pl.DataFrame({"fileId": labs[0]["fileId"],
                            "format": [majority(list(v)) for v in zip(*[l_["format"].to_list() for l_ in labs])],
                            "domain": [majority(list(v)) for v in zip(*[l_["domain"].to_list() for l_ in labs])]})
        out.write_csv(os.path.join(LABEL_DIR, f"{name}.csv"))


def main() -> None:
    """コマンドライン引数に従って教師の評価，ラベル付け，多数決を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--label_pool", action="store_true")
    parser.add_argument("--vote", nargs="*", default=[])
    args = parser.parse_args()
    for m in args.models:
        run_teacher(m, args.label_pool)
    if args.vote:
        run_vote(args.vote)


if __name__ == "__main__":
    main()
