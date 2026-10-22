"""お題画像の種別判定の手法を実行し，評価用ラベルに対する精度と処理速度を記録する．

結果は outputs/ex001_odai_image_type/{手法}/{設定}/{seed}/ に保存する．
- predictions.csv: 評価用画像ごとの予測（形式，画風，7クラス）
- metrics.json: 精度の指標（学習を伴う手法は学習用データに対する指標も含む）
- meta.json: 実験条件と処理速度

実行例:
    .venv_annotate/bin/python programs/ex001_odai_image_type/train.py --methods zeroshot probe_weak probe_hand
    .venv_annotate/bin/python programs/ex001_odai_image_type/train.py --methods vlm --vlm_models Qwen/Qwen3-VL-8B-Instruct
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import (DATASET_DIR, DOMAIN_LABELS, FLAT_LABELS, FORMAT_LABELS, PROJECT_DIR, image_path,  # noqa: E402
                  load_eval_labels, to_flat_label)
from machine_learning_utils import set_seed  # noqa: E402
from model import (DOMAIN_PROMPTS, FORMAT_PROMPTS, EmbeddingModel, LinearProbe, VLMClassifier,  # noqa: E402
                   ZeroShotClassifier, flat_to_axes)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex001_odai_image_type")
EMB_DIR = os.path.join(DATASET_DIR, "embeddings")
TOTAL_IMAGES = 6_823_270  # BoketeDBv4 の画像の数（処理時間の見積もりに用いる）

# 比較する埋め込みモデル
EMBEDDING_MODELS = ["openai/clip-vit-base-patch32", "google/siglip2-base-patch16-224", "google/siglip2-so400m-patch14-384"]
# 比較する VLM
VLM_MODELS = ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"]
# 線形プローブの正則化の探索空間と seed 数
PROBE_C_LIST = [0.01, 0.1, 1.0]
SEEDS = [0, 1, 2]
EMB_BATCH = 64
LOAD_WORKERS = 32


def short_name(model_name: str) -> str:
    """モデル名からディレクトリ名に用いる短い名前を作る．

    Args:
        model_name (str): Hugging Face Hub 上のモデル名．

    Returns:
        str: "/" を含まない短い名前．
    """
    return model_name.split("/")[-1]


def load_image(odai_id: int) -> Image.Image:
    """お題IDの画像を RGB で読み込む．

    Args:
        odai_id (int): お題ID．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(odai_id)) as im:
        return im.convert("RGB")


def load_images(ids: list) -> tuple:
    """画像をスレッドで並列に読み込む．

    Args:
        ids (list[int]): お題IDのリスト（長さ N）．

    Returns:
        tuple[list[PIL.Image.Image], float]: 画像のリストと，読み込みに要した秒数．
    """
    start = time.perf_counter()
    with ThreadPoolExecutor(LOAD_WORKERS) as ex:
        images = list(ex.map(load_image, ids))
    return images, time.perf_counter() - start


def compute_embeddings(embedder: EmbeddingModel, ids: list, desc: str) -> tuple:
    """画像の埋め込みをミニバッチごとに計算する．

    Args:
        embedder (EmbeddingModel): 埋め込みモデル．
        ids (list[int]): お題IDのリスト（長さ N）．
        desc (str): 進捗バーの説明．

    Returns:
        tuple[np.ndarray, float, float]: 形状 (N, D) の埋め込み，読み込みの秒数，推論の秒数．
    """
    chunk = EMB_BATCH * 8
    embs, t_load, t_infer = [], 0.0, 0.0
    for s in tqdm(range(0, len(ids), chunk), desc=desc, leave=False):
        images, t = load_images(ids[s:s + chunk])
        t_load += t
        start = time.perf_counter()
        for b in range(0, len(images), EMB_BATCH):
            embs.append(embedder.encode_images(images[b:b + EMB_BATCH]))
        t_infer += time.perf_counter() - start
    return np.concatenate(embs), t_load, t_infer


def cached_embeddings(embedder: EmbeddingModel, name: str, ids: list) -> tuple:
    """埋め込みをキャッシュから読み込み，なければ計算して保存する．

    Args:
        embedder (EmbeddingModel): 埋め込みモデル．
        name (str): 画像の集合の名前（"eval" または "weak"）．
        ids (list[int]): お題IDのリスト．

    Returns:
        tuple[np.ndarray, dict]: 埋め込みと，処理時間の記録（キャッシュから読んだ場合は保存時の記録）．
    """
    d = os.path.join(EMB_DIR, short_name(embedder.model_name))
    os.makedirs(d, exist_ok=True)
    npy, js = os.path.join(d, f"{name}.npy"), os.path.join(d, f"{name}_timing.json")
    if os.path.exists(npy) and os.path.exists(js):
        return np.load(npy), json.load(open(js))
    emb, t_load, t_infer = compute_embeddings(embedder, ids, f"{short_name(embedder.model_name)}:{name}")
    timing = {"n_images": len(ids), "load_sec": t_load, "infer_sec": t_infer}
    np.save(npy, emb)
    json.dump(timing, open(js, "w"), indent=4)
    return emb, timing


def evaluate(labels: pl.DataFrame, pred_format: list, pred_domain: list, pred_flat: list) -> dict:
    """予測を評価用ラベルと比較して指標を求める．

    画風の指標は，予測の画風が None でない画像のみで求める．

    Args:
        labels (pl.DataFrame): load_eval_labels の戻り値（行数 N）．
        pred_format (list[str]): 長さ N の形式の予測．
        pred_domain (list[str | None]): 長さ N の画風の予測．
        pred_flat (list[str]): 長さ N の7クラスの予測．

    Returns:
        dict: 7クラス・形式・画風の正解率とマクロ F1，および7クラスのクラスごとの適合率・再現率・F1．
    """
    y_flat, y_fmt, y_dom = labels["flat"].to_list(), labels["format"].to_list(), labels["domain"].to_list()
    dom_mask = [p is not None for p in pred_domain]
    y_dom_m = [y for y, m in zip(y_dom, dom_mask) if m]
    p_dom_m = [p for p in pred_domain if p is not None]
    report = classification_report(y_flat, pred_flat, labels=FLAT_LABELS, output_dict=True, zero_division=0)
    return {
        "flat_accuracy": accuracy_score(y_flat, pred_flat),
        "flat_macro_f1": f1_score(y_flat, pred_flat, labels=FLAT_LABELS, average="macro", zero_division=0),
        "format_accuracy": accuracy_score(y_fmt, pred_format),
        "format_macro_f1": f1_score(y_fmt, pred_format, labels=FORMAT_LABELS, average="macro", zero_division=0),
        "domain_accuracy": accuracy_score(y_dom_m, p_dom_m) if p_dom_m else None,
        "domain_macro_f1": f1_score(y_dom_m, p_dom_m, labels=DOMAIN_LABELS, average="macro", zero_division=0)
        if p_dom_m else None,
        "domain_coverage": float(np.mean(dom_mask)),
        "per_class": {c: report[c] for c in FLAT_LABELS},
    }


def save_run(run_dir: str, labels: pl.DataFrame, pred_format: list, pred_domain: list, pred_flat: list,
             meta: dict, extra_metrics: dict = None, extra_columns: dict = None) -> dict:
    """予測，指標，メタデータを保存する．

    Args:
        run_dir (str): 保存先のディレクトリ．
        labels (pl.DataFrame): 評価用ラベル（行数 N）．
        pred_format (list[str]): 長さ N の形式の予測．
        pred_domain (list[str | None]): 長さ N の画風の予測．
        pred_flat (list[str]): 長さ N の7クラスの予測．
        meta (dict): 実験条件と処理速度．
        extra_metrics (dict): 追加の指標（学習用データに対する指標など）．
        extra_columns (dict[str, list]): predictions.csv に追加する列．

    Returns:
        dict: 評価用データに対する指標．
    """
    os.makedirs(run_dir, exist_ok=True)
    metrics = {"eval": evaluate(labels, pred_format, pred_domain, pred_flat)}
    metrics.update(extra_metrics or {})
    cols = {"fileId": labels["fileId"], "pool": labels["pool"], "true_format": labels["format"],
            "true_domain": labels["domain"], "true_flat": labels["flat"], "pred_format": pred_format,
            "pred_domain": pred_domain, "pred_flat": pred_flat}
    cols.update(extra_columns or {})
    pl.DataFrame(cols).write_csv(os.path.join(run_dir, "predictions.csv"))
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=4, ensure_ascii=False)
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=4, ensure_ascii=False)
    m = metrics["eval"]
    print(f"  {run_dir.replace(OUTPUT_DIR + '/', '')}: flat acc={m['flat_accuracy']:.3f} "
          f"macroF1={m['flat_macro_f1']:.3f} format acc={m['format_accuracy']:.3f}")
    return metrics


def throughput_meta(timing: dict) -> dict:
    """処理時間の記録から，処理速度と全画像の処理に要する時間の見積もりを求める．

    Args:
        timing (dict): キー n_images，load_sec，infer_sec を持つ辞書．

    Returns:
        dict: 1秒あたりの画像数と，全画像の処理に要する時間（時間単位）の見積もり．
    """
    ips = timing["n_images"] / timing["infer_sec"]
    return {**timing, "images_per_sec_infer": ips, "est_hours_all_infer": TOTAL_IMAGES / ips / 3600,
            "images_per_sec_load": timing["n_images"] / timing["load_sec"]}


def is_done(run_dir: str) -> bool:
    """正常終了した結果が存在するかを判定する．

    Args:
        run_dir (str): 結果のディレクトリ．

    Returns:
        bool: metrics.json が存在すれば真．
    """
    return os.path.exists(os.path.join(run_dir, "metrics.json"))


def run_zeroshot(labels: pl.DataFrame, embedder: EmbeddingModel) -> None:
    """埋め込みモデルによるゼロショット分類を実行する（決定的なため seed は 0 のみ）．

    Args:
        labels (pl.DataFrame): 評価用ラベル．
        embedder (EmbeddingModel): 埋め込みモデル．

    Returns:
        None
    """
    run_dir = os.path.join(OUTPUT_DIR, f"zeroshot_{short_name(embedder.model_name)}", "default", "0")
    if is_done(run_dir):
        return
    emb, timing = cached_embeddings(embedder, "eval", labels["fileId"].to_list())
    fmt = ZeroShotClassifier(embedder, FORMAT_PROMPTS).predict(emb)
    dom = ZeroShotClassifier(embedder, DOMAIN_PROMPTS).predict(emb)
    flat = [to_flat_label(f, d) for f, d in zip(fmt, dom)]
    meta = {"method": "zeroshot", "model": embedder.model_name, "format_prompts": FORMAT_PROMPTS,
            "domain_prompts": DOMAIN_PROMPTS, "throughput": throughput_meta(timing)}
    save_run(run_dir, labels, fmt, dom, flat, meta)


def run_probe_weak(labels: pl.DataFrame, embedder: EmbeddingModel) -> None:
    """メタデータの弱ラベルで学習した線形プローブを評価する．

    seed ごとに弱ラベルの学習用データの90%を無作為に選んで学習する．

    Args:
        labels (pl.DataFrame): 評価用ラベル．
        embedder (EmbeddingModel): 埋め込みモデル．

    Returns:
        None
    """
    weak = pl.read_csv(os.path.join(DATASET_DIR, "weak_train.csv"))
    emb_w, timing_w = cached_embeddings(embedder, "weak", weak["fileId"].to_list())
    emb_e, timing_e = cached_embeddings(embedder, "eval", labels["fileId"].to_list())
    y_w = np.array(weak["weak_label"].to_list())
    for C, seed in itertools.product(PROBE_C_LIST, SEEDS):
        run_dir = os.path.join(OUTPUT_DIR, f"probe_weak_{short_name(embedder.model_name)}", f"C{C}", str(seed))
        if is_done(run_dir):
            continue
        set_seed(seed)
        idx = np.random.default_rng(seed).permutation(len(y_w))[: int(len(y_w) * 0.9)]
        probe = LinearProbe(C=C, seed=seed).fit(emb_w[idx], y_w[idx])
        flat = probe.predict(emb_e)
        axes = [flat_to_axes(f) for f in flat]
        train_pred = probe.predict(emb_w[idx])
        extra = {"train": {"weak_accuracy": accuracy_score(y_w[idx], train_pred),
                           "weak_macro_f1": f1_score(y_w[idx], train_pred, average="macro")}}
        meta = {"method": "probe_weak", "model": embedder.model_name, "C": C, "seed": seed,
                "n_train": int(len(idx)), "throughput": throughput_meta(timing_e)}
        save_run(run_dir, labels, [a[0] for a in axes], [a[1] for a in axes], flat, meta, extra)


def run_probe_hand(labels: pl.DataFrame, embedder: EmbeddingModel) -> None:
    """目視のラベルによる5分割交差検証で線形プローブを評価する（少数の正解ラベルで学習した場合の参考値）．

    形式と画風の2軸それぞれに分類器を学習する．

    Args:
        labels (pl.DataFrame): 評価用ラベル．
        embedder (EmbeddingModel): 埋め込みモデル．

    Returns:
        None
    """
    emb, timing = cached_embeddings(embedder, "eval", labels["fileId"].to_list())
    y_fmt, y_dom = np.array(labels["format"].to_list()), np.array(labels["domain"].to_list())
    for C, seed in itertools.product(PROBE_C_LIST, SEEDS):
        run_dir = os.path.join(OUTPUT_DIR, f"probe_hand_{short_name(embedder.model_name)}", f"C{C}", str(seed))
        if is_done(run_dir):
            continue
        set_seed(seed)
        p_fmt, p_dom = np.empty(len(y_fmt), dtype=object), np.empty(len(y_fmt), dtype=object)
        train_acc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(emb, y_fmt):
            pf = LinearProbe(C=C, seed=seed).fit(emb[tr], y_fmt[tr])
            pd_ = LinearProbe(C=C, seed=seed).fit(emb[tr], y_dom[tr])
            p_fmt[te], p_dom[te] = pf.predict(emb[te]), pd_.predict(emb[te])
            train_acc.append(accuracy_score(y_fmt[tr], pf.predict(emb[tr])))
        flat = [to_flat_label(f, d) for f, d in zip(p_fmt, p_dom)]
        extra = {"train": {"format_accuracy_mean_over_folds": float(np.mean(train_acc))}}
        meta = {"method": "probe_hand_cv5", "model": embedder.model_name, "C": C, "seed": seed,
                "throughput": throughput_meta(timing)}
        save_run(run_dir, labels, list(p_fmt), list(p_dom), flat, meta, extra)


def run_vlm(labels: pl.DataFrame, model_name: str, max_pixels: int, prompt: str, label_weak: bool) -> None:
    """VLM による分類を実行する（温度 0 の貪欲生成で決定的なため seed は 0 のみ）．

    label_weak が真の場合，弱ラベルの学習用画像にも VLM のラベルを付与して保存する（蒸留に用いる）．

    Args:
        labels (pl.DataFrame): 評価用ラベル．
        model_name (str): VLM のモデル名．
        max_pixels (int): 入力画像の最大画素数．
        prompt (str): プロンプトの版（model.VLM_PROMPTS のキー）．
        label_weak (bool): 弱ラベルの学習用画像にもラベルを付与するか．

    Returns:
        None
    """
    run_dir = os.path.join(OUTPUT_DIR, f"vlm_{short_name(model_name)}", f"{prompt}_px{max_pixels}", "0")
    weak_out = os.path.join(DATASET_DIR, "vlm_labels", f"{short_name(model_name)}_{prompt}_px{max_pixels}.csv")
    if is_done(run_dir) and (not label_weak or os.path.exists(weak_out)):
        return
    clf = VLMClassifier(model_name, max_pixels=max_pixels, prompt_version=prompt)
    images, t_load = load_images(labels["fileId"].to_list())
    clf.predict(images[:16])  # ウォームアップ
    start = time.perf_counter()
    preds = clf.predict(images)
    t_infer = time.perf_counter() - start
    fmt, dom = [p["format"] for p in preds], [p["domain"] for p in preds]
    flat = [to_flat_label(f, d) for f, d in zip(fmt, dom)]
    timing = {"n_images": len(images), "load_sec": t_load, "infer_sec": t_infer}
    meta = {"method": "vlm", "model": model_name, "max_pixels": max_pixels, "prompt": prompt,
            "throughput": throughput_meta(timing)}
    save_run(run_dir, labels, fmt, dom, flat, meta, extra_columns={"raw": [p["raw"] for p in preds]})
    if label_weak:
        label_weak_set(clf, weak_out)


def label_weak_set(clf: VLMClassifier, out_path: str) -> None:
    """弱ラベルの学習用画像に VLM のラベルを付与して CSV に保存する．処理速度も併せて記録する．

    Args:
        clf (VLMClassifier): VLM の分類器．
        out_path (str): 出力先の CSV のパス．

    Returns:
        None
    """
    weak = pl.read_csv(os.path.join(DATASET_DIR, "weak_train.csv"))
    ids = weak["fileId"].to_list()
    rows, t_load, t_infer = [], 0.0, 0.0
    for s in tqdm(range(0, len(ids), 1024), desc="vlm:weak", leave=False):
        images, t = load_images(ids[s:s + 1024])
        t_load += t
        start = time.perf_counter()
        preds = clf.predict(images)
        t_infer += time.perf_counter() - start
        rows += [{"fileId": i, "format": p["format"], "domain": p["domain"]} for i, p in zip(ids[s:s + 1024], preds)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pl.DataFrame(rows).write_csv(out_path)
    timing = {"n_images": len(ids), "load_sec": t_load, "infer_sec": t_infer}
    json.dump(throughput_meta(timing), open(out_path.replace(".csv", "_timing.json"), "w"), indent=4)


def run_probe_vlm(labels: pl.DataFrame, embedder: EmbeddingModel, vlm_label_name: str) -> None:
    """VLM が付与したラベルで学習した線形プローブ（VLM から埋め込みモデルへの蒸留）を評価する．

    形式と画風の2軸それぞれに分類器を学習する．seed ごとに学習用データの90%を無作為に選ぶ．

    Args:
        labels (pl.DataFrame): 評価用ラベル．
        embedder (EmbeddingModel): 埋め込みモデル．
        vlm_label_name (str): datasets/.../vlm_labels/ 内の CSV のファイル名（拡張子なし）．

    Returns:
        None
    """
    vl = pl.read_csv(os.path.join(DATASET_DIR, "vlm_labels", f"{vlm_label_name}.csv"))
    weak = pl.read_csv(os.path.join(DATASET_DIR, "weak_train.csv"))
    assert weak["fileId"].to_list() == vl["fileId"].to_list()
    emb_w, _ = cached_embeddings(embedder, "weak", weak["fileId"].to_list())
    emb_e, timing_e = cached_embeddings(embedder, "eval", labels["fileId"].to_list())
    y_fmt, y_dom = np.array(vl["format"].to_list()), np.array(vl["domain"].to_list())
    for C, seed in itertools.product(PROBE_C_LIST, SEEDS):
        run_dir = os.path.join(OUTPUT_DIR, f"probe_vlm_{short_name(embedder.model_name)}",
                               f"{vlm_label_name}_C{C}", str(seed))
        if is_done(run_dir):
            continue
        set_seed(seed)
        idx = np.random.default_rng(seed).permutation(len(y_fmt))[: int(len(y_fmt) * 0.9)]
        pf = LinearProbe(C=C, seed=seed).fit(emb_w[idx], y_fmt[idx])
        pd_ = LinearProbe(C=C, seed=seed).fit(emb_w[idx], y_dom[idx])
        fmt, dom = pf.predict(emb_e), pd_.predict(emb_e)
        flat = [to_flat_label(f, d) for f, d in zip(fmt, dom)]
        extra = {"train": {"format_accuracy": accuracy_score(y_fmt[idx], pf.predict(emb_w[idx])),
                           "domain_accuracy": accuracy_score(y_dom[idx], pd_.predict(emb_w[idx]))}}
        meta = {"method": "probe_vlm", "model": embedder.model_name, "teacher": vlm_label_name, "C": C,
                "seed": seed, "n_train": int(len(idx)), "throughput": throughput_meta(timing_e)}
        save_run(run_dir, labels, fmt, dom, flat, meta, extra)


def run_ocr(labels: pl.DataFrame) -> None:
    """YomiToku の OCR による文章のお題の判定を実行する（決定的なため seed は 0 のみ）．

    画風は判定できないため None とし，形式は normal と text の2値とする．

    Args:
        labels (pl.DataFrame): 評価用ラベル．

    Returns:
        None
    """
    from model import OcrTextDetector

    detector = OcrTextDetector()
    run_dir = os.path.join(OUTPUT_DIR, "ocr_yomitoku",
                           f"area{detector.area_threshold}_chars{detector.char_threshold}", "0")
    if is_done(run_dir):
        return
    images, t_load = load_images(labels["fileId"].to_list())
    detector.features(images[0])  # ウォームアップ
    start = time.perf_counter()
    feats = [detector.features(im) for im in tqdm(images, desc="ocr", leave=False)]
    t_infer = time.perf_counter() - start
    fmt = [detector.predict(f) for f in feats]
    timing = {"n_images": len(images), "load_sec": t_load, "infer_sec": t_infer}
    meta = {"method": "ocr_rule", "model": "yomitoku", "area_threshold": detector.area_threshold,
            "char_threshold": detector.char_threshold, "throughput": throughput_meta(timing)}
    extra = {"n_chars": [f["n_chars"] for f in feats], "area_ratio": [f["area_ratio"] for f in feats],
             "ocr_text": [f["text"] for f in feats]}
    save_run(run_dir, labels, fmt, [None] * len(fmt), fmt, meta, extra_columns=extra)


def main() -> None:
    """コマンドライン引数で指定した手法を順に実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["zeroshot", "probe_weak", "probe_hand"],
                        choices=["zeroshot", "probe_weak", "probe_hand", "vlm", "probe_vlm", "ocr"])
    parser.add_argument("--embedding_models", nargs="+", default=EMBEDDING_MODELS)
    parser.add_argument("--vlm_models", nargs="+", default=VLM_MODELS)
    parser.add_argument("--max_pixels", type=int, nargs="+", default=[448 * 448])
    parser.add_argument("--prompts", nargs="+", default=["v2"], help="VLM のプロンプトの版（model.VLM_PROMPTS のキー）")
    parser.add_argument("--label_weak", action="store_true", help="VLM で弱ラベルの学習用画像にもラベルを付与する")
    parser.add_argument("--vlm_label_name", default="Qwen3-VL-8B-Instruct_v2_px200704")
    args = parser.parse_args()
    labels = load_eval_labels()

    if "ocr" in args.methods:
        run_ocr(labels)
    vlm_conds = itertools.product(args.vlm_models, args.max_pixels, args.prompts) if "vlm" in args.methods else []
    for model_name, max_pixels, prompt in vlm_conds:
        run_vlm(labels, model_name, max_pixels, prompt, args.label_weak)

    emb_methods = [m for m in args.methods if m in ("zeroshot", "probe_weak", "probe_hand", "probe_vlm")]
    for model_name in args.embedding_models if emb_methods else []:
        embedder = EmbeddingModel(model_name)
        for m in emb_methods:
            print(f"[{m}] {model_name}")
            if m == "zeroshot":
                run_zeroshot(labels, embedder)
            elif m == "probe_weak":
                run_probe_weak(labels, embedder)
            elif m == "probe_hand":
                run_probe_hand(labels, embedder)
            else:
                run_probe_vlm(labels, embedder, args.vlm_label_name)
        del embedder


if __name__ == "__main__":
    main()
