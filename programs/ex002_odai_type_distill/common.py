"""ex002 の各スクリプトが共有する評価と結果の保存の処理．"""
import json
import os

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data import DOMAIN_LABELS, FLAT_LABELS, FORMAT_LABELS, PROJECT_DIR

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex002_odai_type_distill")
TOTAL_IMAGES = 6_823_270


def evaluate(true_fmt: list, true_dom: list, pred_fmt: list, pred_dom: list) -> dict:
    """2軸の予測を正解と比較して指標を求める．

    Args:
        true_fmt (list[str]): 長さ N の形式の正解．
        true_dom (list[str]): 長さ N の画風の正解．
        pred_fmt (list[str]): 長さ N の形式の予測．
        pred_dom (list[str]): 長さ N の画風の予測．

    Returns:
        dict: 7クラス（flat）・形式・画風の正解率とマクロ F1，7クラスのクラスごとの適合率・再現率・F1．
    """
    flat = lambda f, d: d if f == "normal" else f  # noqa: E731
    y = [flat(f, d) for f, d in zip(true_fmt, true_dom)]
    p = [flat(f, d) for f, d in zip(pred_fmt, pred_dom)]
    rep = classification_report(y, p, labels=FLAT_LABELS, output_dict=True, zero_division=0)
    return {
        "flat_accuracy": accuracy_score(y, p),
        "flat_macro_f1": f1_score(y, p, labels=FLAT_LABELS, average="macro", zero_division=0),
        "format_accuracy": accuracy_score(true_fmt, pred_fmt),
        "format_macro_f1": f1_score(true_fmt, pred_fmt, labels=FORMAT_LABELS, average="macro", zero_division=0),
        "domain_accuracy": accuracy_score(true_dom, pred_dom),
        "domain_macro_f1": f1_score(true_dom, pred_dom, labels=DOMAIN_LABELS, average="macro", zero_division=0),
        "per_class": {c: rep[c] for c in FLAT_LABELS},
    }


def evaluate_eval_set(labels: pl.DataFrame, pred_fmt: list, pred_dom: list) -> dict:
    """目視の評価用データ全体と，無作為抽出の画像（pool が random で始まる）に対する指標を求める．

    Args:
        labels (pl.DataFrame): data.load_eval_labels の戻り値（行数 N）．
        pred_fmt (list[str]): 長さ N の形式の予測．
        pred_dom (list[str]): 長さ N の画風の予測．

    Returns:
        dict: キー "eval"（640枚）と "eval_random"（無作為抽出の440枚）を持つ辞書．
    """
    mask = np.array([p.startswith("random") for p in labels["pool"]])
    tf, td = np.array(labels["format"].to_list()), np.array(labels["domain"].to_list())
    pf, pd_ = np.array(pred_fmt), np.array(pred_dom)
    return {"eval": evaluate(tf, td, pf, pd_), "eval_random": evaluate(tf[mask], td[mask], pf[mask], pd_[mask])}


def save_run(run_dir: str, labels: pl.DataFrame, pred_fmt: list, pred_dom: list, meta: dict,
             extra_metrics: dict = None, extra_columns: dict = None) -> dict:
    """評価用データに対する予測，指標，メタデータを保存する．

    Args:
        run_dir (str): 保存先のディレクトリ．
        labels (pl.DataFrame): 評価用ラベル（行数 N）．
        pred_fmt (list[str]): 長さ N の形式の予測．
        pred_dom (list[str]): 長さ N の画風の予測．
        meta (dict): 実験条件と処理速度．
        extra_metrics (dict): 追加の指標（学習用・検証用データに対する指標など）．
        extra_columns (dict[str, list]): predictions.csv に追加する列．

    Returns:
        dict: 保存した指標．
    """
    os.makedirs(run_dir, exist_ok=True)
    metrics = evaluate_eval_set(labels, pred_fmt, pred_dom)
    metrics.update(extra_metrics or {})
    flat = [d if f == "normal" else f for f, d in zip(pred_fmt, pred_dom)]
    cols = {"fileId": labels["fileId"], "pool": labels["pool"], "true_format": labels["format"],
            "true_domain": labels["domain"], "true_flat": labels["flat"], "pred_format": list(pred_fmt),
            "pred_domain": list(pred_dom), "pred_flat": flat}
    cols.update(extra_columns or {})
    pl.DataFrame(cols).write_csv(os.path.join(run_dir, "predictions.csv"))
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=4, ensure_ascii=False)
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=4, ensure_ascii=False)
    m = metrics["eval"]
    print(f"  {os.path.relpath(run_dir, OUTPUT_DIR)}: flat acc={m['flat_accuracy']:.3f} "
          f"macroF1={m['flat_macro_f1']:.3f} random acc={metrics['eval_random']['flat_accuracy']:.3f}")
    return metrics


def is_done(run_dir: str) -> bool:
    """正常終了した結果が存在するかを判定する．

    Args:
        run_dir (str): 結果のディレクトリ．

    Returns:
        bool: metrics.json が存在すれば真．
    """
    return os.path.exists(os.path.join(run_dir, "metrics.json"))


def throughput(n_images: int, sec: float) -> dict:
    """処理枚数と秒数から処理速度と全件の処理時間の見積もりを求める．

    Args:
        n_images (int): 処理した画像の数．
        sec (float): 経過秒数．

    Returns:
        dict: n_images，sec，images_per_sec，est_hours_all．
    """
    ips = n_images / sec
    return {"n_images": n_images, "sec": sec, "images_per_sec": ips, "est_hours_all": TOTAL_IMAGES / ips / 3600}
