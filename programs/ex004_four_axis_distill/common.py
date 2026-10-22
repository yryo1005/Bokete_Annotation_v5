"""ex004 の評価と結果の保存の共通処理．

形式は4クラスの正解率とマクロ F1，属性・媒体・事前知識は項目ごとの適合率・再現率・F1 と，
軸ごとのマクロ F1（評価用データに正例がある項目の平均）で評価する．
"""
import json
import os

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from data import ATTRIBUTES, FORMATS, KNOWLEDGE, MEDIA, PROJECT_DIR

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex004_four_axis_distill")
TOTAL_IMAGES = 6_823_270
AXES = {"attr": ATTRIBUTES, "media": MEDIA, "know": KNOWLEDGE}


def evaluate(true: pl.DataFrame, pred: pl.DataFrame) -> dict:
    """4軸の予測を正解と比較して指標を求める．

    Args:
        true (pl.DataFrame): 列 format と各項目（bool）を持つ正解の表（行数 N）．
        pred (pl.DataFrame): 同じ列を持つ予測の表（行数 N，行の順は true と同じ）．

    Returns:
        dict: format_accuracy，format_macro_f1，format_per_class，各軸の macro_f1 と項目ごとの指標，
        媒体の主ラベル（真の項目の集合の完全一致）の正解率．
    """
    out = {"format_accuracy": accuracy_score(true["format"], pred["format"]),
           "format_macro_f1": f1_score(true["format"], pred["format"], labels=FORMATS, average="macro", zero_division=0)}
    p, r, f, s = precision_recall_fscore_support(true["format"], pred["format"], labels=FORMATS, zero_division=0)
    out["format_per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                               for i, c in enumerate(FORMATS)}
    for axis, items in AXES.items():
        per, f1s = {}, []
        for c in items:
            y, q = true[c].to_numpy().astype(bool), pred[c].to_numpy().astype(bool)
            pp, rr, ff, _ = precision_recall_fscore_support(y, q, average="binary", zero_division=0)
            per[c] = {"precision": float(pp), "recall": float(rr), "f1": float(ff), "support": int(y.sum())}
            if y.sum() > 0:
                f1s.append(ff)
        out[f"{axis}_macro_f1"] = float(np.mean(f1s))
        out[f"{axis}_per_item"] = per
    tm, pm = true.select(MEDIA).to_numpy().astype(bool), pred.select(MEDIA).to_numpy().astype(bool)
    out["media_exact_match"] = float((tm == pm).all(axis=1).mean())
    tk, pk = true.select(KNOWLEDGE).to_numpy().astype(bool), pred.select(KNOWLEDGE).to_numpy().astype(bool)
    out["know_exact_match"] = float((tk == pk).all(axis=1).mean())
    out["overall_macro_f1"] = float(np.mean([out["format_macro_f1"], out["attr_macro_f1"], out["media_macro_f1"],
                                             out["know_macro_f1"]]))
    return out


def evaluate_eval_set(labels: pl.DataFrame, pred: pl.DataFrame) -> dict:
    """評価用データ全体と，無作為抽出の画像（pool が random で始まる）に対する指標を求める．

    Args:
        labels (pl.DataFrame): data.load_eval_labels の戻り値．
        pred (pl.DataFrame): 列 format と各項目を持つ予測の表（行の順は labels と同じ）．

    Returns:
        dict: キー "eval" と "eval_random"．
    """
    mask = labels["pool"].str.starts_with("random")
    return {"eval": evaluate(labels, pred), "eval_random": evaluate(labels.filter(mask), pred.filter(mask))}


def save_run(run_dir: str, labels: pl.DataFrame, pred: pl.DataFrame, meta: dict, extra_metrics: dict = None) -> dict:
    """予測，指標，メタデータを保存する．

    Args:
        run_dir (str): 保存先のディレクトリ．
        labels (pl.DataFrame): 評価用ラベル．
        pred (pl.DataFrame): 予測の表（列 format と各項目，任意で確率などの列）．
        meta (dict): 実験条件と処理速度．
        extra_metrics (dict): 追加の指標（学習用・検証用データに対する指標など）．

    Returns:
        dict: 保存した指標．
    """
    os.makedirs(run_dir, exist_ok=True)
    metrics = evaluate_eval_set(labels, pred)
    metrics.update(extra_metrics or {})
    pl.concat([labels.select("fileId", "pool"), pred.rename({c: f"pred_{c}" for c in pred.columns})],
              how="horizontal").write_csv(os.path.join(run_dir, "predictions.csv"))
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=4, ensure_ascii=False)
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=4, ensure_ascii=False)
    m = metrics["eval"]
    print(f"  {os.path.relpath(run_dir, OUTPUT_DIR)}: format acc={m['format_accuracy']:.3f} "
          f"F1={m['format_macro_f1']:.3f} | attr F1={m['attr_macro_f1']:.3f} media F1={m['media_macro_f1']:.3f} "
          f"know F1={m['know_macro_f1']:.3f}")
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


def plot_image_grid(ids: list, titles: list, suptitle: str, path: str, ncols: int = 5) -> None:
    """画像を正方形の余白に収めて格子状に並べ，各画像に題を付けて保存する．

    Args:
        ids (list[int]): お題IDのリスト．
        titles (list[str]): 各画像の題．
        suptitle (str): 図全体の題．
        path (str): 保存先のパス．
        ncols (int): 列数．

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    from data import image_path

    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.0 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, fid, title in zip(np.ravel(axes), ids, titles):
        im = Image.open(image_path(fid)).convert("RGB")
        im.thumbnail((400, 400))
        canvas = Image.new("RGB", (400, 400), "white")
        canvas.paste(im, ((400 - im.width) // 2, (400 - im.height) // 2))
        ax.imshow(canvas)
        ax.set_title(title, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if nrows > 1 else 0.9))
    fig.suptitle(suptitle)
    fig.savefig(path, bbox_inches="tight", dpi=90)
    plt.close(fig)
