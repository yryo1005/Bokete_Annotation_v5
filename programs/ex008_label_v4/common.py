"""ex007 の評価と結果の保存の共通処理（ex005 の common.py を定義書 v3 の項目で用いる）．

形式は4クラスの正解率とマクロ F1，属性・媒体は項目ごとの F1 と，軸ごとのマクロ F1（正例がある項目の平均）で評価する．
総合の指標は3軸のマクロ F1 の平均（overall_macro_f1）とする．正例が10件以上の項目のみで求めた頑健な指標
（overall_robust_f1）も併せて求める．評価は調整用（dev），最終評価用（test），全体（all）で行う．
"""
import json
import os

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from data import ATTRIBUTES, FORMATS, MEDIA, PROJECT_DIR

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex008_label_v4")
TOTAL_IMAGES = 6_823_270
AXES = {"attr": ATTRIBUTES, "media": MEDIA}
ROBUST_MIN_SUPPORT = 10


def evaluate(true: pl.DataFrame, pred: pl.DataFrame) -> dict:
    """3軸の予測を正解と比較して指標を求める．

    Args:
        true (pl.DataFrame): 列 format と各項目（bool）を持つ正解の表．
        pred (pl.DataFrame): 同じ列を持つ予測の表（行の順は true と同じ）．

    Returns:
        dict: 形式の正解率・マクロ F1・クラスごとの指標，属性と媒体のマクロ F1・項目ごとの指標，overall_macro_f1．
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
    out["overall_macro_f1"] = float(np.mean([out["format_macro_f1"], out["attr_macro_f1"], out["media_macro_f1"]]))
    # 頑健な指標：正例が ROBUST_MIN_SUPPORT 件以上の項目のみの F1 の平均（件数の少ない項目の1枚の正誤に左右されない）
    robust = [out["format_macro_f1"]]
    for axis in AXES:
        f = [v["f1"] for v in out[f"{axis}_per_item"].values() if v["support"] >= ROBUST_MIN_SUPPORT]
        robust.append(float(np.mean(f)))
    out["overall_robust_f1"] = float(np.mean(robust))
    return out


def evaluate_splits(labels: pl.DataFrame, pred: pl.DataFrame) -> dict:
    """評価用データの全体，調整用，最終評価用に対する指標を求める．

    Args:
        labels (pl.DataFrame): data.load_eval_labels("all") の戻り値．
        pred (pl.DataFrame): 列 format と各項目を持つ予測の表（行の順は labels と同じ）．

    Returns:
        dict: キー "all"，"dev"，"test"．
    """
    out = {"all": evaluate(labels, pred)}
    for s in ["dev", "test"]:
        m = (labels["split"] == s).to_numpy()
        out[s] = evaluate(labels.filter(pl.Series(m)), pred.filter(pl.Series(m)))
    return out


def save_run(run_dir: str, labels: pl.DataFrame, pred: pl.DataFrame, meta: dict, extra_metrics: dict = None) -> dict:
    """予測，指標，メタデータを保存する．

    Args:
        run_dir (str): 保存先のディレクトリ．
        labels (pl.DataFrame): 評価用ラベル（640枚）．
        pred (pl.DataFrame): 予測の表．
        meta (dict): 実験条件と処理速度．
        extra_metrics (dict): 追加の指標（学習用・検証用データに対する指標など）．

    Returns:
        dict: 保存した指標．
    """
    os.makedirs(run_dir, exist_ok=True)
    metrics = evaluate_splits(labels, pred)
    metrics.update(extra_metrics or {})
    pl.concat([labels.select("fileId", "pool", "split"), pred.rename({c: f"pred_{c}" for c in pred.columns})],
              how="horizontal").write_csv(os.path.join(run_dir, "predictions.csv"))
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=4, ensure_ascii=False)
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=4, ensure_ascii=False)
    d, t = metrics["dev"], metrics["test"]
    print(f"  {os.path.relpath(run_dir, OUTPUT_DIR)}: dev fmt={d['format_accuracy']:.3f} attr={d['attr_macro_f1']:.3f} "
          f"media={d['media_macro_f1']:.3f} overall={d['overall_macro_f1']:.3f} robust={d['overall_robust_f1']:.3f} | "
          f"test overall={t['overall_macro_f1']:.3f} robust={t['overall_robust_f1']:.3f}")
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
