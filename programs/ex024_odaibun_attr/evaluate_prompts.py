"""Qwen の指示とモデルごとの，お題文ありの判定の適合率・再現率・F1 を求める．

集合は dev（評価用データの調整用320枚），test（最終評価用320枚），real（実データ380枚．重み付きの推定）である．
比較の基準として，現在のパッケージの生徒（形式がお題文）と全プロセスの値も並べる．

出力：outputs/ex024_odaibun_attr/prompt_metrics.csv

実行例:
    .venv_annotate/bin/python programs/ex024_odaibun_attr/evaluate_prompts.py
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ANNOTATION, OUTPUT_DIR, load_truth  # noqa: E402

SPLITS = ["dev", "test", "real"]


def prf(t: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    """重み付きの適合率・再現率・F1 を求める．

    Args:
        t (np.ndarray): 形状 (N,) の正解（bool）．
        p (np.ndarray): 形状 (N,) の予測（bool）．
        w (np.ndarray): 形状 (N,) の重み．

    Returns:
        dict: precision, recall, f1, n_pos（重みなしの正例の数），n_pred（重みなしの予測の正例の数）．
    """
    tp, npred, ntrue = (w * (t & p)).sum(), (w * p).sum(), (w * t).sum()
    return {"precision": tp / npred if npred else float("nan"), "recall": tp / ntrue if ntrue else float("nan"),
            "f1": 2 * tp / (npred + ntrue) if npred + ntrue else float("nan"), "n_pos": int(t.sum()), "n_pred": int(p.sum())}


def judges(truth: pl.DataFrame) -> dict:
    """判断者（Qwen の指示とモデル，現在の生徒と全プロセス）ごとの予測を集める．

    Args:
        truth (pl.DataFrame): load_truth の戻り値．

    Returns:
        dict[str, np.ndarray]: 判断者名 → 形状 (N,) の予測（truth の行の順．答えがない画像は偽）．
    """
    key = truth.select("fileId")
    out = {}
    a = pl.scan_parquet(ANNOTATION).filter(pl.col("odaiId").is_in(key["fileId"].implode())).collect()
    a = key.join(a.rename({"odaiId": "fileId"}), on="fileId", how="left")
    p_fmt = a.select([f"p_format_{f}" for f in ["karuta", "anaume", "odaibun", "normal"]]).to_numpy()
    out["current_student"] = p_fmt.argmax(1) == 2
    out["current_full"] = (a["format"] == "odaibun").to_numpy()
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "qwen", "*", "P*.csv"))):
        name = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)[:-4]}"
        q = key.join(pl.read_csv(path), on="fileId", how="left")
        out[name] = q["answer"].fill_null(False).to_numpy()
        if q["self"].drop_nulls().len():
            out[name + "self"] = q["self"].fill_null(False).to_numpy()
    return out


def speed_of(name: str) -> float:
    """Qwen の判断者の速度（枚/秒）を返す．

    Args:
        name (str): 判断者名（{モデル}__{画素}/{指示}）．

    Returns:
        float: 枚/秒（Qwen 以外は nan）．
    """
    if "/" not in name:
        return float("nan")
    d, p = name.split("/")
    path = os.path.join(OUTPUT_DIR, "qwen", d, "speed.json")
    s = json.load(open(path)) if os.path.exists(path) else {}
    return s.get(p.replace("self", ""), {}).get("images_per_sec", float("nan"))


def main() -> None:
    """集計して保存し，表示する．

    Returns:
        None
    """
    truth = load_truth()
    t = truth["odaibun_attr"].to_numpy()
    w = truth["w"].to_numpy()
    rows = []
    for name, p in judges(truth).items():
        r = {"judge": name, "images_per_sec": speed_of(name)}
        for s in SPLITS:
            m = (truth["split"] == s).to_numpy()
            for k, v in prf(t[m], p[m], w[m]).items():
                r[f"{s}_{k}"] = v
        rows.append(r)
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(OUTPUT_DIR, "prompt_metrics.csv"))
    cols = ["judge", "images_per_sec"] + [f"{s}_{k}" for s in SPLITS for k in ["precision", "recall", "f1"]]
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=250, float_precision=3, fmt_str_lengths=50):
        print(df.select(cols))


if __name__ == "__main__":
    main()


def plot_prompts(df: pl.DataFrame, out: str) -> None:
    """モデルと指示ごとの F1（最終評価用と実データ）と速度の図を作る．

    Args:
        df (pl.DataFrame): main で保存した表（prompt_metrics.csv）．
        out (str): 保存先．

    Returns:
        None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    d = df.filter(pl.col("judge").str.contains("/") & ~pl.col("judge").str.ends_with("self") & ~pl.col("judge").str.ends_with("P4"))
    names = [j.replace("Qwen3-VL-", "").replace("-Instruct", "").replace("__200704", "").replace("__802816", "（高解像度）")
             .replace("-FP8", "") for j in d["judge"]]
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
    axes[0].scatter(d["test_f1"], y + 0.12, color="#2a78d6", marker="o", s=36, label="最終評価用", zorder=3)
    axes[0].scatter(d["real_f1"], y - 0.12, color="#eb6834", marker="s", s=36, label="実データ（推定）", zorder=3)
    cur = df.filter(pl.col("judge") == "current_student")
    axes[0].axvline(cur["test_f1"][0], color="#2a78d6", ls="--", lw=1)
    axes[0].axvline(cur["real_f1"][0], color="#eb6834", ls="--", lw=1)
    axes[0].set_xlabel("お題文ありの F1（破線は現在のパッケージ）")
    axes[0].set_xlim(0.2, 0.9)
    axes[0].grid(axis="x", color="#dddddd", lw=0.6)
    axes[0].legend(fontsize=9, loc="lower left")
    axes[1].barh(y, d["images_per_sec"], color="#7a7a7a", height=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("速度（枚/秒，対数）")
    axes[1].grid(axis="x", color="#dddddd", lw=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_prompts(pl.read_csv(os.path.join(OUTPUT_DIR, "prompt_metrics.csv")), os.path.join(OUTPUT_DIR, "fig_prompts.png"))
