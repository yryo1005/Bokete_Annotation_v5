"""ex010 の方法ごとの速度と精度を集計し，表と図を作る．

精度は Qwen3-VL-32B の採点（最終評価用データ80枚）で，平均点，4点以上の割合，誤りを含む割合を示す．
速度は速度の測定用データ2,000枚での値で，全写真（約337万枚）の処理時間に換算する．

出力：outputs/ex010_photo_caption/figures/summary.csv，fig_speed_quality.png，fig_examples.png

実行例:
    .venv_annotate/bin/python programs/ex010_photo_caption/summarize.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import plot_image_grid  # noqa: E402
from data import OUTPUT_DIR, eval_ids, load_references, make_splits  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
# モデルの系統ごとの色（dataviz の既定の分類色の1〜5番．検証済み）と，指示の種類ごとの印
FAMILY_COLOR = {"q2": "#2a78d6", "q4": "#eb6834", "q8": "#1baf7a", "q30a3": "#eda100", "q32": "#e87ba4"}
FAMILY_NAME = {"q2": "2B", "q4": "4B", "q8": "8B", "q30a3": "30B-A3B", "q32": "32B"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def label(m: str) -> str:
    """方法の名前を図の短い名前にする．

    Args:
        m (str): 方法の名前．

    Returns:
        str: 短い名前．
    """
    fam, px, kind = (m.split("_") + ["", ""])[:3]
    if m == "q8_448_en_ja":
        return "8B 英語→翻訳"
    if m.startswith("retrieval_"):
        return f"近傍の検索（{m.split('_')[1].replace('k', '千')}枚）"
    k = {"ja": "詳しい指示", "short": "短い指示", "en": "英語"}.get(kind, kind)
    return f"{FAMILY_NAME.get(fam, fam)} {px}px {k}"


def collect() -> pl.DataFrame:
    """全ての方法の速度と採点を集める．

    Returns:
        pl.DataFrame: 1行1方法の表．
    """
    sp = make_splits()
    rows = []
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "judge_eval.csv"))):
        d = os.path.dirname(p)
        m = os.path.basename(d)
        j = pl.read_csv(p).join(sp, on="fileId")
        meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
        cap = pl.read_csv(os.path.join(d, "captions_eval.csv")) if m != "reference" else None
        speed = meta.get("speed_speed", {})
        r = {"method": m, "images_per_sec": speed.get("images_per_sec"), "hours_all_photos": speed.get("est_hours_all_photos"),
             "mean_chars": None if cap is None else float(cap["caption"].fill_null("").str.len_chars().mean())}
        for s in ["dev", "test"]:
            x = j.filter(pl.col("split") == s)
            r[f"{s}_score"] = float(x["score"].mean())
            r[f"{s}_ok"] = float((x["score"] >= 4).mean())
            r[f"{s}_halluc"] = float(x["hallucination"].mean())
            r[f"{s}_subject"] = float(x["subject"].mean())
            r[f"{s}_score_se"] = float(x["score"].std() / np.sqrt(len(x)))
        rows.append(r)
    return pl.DataFrame(rows).sort("test_score", descending=True)


def plot(df: pl.DataFrame) -> None:
    """速度と精度の散布図を作る（VLM の方法のみ．近傍の検索は精度が低いため表のみに示す）．

    Args:
        df (pl.DataFrame): collect の戻り値．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    d = df.filter(pl.col("images_per_sec").is_not_null() & pl.col("method").str.starts_with("q"))
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    markers = {"ja": "o", "short": "s", "en": "^", "en_ja": "D"}
    for r in d.iter_rows(named=True):
        fam, _, kind = r["method"].split("_")[:3]
        kind = "en_ja" if r["method"].endswith("en_ja") else kind
        ax.scatter(r["images_per_sec"], r["test_score"], s=70, color=FAMILY_COLOR[fam], marker=markers.get(kind, "D"),
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(label(r["method"]), (r["images_per_sec"], r["test_score"]), xytext=(6, 10 if kind == "en_ja" else 4),
                    textcoords="offset points",
                    fontsize=8, color=INK)
    sec = ax.secondary_xaxis("top", functions=(lambda x: 3_371_140 / np.maximum(x, 1e-6) / 3600,
                                               lambda h: 3_371_140 / np.maximum(h, 1e-6) / 3600))
    sec.set_xlabel("全写真（約337万枚）の処理時間（時間）", color=INK2)
    sec.set_xticks([100, 50, 30, 20, 15, 12, 10])
    ax.set_xlabel("処理速度（枚/秒，2,000枚で測定）", color=INK2)
    ax.set_ylabel("採点者の平均点（最終評価用80枚，1〜5）", color=INK2)
    ax.set_xlim(5, 95)
    ax.grid(alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=FAMILY_NAME[f]) for f, c in FAMILY_COLOR.items()]
    handles += [plt.Line2D([], [], marker=mk, ls="", color=INK2, label=n) for n, mk in
                [("詳しい指示", "o"), ("短い指示", "s"), ("英語", "^"), ("英語→翻訳", "D")]]
    ax.legend(handles=handles, fontsize=8, ncol=3, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_speed_quality.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)


def examples(methods: list, n: int = 8) -> None:
    """評価用データの一部について，参照と各方法のキャプションを並べた図を作る．

    Args:
        methods (list[str]): 並べる方法．
        n (int): 画像の数．

    Returns:
        None
    """
    ids = eval_ids("test")[40:40 + n]
    refs = load_references()
    caps = {m: dict(pl.read_csv(os.path.join(OUTPUT_DIR, m, "captions_eval.csv")).select("fileId", "caption").iter_rows()) for m in methods}
    titles = [f"参照：{refs[i]}\n" + "\n".join(f"{label(m)}：{caps[m][i]}" for m in methods) for i in ids]
    plot_image_grid(ids, titles, "キャプションの例（最終評価用データ）", os.path.join(FIG, "fig_examples.png"), ncols=4, wrap=38)


def main() -> None:
    """集計と図の作成を行う．

    Returns:
        None
    """
    os.makedirs(FIG, exist_ok=True)
    df = collect()
    df.write_csv(os.path.join(FIG, "summary.csv"))
    plot(df)
    examples(["q8_448_short", "q30a3_448_short", "q2_448_short", "retrieval_20k"])
    with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=220, float_precision=2):
        print(df.select("method", "images_per_sec", "hours_all_photos", "mean_chars", "dev_score", "test_score", "test_ok",
                        "test_halluc", "test_subject"))


if __name__ == "__main__":
    main()
