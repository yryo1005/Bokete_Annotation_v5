"""Gemini，Claude Code，自動の方法（パッケージの最終版）のお題画像のラベルを比べる．

比べるのは，Gemini がラベルを付けた画像（--set の集合のうち raw/ に JSON がある画像）である．
- 判断者の組（Claude Code と Gemini，Claude Code と自動，Gemini と自動）ごとに，項目ごとの一致率と Cohen の κ 係数
- Claude Code を基準としたときの，Gemini と自動の項目ごとの F1（形式はクラスごと，属性と媒体は2値）
- 全項目の一致率（層別抽出では，層の重みで実際の分布での値も推定する）
- 層ごとの全項目の一致率
- 不一致の例の一覧（Gemini の判断の根拠を付ける）と，例の図

出力：outputs/ex022_gemini_image_labels/compare_{モデル}/（pairwise.csv，f1.csv，strata.csv，disagreements.csv，summary.json），
      .reports/figures_029/（disagree_{軸}.png）

実行例:
    .venv_annotate/bin/python programs/ex022_gemini_image_labels/compare.py --model gemini-3.1-flash-lite
"""
import argparse
import json
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (ATTRIBUTES, FORMATS, LABEL_COLS, MEDIA, OUTPUT_DIR, PROJECT_DIR, image_path,  # noqa: E402
                  load_auto_labels, load_claude_labels, load_gemini_labels)

FIG_DIR = os.path.join(PROJECT_DIR, ".reports", "figures_029")
FONT = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
JUDGES = {"c": "Claude Code", "g": "Gemini", "a": "自動"}
JA = {"format": "形式", "has_blank": "空欄あり", "has_karuta_mark": "カルタ記号あり", "has_text": "文字あり", "photo": "写真",
      "anime": "アニメ", "game": "ゲーム", "manga": "マンガ", "text_graphic": "テキストグラフィック", "painting": "絵画",
      "illustration": "イラスト", "karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常"}


def cohen_kappa(x: list, y: list) -> float:
    """2人の判断者の Cohen の κ 係数を求める．

    Args:
        x (list): 判断者1のラベル（長さ N）．
        y (list): 判断者2のラベル（長さ N）．

    Returns:
        float: κ 係数（偶然の一致率が1のときは nan）．
    """
    x, y = np.asarray(x), np.asarray(y)
    po = float(np.mean(x == y))
    pe = sum(float(np.mean(x == c)) * float(np.mean(y == c)) for c in np.union1d(x, y))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def f1_score(truth: list, pred: list) -> tuple:
    """2値の F1，適合率，再現率と正例の件数を求める．

    Args:
        truth (list[bool]): 基準のラベル（長さ N）．
        pred (list[bool]): 比べるラベル（長さ N）．

    Returns:
        tuple[float, float, float, int]: F1，適合率，再現率，基準の正例の件数（定義できない値は nan）．
    """
    t, p = np.asarray(truth, bool), np.asarray(pred, bool)
    tp = int((t & p).sum())
    prec = tp / p.sum() if p.sum() else float("nan")
    rec = tp / t.sum() if t.sum() else float("nan")
    f1 = 2 * tp / (t.sum() + p.sum()) if (t.sum() + p.sum()) else float("nan")
    return f1, prec, rec, int(t.sum())


def merge_judges(model: str, target: str) -> pl.DataFrame:
    """3者のラベルを1つの表に結合する．

    Args:
        model (str): Gemini のモデル名．
        target (str): 対象の集合（all / eval / random / strat）．

    Returns:
        pl.DataFrame: 列 fileId, set, stratum, weight, reason と，各項目に接尾辞 _c，_g，_a を付けた列を持つ表．
    """
    c = load_claude_labels()
    if target != "all":
        c = c.filter(pl.col("set") == target)
    g = load_gemini_labels(model)
    a = load_auto_labels(c["fileId"].unique().to_list())
    d = c.rename({k: f"{k}_c" for k in LABEL_COLS}).join(
        g.select(["fileId", "reason"] + LABEL_COLS).rename({k: f"{k}_g" for k in LABEL_COLS}), on="fileId").join(
        a.rename({k: f"{k}_a" for k in LABEL_COLS}), on="fileId", how="left")
    return d.with_columns([pl.all_horizontal([pl.col(f"{k}_{j}") == pl.col(f"{k}_c") for k in LABEL_COLS]).alias(f"all_{j}")
                           for j in "ga"] + [pl.all_horizontal([pl.col(f"{k}_g") == pl.col(f"{k}_a") for k in LABEL_COLS]).alias("all_ga")])


def pairwise_table(d: pl.DataFrame) -> pl.DataFrame:
    """判断者の組ごとに，項目ごとの一致率と κ 係数を求める．

    Args:
        d (pl.DataFrame): merge_judges の表．

    Returns:
        pl.DataFrame: 列 item, pair, agree, kappa を持つ表．
    """
    rows = []
    for item in LABEL_COLS:
        for x, y in [("c", "g"), ("c", "a"), ("g", "a")]:
            u, v = d[f"{item}_{x}"].to_list(), d[f"{item}_{y}"].to_list()
            rows.append({"item": item, "pair": f"{x}-{y}", "agree": float(np.mean(np.array(u) == np.array(v))),
                         "kappa": cohen_kappa(u, v)})
    return pl.DataFrame(rows)


def f1_table(d: pl.DataFrame) -> pl.DataFrame:
    """Claude Code を基準として，Gemini と自動のクラスごとの F1 を求める．

    Args:
        d (pl.DataFrame): merge_judges の表．

    Returns:
        pl.DataFrame: 列 item, judge, f1, precision, recall, n_pos を持つ表．
    """
    rows = []
    items = [("format", f) for f in FORMATS] + [(k, None) for k in ATTRIBUTES + MEDIA]
    for col, cls in items:
        t = d[f"{col}_c"].to_list() if cls is None else [v == cls for v in d[f"{col}_c"].to_list()]
        for j in "ga":
            p = d[f"{col}_{j}"].to_list() if cls is None else [v == cls for v in d[f"{col}_{j}"].to_list()]
            f1, pr, rc, n = f1_score(t, p)
            rows.append({"item": cls or col, "judge": j, "f1": f1, "precision": pr, "recall": rc, "n_pos": n})
    return pl.DataFrame(rows)


def strata_table(d: pl.DataFrame) -> pl.DataFrame:
    """層ごとの全項目の一致率を求める（層別抽出の画像のみ）．

    Args:
        d (pl.DataFrame): merge_judges の表．

    Returns:
        pl.DataFrame: 列 stratum, n, weight, all_g, all_a, all_ga を持つ表．
    """
    s = d.filter(pl.col("set") == "strat")
    return s.group_by("stratum").agg(pl.len().alias("n"), pl.col("weight").first(),
                                     *[pl.col(f"all_{j}").mean() for j in ["g", "a", "ga"]]).sort("weight")


def weighted_all(d: pl.DataFrame) -> dict:
    """層の重み（抽出確率の逆数）で，実際の分布での全項目の一致率を推定する．

    Args:
        d (pl.DataFrame): merge_judges の表（層別抽出の画像のみを用いる）．

    Returns:
        dict[str, float]: 組の名前 → 推定値．
    """
    s = d.filter(pl.col("set") == "strat")
    if s.is_empty():
        return {}
    w = s["weight"].to_numpy()
    return {j: float((w * s[f"all_{j}"].to_numpy()).sum() / w.sum()) for j in ["g", "a", "ga"]}


def label_text(row: dict, j: str) -> str:
    """1人の判断者のラベルを短い文字列にする．

    Args:
        row (dict): merge_judges の1行．
        j (str): 判断者（c / g / a）．

    Returns:
        str: 「形式／属性／媒体」の文字列．
    """
    attrs = "・".join(JA[k].replace("あり", "") for k in ATTRIBUTES if row[f"{k}_{j}"]) or "なし"
    media = "・".join(JA[m] for m in MEDIA if row[f"{m}_{j}"])
    return f"{JA[row[f'format_{j}']]}／{attrs}／{media}"


def disagreement_table(d: pl.DataFrame) -> pl.DataFrame:
    """Claude Code と Gemini のラベルが異なる画像の一覧を作る．

    Args:
        d (pl.DataFrame): merge_judges の表．

    Returns:
        pl.DataFrame: 列 fileId, set, stratum, axes（異なる軸）, claude, gemini, auto, reason を持つ表．
    """
    rows = []
    for r in d.filter(~pl.col("all_g")).iter_rows(named=True):
        axes = [ax for ax, cols in [("形式", ["format"]), ("属性", ATTRIBUTES), ("媒体", MEDIA)]
                if any(r[f"{k}_c"] != r[f"{k}_g"] for k in cols)]
        rows.append({"fileId": r["fileId"], "set": r["set"], "stratum": r["stratum"], "axes": "・".join(axes),
                     "claude": label_text(r, "c"), "gemini": label_text(r, "g"),
                     "auto": label_text(r, "a") if r["format_a"] is not None else "", "reason": r["reason"]})
    return pl.DataFrame(rows)


def plot_examples(rows: list, path: str) -> None:
    """不一致の例を，画像と3者のラベル，Gemini の根拠で並べて保存する．

    Args:
        rows (list[dict]): disagreement_table の行（最大6件）．
        path (str): 保存先．

    Returns:
        None
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.0 * n), gridspec_kw={"width_ratios": [1, 2.2]}, squeeze=False)
    for (ax_i, ax_t), r in zip(axes, rows):
        ax_i.axis("off")
        ax_t.axis("off")
        im = Image.open(image_path(r["fileId"])).convert("RGB")
        im.thumbnail((400, 400))
        ax_i.imshow(im)
        ax_i.set_title(str(r["fileId"]), fontsize=10)
        body = (f"Claude Code：{r['claude']}\nGemini：{r['gemini']}\n自動：{r['auto']}\n\nGemini の根拠：\n"
                + "\n".join(textwrap.wrap(r["reason"] or "", 38)))
        ax_t.text(0.0, 0.95, body, va="top", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=85, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """3者のラベルを比べ，表と図を保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--set", default="strat", choices=["all", "eval", "random", "strat"])
    args = parser.parse_args()
    out = os.path.join(OUTPUT_DIR, f"compare_{args.model}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()

    d = merge_judges(args.model, args.set)
    pw, f1, st, dis = pairwise_table(d), f1_table(d), strata_table(d), disagreement_table(d)
    pw.write_csv(os.path.join(out, "pairwise.csv"))
    f1.write_csv(os.path.join(out, "f1.csv"))
    st.write_csv(os.path.join(out, "strata.csv"))
    dis.write_csv(os.path.join(out, "disagreements.csv"))
    summary = {"model": args.model, "n": d.height, "all_match": {j: float(d[f"all_{j}"].mean()) for j in ["g", "a", "ga"]},
               "all_match_weighted": weighted_all(d), "n_auto_missing": int(d["format_a"].null_count())}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for ax in ["形式", "属性", "媒体"]:
        sub = dis.filter(pl.col("axes").str.contains(ax))
        if sub.height:
            plot_examples(sub.sample(n=min(6, sub.height), seed=0).to_dicts(), os.path.join(FIG_DIR, f"disagree_{ax}.png"))
    print(json.dumps(summary, ensure_ascii=False))
    print(pw.pivot(on="pair", index="item", values="kappa"))


if __name__ == "__main__":
    main()
