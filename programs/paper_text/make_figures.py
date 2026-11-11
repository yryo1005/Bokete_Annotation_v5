"""論文「お題画像の文字の扱い」（paper_text/）の図と数値を作るプログラム．

- 入力：ex026 の目視のラベル（v6_labels.txt，v6_labels_repeat.txt）と評価の出力（outputs/ex026_text_attr_v6/eval/）．
- 出力：outputs/paper_text/（numbers.json，fig_examples.png，fig_attrs.pdf，fig_curve.pdf）．論文で用いる図は paper_text/figures/ に複製する．

実行：
    .venv_annotate/bin/python programs/paper_text/make_figures.py
"""
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from PIL import Image  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX026_DIR = os.path.join(PROJECT_DIR, "programs", "ex026_text_attr_v6")
sys.path.insert(0, EX026_DIR)
from data import DATASET_DIR, FORMS, LAYERS, add_derived, image_path, load_all, load_labels  # noqa: E402
from evaluate import kappa  # noqa: E402

EVAL_DIR = os.path.join(PROJECT_DIR, "outputs", "ex026_text_attr_v6", "eval")
OUT = os.path.join(PROJECT_DIR, "outputs", "paper_text")
PAPER_FIG = os.path.join(PROJECT_DIR, "paper_text", "figures")
plt.rcParams["font.family"] = ["IPAexGothic", "IPAGothic", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
# 系列の色（固定の順）
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# 例の画像（番号は 1,620枚の通し番号）．(a) 場面の文字，(b) 吹き出し，(c) 重ねた文字のその他の文，
# (d) 元の字幕の問い，(e) 投稿者が重ねた問い，(f) 余白の問い，(g) 余白の空欄を含む文と名札，(h) 余白のその他の文
EXAMPLES = [8, 50, 109, 841, 1080, 199, 638, 100]
RULES = ["odai_R0", "odai_R1", "odai_R1b", "odai_R1b_noO", "odai_R2"]


def add_rule_variants(d: pl.DataFrame) -> pl.DataFrame:
    """規則 R1b から重ねた文字の形を除いた比較用の規則（odai_R1b_noO）を加える．

    Args:
        d (pl.DataFrame): F_{層}_{形} の列を持つ表．

    Returns:
        pl.DataFrame: 列 odai_R1b_noO を加えた表．
    """
    margin_task = pl.col("F_m_q") | pl.col("F_m_i") | pl.col("F_m_t")
    return d.with_columns((margin_task | (pl.col("F_m_x") & ~pl.col("F_m_k"))).alias("odai_R1b_noO"))


def definition_shift(d: pl.DataFrame) -> list:
    """v5 の目視のお題文ありと，規則ごとに導いたお題文ありの一致を，1,620枚と全データの推定で求める．

    Args:
        d (pl.DataFrame): 1,620枚の表（列 source，w，label_v5，規則の列）．

    Returns:
        list[dict]: 規則と集合ごとの一致率，κ，陽性率，片方のみ陽性の割合．
    """
    e25 = d.filter(pl.col("source") == "ex025")
    rows = []
    for rule in RULES:
        for name, t, w in [("all_1620", d, np.ones(d.height)), ("population", e25, e25["w"].to_numpy())]:
            a, b = t["label_v5"].to_numpy(), t[rule].to_numpy()
            ww = w / w.sum()
            rows.append({"rule": rule, "set": name, "agree": float((ww * (a == b)).sum()), "kappa": kappa(a, b, w),
                         "rate_v5": float((ww * a).sum()), "rate_v6": float((ww * b).sum()),
                         "v5_only": float((ww * (a & ~b)).sum()), "v6_only": float((ww * (~a & b)).sum())})
    return rows


def repeat_agreement(d: pl.DataFrame) -> dict:
    """付け直した90枚のラベルと元のラベルの一致（記号の完全一致率，層・規則ごとの κ）を求める．

    Args:
        d (pl.DataFrame): 1,620枚の表（列 no，code，L_*，odai_*）．

    Returns:
        dict: キー exact（完全一致率），n，kappa（属性名 → κ）．
    """
    rep = load_labels(os.path.join(EX026_DIR, "v6_labels_repeat.txt")).rename({"no": "rno"})
    rep = add_rule_variants(add_derived(rep))
    keys = pl.read_csv(os.path.join(DATASET_DIR, "repeat_targets.csv"))
    rep = rep.join(keys, on="rno")
    cols = [f"L_{l}" for l in LAYERS] + ["odai_R0", "odai_R1", "odai_R1b", "odai_R2"]
    old = d.select(["no", "code"] + cols)
    m = rep.select(["no", "code"] + cols).join(old, on="no", suffix="_old")
    return {"n": m.height, "exact": float((m["code"] == m["code_old"]).mean()),
            "kappa": {c: kappa(m[c].to_numpy(), m[c + "_old"].to_numpy()) for c in cols}}


def overlay_counts(d: pl.DataFrame) -> dict:
    """重ねた文字に問い・指示・題目の句がある画像と，R1 と R1b で分かれる画像を，v5 のお題文ありごとに数える．

    Args:
        d (pl.DataFrame): 1,620枚の表．

    Returns:
        dict: 集計の辞書（枚数と全データの推定の割合）．
    """
    oo = pl.col("F_o_q") | pl.col("F_o_i") | pl.col("F_o_t")
    split = pl.col("odai_R1") & ~pl.col("odai_R1b")
    e25 = d.filter(pl.col("source") == "ex025")
    w = e25["w"].to_numpy() / e25["w"].sum()
    v5 = e25["label_v5"].to_numpy()
    o_e = e25.select(oo.alias("o"))["o"].to_numpy()
    res = {"overlay_forms": {}, "r1_vs_r1b": {}}
    for v in [True, False]:
        t = d.filter(oo & (pl.col("label_v5") == v))
        res["overlay_forms"]["v5_pos" if v else "v5_neg"] = {
            "n": t.height, "q": int(t["F_o_q"].sum()), "i": int(t["F_o_i"].sum()), "t": int(t["F_o_t"].sum()),
            "population": float((w * (o_e & (v5 == v))).sum())}
        res["r1_vs_r1b"]["v5_pos" if v else "v5_neg"] = d.filter(split & (pl.col("label_v5") == v)).height
    return res


def code_frequency(d: pl.DataFrame) -> dict:
    """層と形の頻度（1,620枚と全データの推定）を求める．

    Args:
        d (pl.DataFrame): 1,620枚の表．

    Returns:
        dict: 属性名 → {"all_1620": 割合，"population": 割合}．
    """
    e25 = d.filter(pl.col("source") == "ex025")
    w = e25["w"].to_numpy() / e25["w"].sum()
    cols = [f"L_{l}" for l in LAYERS] + [f"F_{l}_{f}" for l in ["o", "m"] for f in FORMS]
    cols += ["text_any"]
    d = d.with_columns(pl.any_horizontal([pl.col(f"L_{l}") for l in LAYERS]).alias("text_any"))
    e25 = e25.with_columns(pl.any_horizontal([pl.col(f"L_{l}") for l in LAYERS]).alias("text_any"))
    return {c: {"all_1620": float(d[c].mean()), "population": float((w * e25[c].to_numpy()).sum())} for c in cols}


def fig_examples(d: pl.DataFrame, out: str) -> None:
    """例の画像を2行4列に並べた図を保存する．

    Args:
        d (pl.DataFrame): 1,620枚の表（列 no，rep_odaiId）．
        out (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 4, figsize=(3.5, 2.2))
    for k, (ax, no) in enumerate(zip(axes.ravel(), EXAMPLES)):
        r = d.filter(pl.col("no") == no).row(0, named=True)
        im = Image.open(image_path(r["rep_odaiId"])).convert("RGB")
        im.thumbnail((480, 480))
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(f"({chr(ord('a') + k)}) {r['code']}", fontsize=6.5, family="DejaVu Sans", pad=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=400)
    plt.close(fig)


def fig_attrs(out: str) -> None:
    """層と文の形ごとの F1（1,620枚）を，教師と2つの生徒で比べる図を保存する．

    Args:
        out (str): 保存先．

    Returns:
        None
    """
    m = pl.read_csv(os.path.join(EVAL_DIR, "attrs.csv")).filter(pl.col("set") == "all_1620")
    methods = {"qwen": "Qwen3-VL-32B（教師）", "emb": "生徒：埋め込み", "emb+text_yomi": "生徒：埋め込み＋文（YomiToku）"}
    layer = {"s": "場面", "b": "吹き出し", "o": "重ねた", "m": "余白"}
    form = {"k": "空欄", "q": "問い", "i": "指示", "t": "題目", "x": "その他"}
    attrs = [f"L_{l}" for l in LAYERS] + [f"F_{l}_{f}" for l in ["o", "m"] for f in FORMS]
    names = [f"{layer[a[2]]}" if a.startswith("L_") else f"{layer[a[2]]}：{form[a[4]]}" for a in attrs]
    npos = {r["attr"]: int(r["n_pos"]) for r in m.filter(pl.col("method") == "qwen").iter_rows(named=True)}
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(attrs))
    width = 0.27
    for j, (meth, label) in enumerate(methods.items()):
        g = {r["attr"]: r["f1"] for r in m.filter(pl.col("method") == meth).iter_rows(named=True)}
        y = [g.get(a, np.nan) if a.startswith("F_") or meth != "emb+text_yomi" else np.nan for a in attrs]
        ax.bar(x + (j - 1) * width, y, width * 0.92, color=COLORS[j], label=label)
    ax.axvline(3.5, color="#999999", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({npos[a]})" for n, a in zip(names, attrs)], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("F1", fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_curve(out: str) -> None:
    """学習の画像の数と分割外の F1 の関係（学習曲線）を保存する．

    Args:
        out (str): 保存先．

    Returns:
        None
    """
    lc = pl.read_csv(os.path.join(EVAL_DIR, "learning_curve.csv"))
    targets = {"L_m": "余白の文字", "odai_R1b": "お題文あり（R1b）", "A_q": "問い", "A_x": "その他の文"}
    feats = {"emb": "-", "emb+text_v6": "--"}
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for j, (t, label) in enumerate(targets.items()):
        for f, ls in feats.items():
            g = lc.filter((pl.col("target") == t) & (pl.col("feature") == f)).sort("n_train")
            ax.plot(g["n_train"], g["f1"], ls=ls, marker="o", ms=3, lw=1.5, color=COLORS[j],
                    label=label if f == "emb" else None)
    ax.plot([], [], ls="-", color="#555555", label="埋め込み")
    ax.plot([], [], ls="--", color="#555555", label="埋め込み＋文")
    ax.set_xlabel("学習に用いた画像の数", fontsize=8)
    ax.set_ylabel("F1", fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """数値と図を作り，論文の図のディレクトリに複製する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PAPER_FIG, exist_ok=True)
    d = add_rule_variants(load_all())
    numbers = {"definition_shift": definition_shift(d), "repeat": repeat_agreement(d),
               "overlay": overlay_counts(d), "frequency": code_frequency(d)}
    json.dump(numbers, open(os.path.join(OUT, "numbers.json"), "w"), ensure_ascii=False, indent=1)
    fig_examples(d, os.path.join(OUT, "fig_examples.png"))
    fig_attrs(os.path.join(OUT, "fig_attrs.pdf"))
    fig_curve(os.path.join(OUT, "fig_curve.pdf"))
    for name in ["fig_examples.png", "fig_attrs.pdf", "fig_curve.pdf"]:
        shutil.copy(os.path.join(OUT, name), os.path.join(PAPER_FIG, name))
    print(json.dumps(numbers, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
