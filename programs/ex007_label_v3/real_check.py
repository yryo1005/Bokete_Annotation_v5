"""定義書 v3 の生徒による実データへのアノテーションの分布を集計し，無作為抽出の画像を目視で確認する．

1. --make：未使用の実データ（約29.5万枚）から，評価用データに使っていない画像を無作為に N_CHECK 枚抽出し，
   一覧画像を作る（datasets/ex005_rare_weighted_distill/real_check_v3/）．予測は表示しない．
2. 目視のラベル付け：Claude Code が定義書 v3 で全項目のラベルを付け，real_check_v3.txt に記録する．
   書式は「番号 形式 属性 媒体」で，形式は K/A/O/N，属性は空欄 b・カルタ記号 k・文字 t の並び（なしは -），
   媒体は P/A/G/M/T/E/I の並びとする．
3. --analyze：指定した複数の生徒（seed 0）について，実データでのラベルの分布と，目視のラベルに対する精度を集計する．

実行例:
    .venv_annotate/bin/python programs/ex007_label_v3/real_check.py --runs student_mlp/T_v8m__base+rare__weighted --make
    .venv_annotate/bin/python programs/ex007_label_v3/real_check.py --runs student_mlp/T_v8m__base+rare__weighted \
        student_mlp/T_v8m__base+rare__unweighted --analyze
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, plot_image_grid  # noqa: E402
from data import ATTRIBUTES, DATASET_DIR, FORMATS, MEDIA, load_eval_labels, load_rare_labels  # noqa: E402

N_CHECK = 160
CHECK_DIR = os.path.join(DATASET_DIR, "real_check_v3")
CHECK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_check_v3.txt")
F_CODE = {"karuta": "K", "anaume": "A", "odaibun": "O", "normal": "N"}
A_CODE = {"has_blank": "b", "has_karuta_mark": "k", "has_text": "t"}
M_CODE = {"photo": "P", "anime": "A", "game": "G", "manga": "M", "text_graphic": "T", "painting": "E", "illustration": "I"}


def codes(row: dict) -> tuple:
    """予測の行を記号の3つ組にする．

    Args:
        row (dict): 列 format と各項目を持つ行．

    Returns:
        tuple[str, str, str]: (形式，属性，媒体) の記号．
    """
    a = "".join(A_CODE[x] for x in ATTRIBUTES if row[x]) or "-"
    m = "".join(M_CODE[x] for x in MEDIA if row[x])
    return F_CODE[row["format"]], a, m


def make(ann: pl.DataFrame, seed: int = 0) -> None:
    """目視のラベル付け用の画像を抽出して一覧画像を作る（予測は表示しない）．

    Args:
        ann (pl.DataFrame): 実データへのアノテーション（お題IDの集合として用いる）．
        seed (int): 乱数のシード．

    Returns:
        None
    """
    used = set(load_eval_labels()["fileId"].to_list()) | set(load_rare_labels()["fileId"].to_list())
    s = ann.filter(~pl.col("fileId").is_in(list(used))).select("fileId").sample(n=N_CHECK, seed=seed)
    os.makedirs(CHECK_DIR, exist_ok=True)
    s.write_csv(os.path.join(CHECK_DIR, "sample.csv"))
    ids = s["fileId"].to_list()
    for k in range(0, len(ids), 20):
        part = ids[k:k + 20]
        plot_image_grid(part, [f"{k + i:03d}:{f}" for i, f in enumerate(part)], f"実データの確認 {k // 20}",
                        os.path.join(CHECK_DIR, f"sheet_{k // 20:02d}.png"), ncols=5)
    print(f"{len(ids)} 枚の一覧を作成した: {CHECK_DIR}")


def load_check_labels() -> pl.DataFrame:
    """目視のラベルを読み込む．

    Returns:
        pl.DataFrame: 列 fileId, format と属性・媒体の各項目（bool）を持つ表．
    """
    ids = pl.read_csv(os.path.join(CHECK_DIR, "sample.csv"))["fileId"].to_list()
    inv_f = {v: k for k, v in F_CODE.items()}
    rows = {}
    for line in open(CHECK_FILE, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            k, f, a, m = line.split()
            r = {"fileId": ids[int(k)], "format": inv_f[f]}
            r.update({x: (c in a) for x, c in A_CODE.items()})
            r.update({x: (c in m) for x, c in M_CODE.items()})
            rows[int(k)] = r
    assert len(rows) == len(ids)
    return pl.DataFrame([rows[k] for k in range(len(ids))])


def analyze(runs: list) -> None:
    """生徒ごとに実データでの分布と目視のラベルに対する精度を集計して保存する．

    Args:
        runs (list[str]): 生徒の条件のパス（OUTPUT_DIR からの相対，seed を除く）．

    Returns:
        None
    """
    from common import evaluate

    truth = load_check_labels()
    out = {"n_check": len(truth), "truth_rate": {
        **{f: float((truth["format"] == f).mean()) for f in FORMATS},
        **{c: float(truth[c].mean()) for c in ATTRIBUTES + MEDIA}}}
    for run in runs:
        ann = pl.read_parquet(os.path.join(OUTPUT_DIR, run, "0", "real_annotation_v3.parquet"))
        pred = truth.select("fileId").join(ann, on="fileId", how="left", maintain_order="left")
        m = evaluate(truth, pred.select(["format"] + ATTRIBUTES + MEDIA))
        out[run] = {
            "real_rate": {**{f: float((ann["format"] == f).mean()) for f in FORMATS},
                          **{c: float(ann[c].mean()) for c in ATTRIBUTES + MEDIA}},
            "media_multi_rate": float((ann.select(MEDIA).sum_horizontal() >= 2).mean()),
            "check": {"format_accuracy": m["format_accuracy"], "format_macro_f1": m["format_macro_f1"],
                      "attr_macro_f1": m["attr_macro_f1"], "media_macro_f1": m["media_macro_f1"],
                      "attr_per_item": {k: v["f1"] for k, v in m["attr_per_item"].items()},
                      "media_per_item": {k: (v["f1"], v["support"]) for k, v in m["media_per_item"].items()},
                      "all_correct": float(np.mean([
                          (t["format"] == p["format"]) and all(t[c] == p[c] for c in ATTRIBUTES + MEDIA)
                          for t, p in zip(truth.to_dicts(), pred.to_dicts())]))}}
        errs = [(t["fileId"], codes(t), codes(p)) for t, p in zip(truth.to_dicts(), pred.to_dicts()) if codes(t) != codes(p)]
        out[run]["errors"] = [{"fileId": e[0], "truth": e[1], "pred": e[2]} for e in errs]
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    json.dump(out, open(os.path.join(OUTPUT_DIR, "figures", "real_check.json"), "w"), indent=2, ensure_ascii=False)
    for run in runs:
        print(run, json.dumps(out[run]["check"], ensure_ascii=False))


def main() -> None:
    """コマンドライン引数に従って抽出または集計を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="OUTPUT_DIR からの生徒の条件のパス（seed を除く）")
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.make:
        make(pl.read_parquet(os.path.join(OUTPUT_DIR, args.runs[0], "0", "real_annotation_v3.parquet")))
    if args.analyze:
        analyze(args.runs)


if __name__ == "__main__":
    main()
