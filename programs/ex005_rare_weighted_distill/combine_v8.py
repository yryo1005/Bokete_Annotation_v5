"""order_009 の改善策（v8 の部品）を組み合わせた教師を構成し，評価用データと少ない区分の評価用データで評価する．

教師の構成（TEACHERS_V8）は，combine.TEACHERS の項目に加えて以下の任意の項目を持つ．
- "pc"：絵画と CG の独立の質問（P_v8_*）．媒体の painting と cg をその回答で置き換え，いずれかが真なら
  写真・イラストを偽にする（合成画像の場合を除く）．
- "composite"：合成画像の観測しやすい質問（C_v8_*）．別々の写真や絵が2枚以上，または切り抜きの貼り付けがあれば真とする．
- "tl"：文字の位置と体裁の補助の質問（T_v8_*）．投稿者が重畳した位置（テンプレート，帯，無地の背景，絵の上）にあり，
  回答者に向けた文であれば「お題文あり」とする．"tl_format" が真なら，形式が通常でお題文ありの場合はお題文に，
  形式がお題文で文が元からある文字（original_text）または文がない場合は通常に変える．

評価用データは，通常の評価用データ（640枚，dev/test）と，少ない区分の評価用データ（第1版200枚と第2版600枚，
層ごとに dev/test に二分）である．プロンプトの選択には dev のみを用いる．
結果は outputs/ex005_rare_weighted_distill/teacher_v8_{構成}/rules/0/ に保存する（metrics.json に rare_dev，rare_test を追加）．

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/combine_v8.py
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combine  # noqa: E402
from common import OUTPUT_DIR, save_run  # noqa: E402
from data import DATASET_DIR, MEDIA, load_eval_labels  # noqa: E402
from teacher import output_path  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RARE_CLASSES = ["game", "cg", "painting", "karuta"]
MEDIA_CODE = {"P": "photo", "A": "anime", "G": "game", "C": "cg", "M": "manga", "T": "text_graphic", "E": "painting",
              "I": "illustration"}
OVERLAY_LAYOUTS = ["bokete_template", "added_band", "plain_background", "overlaid_on_picture"]

BASE = {"format": ["F_v2"], "attr": "J_v6", "media": "M_v6"}
TEACHERS_V8 = {
    "base": dict(BASE),
    "F8": {**BASE, "format": ["F_v8_en"]},
    "J8": {**BASE, "attr": "J_v8_en"},
    "M8": {**BASE, "media": "M_v8_en"},
    "PCen": {**BASE, "pc": "P_v8_en"},
    "PCja": {**BASE, "pc": "P_v8_ja"},
    "PCen_paint": {**BASE, "pc": "P_v8_en", "pc_mode": "paint"},
    "PCen_paint_or": {**BASE, "pc": "P_v8_en", "pc_mode": "paint_or"},
    "PCen_cgand": {**BASE, "pc": "P_v8_en", "pc_mode": "cg_and"},
    "PCen_cgvote": {**BASE, "pc": "P_v8_en", "pc_mode": "cg_vote"},
    "PCja_cgand": {**BASE, "pc": "P_v8_ja", "pc_mode": "cg_and"},
    "M6M8paint": {**BASE, "media2": "M_v8_en"},
    "M6M8paint_PCen_cgand": {**BASE, "media2": "M_v8_en", "pc": "P_v8_en", "pc_mode": "cg_and"},
    "F8b": {**BASE, "format": ["F_v8b_en"]},
    "F8c": {**BASE, "format": ["F_v8c_en"]},
    "Cen": {**BASE, "composite": "C_v8_en"},
    "Cja": {**BASE, "composite": "C_v8_ja"},
    "TLen": {**BASE, "tl": "T_v8_en"},
    "TLja": {**BASE, "tl": "T_v8_ja"},
    "TLen_fmt": {**BASE, "tl": "T_v8_en", "tl_format": True},
    "TLja_fmt": {**BASE, "tl": "T_v8_ja", "tl_format": True},
}


def load_rare_eval() -> pl.DataFrame:
    """少ない区分の評価用データ（第1版と第2版）を読み込み，層ごとに dev/test に二分する．

    Returns:
        pl.DataFrame: 列 fileId, set（"rare_eval" または "rare_eval_v2"）, stratum, split, karuta（bool）と
        媒体の各項目（bool）を持つ表．
    """
    rows = []
    for set_name, label_file, ids_file, col in [
            ("rare_eval", "rare_eval_labels.txt", "rare_eval_ids.csv", "pool"),
            ("rare_eval_v2", "rare_eval_labels_v2.txt", "rare_eval_v2_ids.csv", "stratum")]:
        strata = dict(pl.read_csv(os.path.join(DATASET_DIR, ids_file)).select("fileId", col).iter_rows())
        for line in open(os.path.join(THIS_DIR, label_file), encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            fid, f, m = line.split()
            r = {"fileId": int(fid), "set": set_name, "stratum": f"{set_name}:{strata[int(fid)]}", "karuta": f == "K"}
            r.update({v: (k in m) for k, v in MEDIA_CODE.items()})
            rows.append(r)
    df = pl.DataFrame(rows)
    dev, _ = train_test_split(df["fileId"].to_numpy(), test_size=0.5, random_state=0, stratify=df["stratum"].to_numpy())
    return df.with_columns(pl.when(pl.col("fileId").is_in(dev.tolist())).then(pl.lit("dev")).otherwise(pl.lit("test"))
                           .alias("split"))


def build_v8(config: dict, set_name: str, ids: list) -> pl.DataFrame:
    """v8 の補助の質問を含む教師の構成に従って3軸の予測を組み立てる．

    Args:
        config (dict): TEACHERS_V8 の値．
        set_name (str): 部品の出力の集合の名前．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 format, karuta_char と属性・媒体の各項目を持つ表．
    """
    df = combine.build({k: config[k] for k in ["format", "attr", "media"]}, set_name, ids)
    if "media2" in config:  # 2つ目の媒体の部品と，絵画のみ論理和をとる
        m2 = combine.load_component(config["media2"], set_name, ids)
        df = df.with_columns(pl.col("painting") | m2["painting"].fill_null(False))
    if "pc" in config:
        pc = combine.load_component(config["pc"], set_name, ids)
        paint, cg = pc["is_fine_art_painting"].fill_null(False), pc["is_3dcg_or_ai"].fill_null(False)
        mode = config.get("pc_mode", "both")
        if mode in ("paint", "paint_or"):  # CG は媒体の部品の回答のまま
            cg = df["cg"]
        elif mode == "cg_and":  # CG は独立の質問と媒体の部品の両方が真の場合のみ
            cg = cg & (df["cg"] | df["illustration"])
        elif mode == "cg_vote":  # CG は独立の質問が真で，媒体の部品が写真のみでない場合
            cg = cg & ~(df["photo"] & ~df["illustration"] & ~df["cg"])
        if mode == "paint_or":
            paint = paint | df["painting"]
        either = paint | cg
        keep = df["composite"]
        df = df.with_columns(paint.alias("painting"), cg.alias("cg"),
                             pl.when(either & ~keep).then(False).otherwise(pl.col("illustration")).alias("illustration"),
                             pl.when(cg & ~keep).then(False).otherwise(pl.col("photo")).alias("photo"))
        df = df.with_columns(pl.when(pl.sum_horizontal([pl.col(m).cast(int) for m in MEDIA]) == 0).then(True)
                             .otherwise(pl.col("illustration")).alias("illustration"))
    if "composite" in config:
        c = combine.load_component(config["composite"], set_name, ids)
        df = df.with_columns(((c["n_separate_pictures"].fill_null(1) >= 2) | c["has_cutout_paste"].fill_null(False))
                             .alias("composite"))
    if "tl" in config:
        t = combine.load_component(config["tl"], set_name, ids)
        prompt = t["text_layout"].is_in(OVERLAY_LAYOUTS) & t["addresses_respondents"].fill_null(False)
        df = df.with_columns(prompt.alias("has_prompt_text"))
        if config.get("tl_format"):
            original = t["text_layout"].is_in(["original_text", "none"])
            df = df.with_columns(
                pl.when((pl.col("format") == "normal") & prompt).then(pl.lit("odaibun"))
                .when((pl.col("format") == "odaibun") & pl.Series(original)).then(pl.lit("normal"))
                .otherwise(pl.col("format")).alias("format"))
        df = df.with_columns(pl.col("has_prompt_text") | (pl.col("format") == "odaibun"))
    return df


def rare_scores(pred: pl.DataFrame, labels: pl.DataFrame) -> dict:
    """少ない区分ごとの適合率・再現率・F1 と，その平均を求める．

    Args:
        pred (pl.DataFrame): 列 format と媒体の各項目を持つ予測．
        labels (pl.DataFrame): load_rare_eval の戻り値の一部（行の順は pred と同じ）．

    Returns:
        dict: 区分ごとの指標と macro_f1．
    """
    from sklearn.metrics import precision_recall_fscore_support

    out = {}
    for c in RARE_CLASSES:
        p = (pred["format"] == "karuta").to_numpy() if c == "karuta" else pred[c].to_numpy().astype(bool)
        y = labels["karuta" if c == "karuta" else c].to_numpy()
        pp, rr, ff, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        out[c] = {"precision": float(pp), "recall": float(rr), "f1": float(ff), "support": int(y.sum())}
    out["macro_f1"] = float(np.mean([out[c]["f1"] for c in RARE_CLASSES]))
    return out


def evaluate_teacher(name: str, config: dict, labels: pl.DataFrame, rare: pl.DataFrame) -> None:
    """教師の構成を評価用データと少ない区分の評価用データで評価して保存する．

    Args:
        name (str): 構成の名前．
        config (dict): 構成．
        labels (pl.DataFrame): 通常の評価用データ．
        rare (pl.DataFrame): 少ない区分の評価用データ．

    Returns:
        None
    """
    run_dir = os.path.join(OUTPUT_DIR, f"teacher_v8_{name}", "rules", "0")
    pred = build_v8(config, "eval", labels["fileId"].to_list()).drop("karuta_char")
    extra = {}
    rare_comps = {c for k in ["format", "media", "pc", "media2"] for c in ([config[k]] if isinstance(config.get(k), str) else
                                                                 config.get(k, []))}
    rare_comps.add(config["attr"])
    if all(os.path.exists(output_path(c, s)) for c in rare_comps for s in ["rare_eval", "rare_eval_v2"]):
        parts = []
        for s in ["rare_eval", "rare_eval_v2"]:
            sub = rare.filter(pl.col("set") == s)
            parts.append(build_v8({k: v for k, v in config.items() if k not in ("composite", "tl", "tl_format")}, s,
                                  sub["fileId"].to_list()))
        rp = pl.concat([p.select(["format"] + MEDIA) for p in parts])
        for sp in ["dev", "test"]:
            m = (rare["split"] == sp).to_numpy()
            extra[f"rare_{sp}"] = rare_scores(rp.filter(pl.Series(m)), rare.filter(pl.Series(m)))
    m = save_run(run_dir, labels, pred, {"method": "teacher_v8", "config": config}, extra)
    if "rare_dev" in m:
        print(f"    rare dev={m['rare_dev']['macro_f1']:.3f} test={m['rare_test']['macro_f1']:.3f}  "
              + " ".join(f"{c}:{m['rare_dev'][c]['f1']:.2f}" for c in RARE_CLASSES))


def main() -> None:
    """全ての構成を評価する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", default=list(TEACHERS_V8))
    args = parser.parse_args()
    labels = load_eval_labels()
    rare = load_rare_eval()
    for name in args.teachers:
        evaluate_teacher(name, TEACHERS_V8[name], labels, rare)


if __name__ == "__main__":
    main()
