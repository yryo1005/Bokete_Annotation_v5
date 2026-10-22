"""プロンプトの部品の出力を組み合わせて教師を構成し，評価用データで評価する．教師ラベルの作成にも用いる．

教師の構成は TEACHERS に定義する．各構成は，形式・属性・媒体をどの部品から取るかと，後処理の規則の有無を持つ．
形式の vote は複数の部品の形式の多数決である（同数の場合は先頭の部品に従う）．
後処理は，(1) 属性を形式と整合させる（カルタ記号ありは形式がカルタの場合のみ，空欄あり・お題文ありは形式から補う），
(2) ボケの書き出しによるカルタの照合（ex004 の rules.py と同じ規則）である．

評価：outputs/ex005_rare_weighted_distill/teacher_{構成}/default/0/
教師ラベル：datasets/ex005_rare_weighted_distill/teacher_labels/{構成}_{集合}.csv

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/combine.py
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/combine.py --label_sets pool --teachers T_best
"""
import argparse
import importlib.util
import os
import sys
from collections import Counter

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, save_run  # noqa: E402
from data import ATTRIBUTES, DATASET_DIR, MEDIA, PROJECT_DIR, load_eval_labels  # noqa: E402
from teacher import load_ids, output_path  # noqa: E402

_spec = importlib.util.spec_from_file_location("ex004_rules", os.path.join(PROJECT_DIR, "programs", "ex004_four_axis_distill",
                                                                        "rules.py"))
ex004_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex004_rules)

TEACHERS = {
    "T_joint": {"format": ["J_v6"], "attr": "J_v6", "media": "J_v6"},
    "T_v2_joint": {"format": ["F_v2"], "attr": "J_v6", "media": "J_v6"},
    "T_v6_joint": {"format": ["F_v6"], "attr": "J_v6", "media": "J_v6"},
    "T_v6hr_joint": {"format": ["F_v6_hr"], "attr": "J_v6", "media": "J_v6"},
    "T_v6cot_joint": {"format": ["F_v6_cot"], "attr": "J_v6", "media": "J_v6"},
    "T_v6_sep": {"format": ["F_v6"], "attr": "A_v6", "media": "M_v6"},
    "T_v2_sep": {"format": ["F_v2"], "attr": "A_v6", "media": "M_v6"},
    "T_vote_sep": {"format": ["F_v6", "F_v2", "F_v6_hr", "F_v6_cot", "J_v6"], "attr": "A_v6", "media": "M_v6"},
    "T_vote3_sep": {"format": ["F_v6", "F_v2", "F_v6_hr"], "attr": "A_v6", "media": "M_v6"},
    "T_v2_JM": {"format": ["F_v2"], "attr": "J_v6", "media": "M_v6"},
    "T_vote_JM": {"format": ["F_v6", "F_v2", "F_v6_hr", "F_v6_cot", "J_v6"], "attr": "J_v6", "media": "M_v6"},
    "T_v2_J7M7": {"format": ["F_v2"], "attr": "J_v7", "media": "M_v7"},
    "T_v2_J6M7": {"format": ["F_v2"], "attr": "J_v6", "media": "M_v7"},
    "T_v2_J7M6": {"format": ["F_v2"], "attr": "J_v7", "media": "M_v6"},
    "T_v2_J6_Mvote": {"format": ["F_v2"], "attr": "J_v6", "media": ["M_v6", "M_v7", "J_v6"]},
    "T_v2_Avote_Mvote": {"format": ["F_v2"], "attr": ["J_v6", "J_v7", "A_v6"], "media": ["M_v6", "M_v7", "J_v6"]},
    "T_v2_Avote_M6": {"format": ["F_v2"], "attr": ["J_v6", "J_v7", "A_v6"], "media": "M_v6"},
    "T_vote_J6_Mvote": {"format": ["F_v6", "F_v2", "F_v6_hr", "F_v6_cot", "J_v6"], "attr": "J_v6",
                        "media": ["M_v6", "M_v7", "J_v6"]},
    "T_vote_joint": {"format": ["F_v6", "F_v2", "F_v6_hr", "F_v6_cot", "J_v6"], "attr": "J_v6", "media": "J_v6"},
}


def load_component(component: str, set_name: str, ids: list) -> pl.DataFrame:
    """部品の出力を読み込み，ids の順に並べる．

    Args:
        component (str): 部品の名前．
        set_name (str): 集合の名前．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 fileId と部品の項目を持つ表．
    """
    return pl.DataFrame({"fileId": ids}).join(pl.read_csv(output_path(component, set_name)), on="fileId",
                                              how="left", maintain_order="left")


def vote_items(sources, items: list, set_name: str, ids: list) -> pl.DataFrame:
    """1つまたは複数の部品から項目の値を取る．複数の場合は項目ごとの多数決（過半数が真なら真）とする．

    Args:
        sources (str | list[str]): 部品の名前，またはそのリスト．
        items (list[str]): 項目名のリスト．
        set_name (str): 集合の名前．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 karuta_char（先頭の部品にあれば）と各項目（bool）を持つ表．
    """
    sources = [sources] if isinstance(sources, str) else sources
    dfs = [load_component(c, set_name, ids) for c in sources]
    out = {}
    first_kc = next((d["karuta_char"] for d in dfs if "karuta_char" in d.columns), None)
    out["karuta_char"] = first_kc if first_kc is not None else pl.Series([""] * len(ids))
    for k in items:
        votes = sum(d[k].fill_null(False).cast(int) for d in dfs)
        out[k] = votes * 2 > len(dfs)
    return pl.DataFrame(out)


def build(config: dict, set_name: str, ids: list, use_rules: bool = True) -> pl.DataFrame:
    """教師の構成に従って3軸の予測を組み立てる．

    Args:
        config (dict): TEACHERS の値．
        set_name (str): 集合の名前．
        ids (list[int]): お題IDのリスト．
        use_rules (bool): 後処理の規則を適用するか．

    Returns:
        pl.DataFrame: 列 format, karuta_char と属性・媒体の各項目を持つ表（行の順は ids と同じ）．
    """
    fmts = [load_component(c, set_name, ids)["format"].to_list() for c in config["format"]]
    fmt = [v[0] if len(v) == 1 else next(x for x in v if Counter(v)[x] == max(Counter(v).values())) for v in zip(*fmts)]
    a = vote_items(config["attr"], ATTRIBUTES, set_name, ids)
    m = vote_items(config["media"], MEDIA, set_name, ids)
    df = pl.DataFrame({"format": fmt, "karuta_char": a["karuta_char"].fill_null("")})
    df = pl.concat([df, a.select(ATTRIBUTES), m.select(MEDIA)], how="horizontal")
    df = df.with_columns(pl.when(pl.sum_horizontal([pl.col(c).cast(int) for c in MEDIA]) == 0).then(True)
                         .otherwise(pl.col("illustration")).alias("illustration"))
    if not use_rules:
        return df.with_columns(pl.col("has_karuta_mark") | (pl.col("format") == "karuta"),
                               pl.col("has_blank") | (pl.col("format") == "anaume"),
                               pl.col("has_prompt_text") | (pl.col("format") == "odaibun"))
    stats = ex004_rules.boke_kana_stats(ids)
    rate = [ex004_rules.karuta_rate(c or "", n or 0, ch) if ch else float("nan") for c, n, ch in
            zip(stats["kana_counts"], stats["n_kana"], df["karuta_char"])]
    veto = (pl.col("format") == "karuta") & pl.Series(stats["n_kana"].fill_null(0) >= ex004_rules.MIN_KANA) & \
        pl.Series(rate, dtype=pl.Float64).lt(ex004_rules.VETO_RATE)
    df = df.with_columns(pl.when(veto).then(pl.when(pl.col("has_blank")).then(pl.lit("anaume"))
                                            .when(pl.col("has_prompt_text")).then(pl.lit("odaibun"))
                                            .otherwise(pl.lit("normal"))).otherwise(pl.col("format")).alias("format"))
    return df.with_columns((pl.col("format") == "karuta").alias("has_karuta_mark"),
                           pl.col("has_blank") | (pl.col("format") == "anaume"),
                           pl.col("has_prompt_text") | (pl.col("format") == "odaibun"))


def components_of(config: dict) -> set:
    """構成が用いる部品の集合を返す．

    Args:
        config (dict): TEACHERS の値．

    Returns:
        set[str]: 部品の名前の集合．
    """
    as_list = lambda v: [v] if isinstance(v, str) else list(v)  # noqa: E731
    return set(config["format"]) | set(as_list(config["attr"])) | set(as_list(config["media"]))


def main() -> None:
    """評価用データで全ての構成を評価し，必要に応じて教師ラベルを作る．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", default=list(TEACHERS))
    parser.add_argument("--label_sets", nargs="*", default=[])
    args = parser.parse_args()
    labels = load_eval_labels()
    ids = labels["fileId"].to_list()
    for name in args.teachers:
        cfg = TEACHERS[name]
        if not all(os.path.exists(output_path(c, "eval")) for c in components_of(cfg)):
            print(f"  {name}: 部品の出力がないため省略")
            continue
        for rules, tag in [(True, "rules"), (False, "norules")]:
            save_run(os.path.join(OUTPUT_DIR, f"teacher_{name}", tag, "0"), labels,
                     build(cfg, "eval", ids, rules).drop("karuta_char"), {"method": "teacher", "config": cfg,
                                                                          "rules": rules})
        for s in args.label_sets:
            sids = load_ids(s)
            out = pl.concat([pl.DataFrame({"fileId": sids}), build(cfg, s, sids)], how="horizontal")
            os.makedirs(os.path.join(DATASET_DIR, "teacher_labels"), exist_ok=True)
            out.write_csv(os.path.join(DATASET_DIR, "teacher_labels", f"{name}_{s}.csv"))
            print(f"教師ラベルを保存した: {name}_{s}（{len(out)} 枚）", out["format"].value_counts(sort=True).to_dicts())


if __name__ == "__main__":
    main()
