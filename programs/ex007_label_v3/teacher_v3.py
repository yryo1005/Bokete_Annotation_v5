"""定義書 v3 の教師 T_v9 を部品の出力から組み立て，評価し，教師ラベルを作る．

教師 T_v9 の構成：
- 形式：F_v2（2軸用の英語のプロンプト．text は odaibun に読み替える）
- 属性：J_v6 の回答から，空欄あり・カルタ記号あり・文字あり（お題文あり OR その他の文字あり）を作る
- 媒体：M_v9（CG の選択肢なし）．絵画は P_v8_en の is_fine_art_painting で置き換え，絵画が真なら（文字ありでも）イラストを偽にする
- 後処理：媒体が空ならイラスト，ボケの書き出しによるカルタの照合，属性を形式と整合させる
比較のため，旧教師 T_v8 の回答を v3 に読み替えた教師（CG を削除して空ならイラスト）も評価する．

出力：outputs/ex007_label_v3/teacher_{T_v9,T_v8_mapped}/rules/0/，datasets/ex005_rare_weighted_distill/teacher_labels/T_v9_{pool,rare}.csv

実行例:
    .venv_annotate/bin/python programs/ex007_label_v3/teacher_v3.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ATTRIBUTES, DATASET_DIR, EX5_DIR, MEDIA, _load, load_eval_labels, load_rare_labels  # noqa: E402

RULES = _load("ex004_rules_for_v3", os.path.join(os.path.dirname(EX5_DIR), "ex004_four_axis_distill", "rules.py"))
T = _load("ex005_teacher_for_v3", os.path.join(EX5_DIR, "teacher.py"))
COMMON = _load("ex007_common", os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py"))
RARE_SETS = [f"rare_r{i:02d}" for i in range(5)]


def _comp(name: str, set_name: str, ids: list) -> pl.DataFrame:
    """部品の出力を ids の順に読み込む．

    Args:
        name (str): 部品の名前．
        set_name (str): 集合の名前．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 部品の項目の表．
    """
    return pl.DataFrame({"fileId": ids}).join(pl.read_csv(T.output_path(name, set_name)), on="fileId", how="left",
                                              maintain_order="left")


def finalize(df: pl.DataFrame, ids: list) -> pl.DataFrame:
    """後処理（媒体の補完，ボケによるカルタの照合，属性と形式の整合）を適用する．

    Args:
        df (pl.DataFrame): 列 format, karuta_char と属性・媒体の各項目を持つ表．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 後処理後の表．
    """
    R = RULES
    df = df.with_columns(pl.when(pl.sum_horizontal([pl.col(m).cast(int) for m in MEDIA]) == 0).then(True)
                         .otherwise(pl.col("illustration")).alias("illustration"))
    stats = R.boke_kana_stats(ids)
    rate = [R.karuta_rate(c or "", n or 0, ch) if ch else float("nan") for c, n, ch in
            zip(stats["kana_counts"], stats["n_kana"], df["karuta_char"].fill_null(""))]
    veto = (pl.col("format") == "karuta") & pl.Series(stats["n_kana"].fill_null(0) >= R.MIN_KANA) & \
        pl.Series(rate, dtype=pl.Float64).lt(R.VETO_RATE)
    df = df.with_columns(pl.when(veto).then(pl.when(pl.col("has_blank")).then(pl.lit("anaume"))
                                            .when(pl.col("has_text")).then(pl.lit("odaibun")).otherwise(pl.lit("normal")))
                         .otherwise(pl.col("format")).alias("format"))
    return df.with_columns((pl.col("format") == "karuta").alias("has_karuta_mark"),
                           pl.col("has_blank") | (pl.col("format") == "anaume"),
                           pl.col("has_text") | (pl.col("format") == "odaibun"))


def build_v9(set_name: str, ids: list, media_component: str = "M_v9") -> pl.DataFrame:
    """教師 T_v9 の予測を組み立てる．

    Args:
        set_name (str): 部品の出力の集合の名前．
        ids (list[int]): お題IDのリスト．
        media_component (str): 媒体の部品の名前（比較用に M_v6 も指定できる）．

    Returns:
        pl.DataFrame: 列 format, karuta_char と属性3項目・媒体7項目を持つ表．
    """
    f = _comp("F_v2", set_name, ids)
    a = _comp("J_v6", set_name, ids)
    m = _comp(media_component, set_name, ids)
    p = _comp("P_v8_en", set_name, ids)
    paint = p["is_fine_art_painting"].fill_null(False)
    df = pl.DataFrame({"format": f["format"], "karuta_char": a["karuta_char"].fill_null("").cast(pl.String),
                       "has_blank": a["has_blank"].fill_null(False), "has_karuta_mark": a["has_karuta_mark"].fill_null(False),
                       "has_text": a["has_prompt_text"].fill_null(False) | a["has_other_text"].fill_null(False)})
    media = {k: m[k].fill_null(False) for k in MEDIA}
    media["painting"] = paint
    media["illustration"] = media["illustration"] & ~paint
    df = pl.concat([df, pl.DataFrame(media)], how="horizontal")
    return finalize(df, ids)


def build_v8_mapped(set_name: str, ids: list) -> pl.DataFrame:
    """旧教師 T_v8 の回答を v3 に読み替えた予測を作る（媒体は M_v6 から CG を削除し，空ならイラスト）．

    T_v8 の CG の規則は CG を削除するため不要であり，絵画の置き換えは T_v9 と同じ規則を用いる．

    Args:
        set_name (str): 集合の名前．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 format, karuta_char と属性3項目・媒体7項目を持つ表．
    """
    return build_v9(set_name, ids, media_component="M_v6")


def rare_scores(pred: pl.DataFrame, rare: pl.DataFrame) -> dict:
    """少ない区分（ゲーム，絵画，カルタ）の F1 と平均を，dev と test で求める．

    Args:
        pred (pl.DataFrame): 少ない区分の評価用データの予測（列 format と媒体）．
        rare (pl.DataFrame): load_rare_labels の戻り値．

    Returns:
        dict: rare_dev，rare_test．
    """
    from sklearn.metrics import f1_score

    out = {}
    for sp in ["dev", "test"]:
        m = (rare["split"] == sp).to_numpy()
        r = {}
        for c in ["game", "painting", "karuta"]:
            y = rare["karuta" if c == "karuta" else c].to_numpy()[m]
            q = ((pred["format"] == "karuta").to_numpy() if c == "karuta" else pred[c].to_numpy().astype(bool))[m]
            r[c] = float(f1_score(y, q, zero_division=0))
        r["macro_f1"] = float(np.mean(list(r.values())))
        out[f"rare_{sp}"] = r
    return out


def make_labels(name: str, fn) -> None:
    """教師ラベル（基本の4万枚と追加分）を保存する．

    Args:
        name (str): 教師ラベルの名前（"T_v9" または "T_v8m"）．
        fn (callable): build_v9 または build_v8_mapped．

    Returns:
        None
    """
    d = os.path.join(DATASET_DIR, "teacher_labels")
    ids = T.load_ids("pool")
    pool = pl.concat([pl.DataFrame({"fileId": ids}), fn("pool", ids).drop("karuta_char")], how="horizontal")
    pool.write_csv(os.path.join(d, f"{name}_pool.csv"))
    rare_lab = pl.concat([pl.concat([pl.DataFrame({"fileId": T.load_ids(s)}), fn(s, T.load_ids(s)).drop("karuta_char")],
                                    how="horizontal") for s in RARE_SETS])
    rare_lab.write_csv(os.path.join(d, f"{name}_rare.csv"))
    counts = {"pool": {c: int(pool[c].sum()) for c in ATTRIBUTES + MEDIA},
              "rare": {c: int(rare_lab[c].sum()) for c in ATTRIBUTES + MEDIA}}
    json.dump(counts, open(os.path.join(COMMON.OUTPUT_DIR, f"teacher_label_counts_{name}.json"), "w"), indent=2)
    print(name, counts)


def main() -> None:
    """教師を評価し，教師ラベルを作る．

    Returns:
        None
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="*", default=["T_v9", "T_v8m"])
    args = parser.parse_args()
    labels, rare = load_eval_labels(), load_rare_labels()
    builders = {"T_v9": build_v9, "T_v8m": build_v8_mapped}
    for name, fn in builders.items():
        pred = fn("eval", labels["fileId"].to_list()).drop("karuta_char")
        rp = pl.concat([fn(s, rare.filter(pl.col("set") == s)["fileId"].to_list()).select(["format"] + MEDIA)
                        for s in ["rare_eval", "rare_eval_v2"]])
        m = COMMON.save_run(os.path.join(COMMON.OUTPUT_DIR, f"teacher_{name}", "rules", "0"), labels, pred,
                            {"method": "teacher", "name": name}, rare_scores(rp, rare))
        print(f"    rare dev={m['rare_dev']['macro_f1']:.3f} test={m['rare_test']['macro_f1']:.3f}")
    for name in args.labels:
        make_labels(name, builders[name])


if __name__ == "__main__":
    main()
