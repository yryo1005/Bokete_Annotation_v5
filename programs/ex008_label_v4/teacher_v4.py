"""定義書 v4 の教師を部品の出力から組み立て，評価し，教師ラベルを作る．

教師の構成（T_v8m を基にし，形式と文字ありの部品を入れ替える）：
- 形式：F_v2，F_v9a，F_v9b，F_v9c のいずれか（F_v9a/b は F_v2 の text の説明に「画像に元からある文字はお題文ではない」を，
  F_v9c は normal の説明に同じ趣旨を加えたもの）．確認ありの教師は，形式がお題文の画像のうち O_v10_en で
  投稿者の問いではないと答えたものを通常にする
- 属性：J_v6 の回答から，空欄あり・カルタ記号ありを作る．文字ありは，J_v6（お題文あり OR その他の文字あり），
  X_v10_en（ロゴ等を含め読める文字があるか），両者の論理和（JX），または J_v6 と「X_v10_en かつカルタ記号なし」の
  論理和（JXk．X_v10_en はカルタ記号だけの画像も文字ありとするため）から作る
- 媒体：M_v6 の回答から CG を削除する．絵画は P_v8_en で置き換え，絵画が真ならイラストを偽にする
- 後処理：媒体が空ならイラスト，ボケの書き出しによるカルタの照合，属性を形式と整合させる

教師の名前と構成は TEACHERS に示す（T_v8m は ex007 と同じ構成で，比較用）．

出力：outputs/ex008_label_v4/teacher_{名前}/rules/0/，datasets/ex005_rare_weighted_distill/teacher_labels/{名前}_{pool,rare}.csv

実行例:
    .venv_annotate/bin/python programs/ex008_label_v4/teacher_v4.py                 # 評価のみ
    .venv_annotate/bin/python programs/ex008_label_v4/teacher_v4.py --labels T_v10  # 教師ラベルも作る
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, save_run  # noqa: E402
from data import (ATTRIBUTES, DATASET_DIR, EX5_DIR, MEDIA, PROJECT_DIR, _load, load_eval_labels,  # noqa: E402
                  load_rare_labels)

RULES = _load("ex004_rules_for_v4", os.path.join(os.path.dirname(EX5_DIR), "ex004_four_axis_distill", "rules.py"))
RARE_SETS = [f"rare_r{i:02d}" for i in range(5)]
# 教師の名前 → (形式の部品，お題文の確認の有無，文字ありの出所)
TEACHERS = {"T_v8m": ("F_v2", False, "J"), "T_v10a": ("F_v9a", False, "J"), "T_v10b": ("F_v9b", False, "J"),
            "T_v10c": ("F_v9c", False, "J"), "T_v10o": ("F_v2", True, "J"),
            "T_v10x": ("F_v2", False, "X"), "T_v10ox": ("F_v2", True, "X"), "T_v10oJX": ("F_v2", True, "JX"),
            "T_v10oJXk": ("F_v2", True, "JXk"), "T_v10coJXk": ("F_v9c", True, "JXk"),
            # order_013：確認の質問を典型的なお題文の型を明示した O_v11_en に替えた教師（spec の4番目は確認の質問の部品）
            "T_v11": ("F_v2", True, "JXk", "O_v11_en")}


def component_path(name: str, set_name: str) -> str:
    """部品の出力の CSV のパスを返す（ex005 の teacher.py と同じ規則）．

    Args:
        name (str): 部品の名前．
        set_name (str): 集合の名前．

    Returns:
        str: CSV のパス．
    """
    if set_name == "eval":
        return os.path.join(PROJECT_DIR, "outputs", "ex005_rare_weighted_distill", "teacher_components", name, "eval.csv")
    return os.path.join(DATASET_DIR, "component_labels", f"{name}_{set_name}.csv")


def load_ids(set_name: str) -> list:
    """集合の名前からお題IDのリストを得る．

    Args:
        set_name (str): "eval"，"pool"，または candidates/{集合}.csv の名前．

    Returns:
        list[int]: お題IDのリスト．
    """
    if set_name == "eval":
        return load_eval_labels()["fileId"].to_list()
    if set_name == "pool":
        return _load("ex005_data_for_v3", os.path.join(EX5_DIR, "data.py")).load_pool()["fileId"].to_list()
    return pl.read_csv(os.path.join(DATASET_DIR, "candidates", f"{set_name}.csv"))["fileId"].to_list()


def _comp(name: str, set_name: str, ids: list) -> pl.DataFrame:
    """部品の出力を ids の順に読み込む．

    Args:
        name (str): 部品の名前．
        set_name (str): 集合の名前．
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 部品の項目の表．
    """
    df = pl.read_csv(component_path(name, set_name)).unique("fileId", keep="first")
    return pl.DataFrame({"fileId": ids}).join(df, on="fileId", how="left", maintain_order="left")


def finalize(df: pl.DataFrame, ids: list) -> pl.DataFrame:
    """後処理（媒体の補完，ボケによるカルタの照合，属性と形式の整合）を適用する（ex007 と同じ）．

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


def build(set_name: str, ids: list, spec: tuple) -> pl.DataFrame:
    """教師の予測を組み立てる．

    Args:
        set_name (str): 部品の出力の集合の名前．
        ids (list[int]): お題IDのリスト．
        spec (tuple): (形式の部品の名前，お題文の確認の有無，文字ありの出所 "J"・"X"・"JX"・"JXk"[，確認の質問の部品])．

    Returns:
        pl.DataFrame: 列 format, karuta_char と属性3項目・媒体7項目を持つ表．
    """
    format_component, verify, text_source = spec[:3]
    verify_component = spec[3] if len(spec) > 3 else "O_v10_en"
    f = _comp(format_component, set_name, ids)
    a = _comp("J_v6", set_name, ids)
    m = _comp("M_v6", set_name, ids)
    paint = _comp("P_v8_en", set_name, ids)["is_fine_art_painting"].fill_null(False)
    fmt = f["format"].fill_null("normal").replace({"text": "odaibun"})
    if verify:
        ok = _comp(verify_component, set_name, ids)["is_poster_prompt"].fill_null(True)
        fmt = pl.Series("format", np.where((fmt == "odaibun").to_numpy() & ~ok.to_numpy(), "normal", fmt.to_numpy()))
    text_j = a["has_prompt_text"].fill_null(False) | a["has_other_text"].fill_null(False)
    text_x = _comp("X_v10_en", set_name, ids)["has_any_text"].fill_null(False) if "X" in text_source else None
    if text_source == "J":
        has_text = text_j
    elif text_source == "X":
        has_text = text_x
    elif text_source == "JX":
        has_text = text_j | text_x
    else:
        has_text = text_j | (text_x & ~a["has_karuta_mark"].fill_null(False))
    df = pl.DataFrame({"format": fmt, "karuta_char": a["karuta_char"].fill_null("").cast(pl.String),
                       "has_blank": a["has_blank"].fill_null(False), "has_karuta_mark": a["has_karuta_mark"].fill_null(False),
                       "has_text": has_text})
    media = {k: m[k].fill_null(False) for k in MEDIA}
    media["painting"] = paint
    media["illustration"] = media["illustration"] & ~paint
    return finalize(pl.concat([df, pl.DataFrame(media)], how="horizontal"), ids)


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
        mask = (rare["split"] == sp).to_numpy()
        r = {}
        for c in ["game", "painting", "karuta"]:
            y = rare["karuta" if c == "karuta" else c].to_numpy()[mask]
            q = ((pred["format"] == "karuta").to_numpy() if c == "karuta" else pred[c].to_numpy().astype(bool))[mask]
            r[c] = float(f1_score(y, q, zero_division=0))
        r["macro_f1"] = float(np.mean(list(r.values())))
        out[f"rare_{sp}"] = r
    return out


def make_labels(name: str) -> None:
    """教師ラベル（基本の4万枚と追加分）を保存する．

    Args:
        name (str): 教師の名前（TEACHERS のキー）．

    Returns:
        None
    """
    d = os.path.join(DATASET_DIR, "teacher_labels")
    fc = TEACHERS[name]
    ids = load_ids("pool")
    pool = pl.concat([pl.DataFrame({"fileId": ids}), build("pool", ids, fc).drop("karuta_char")], how="horizontal")
    pool.write_csv(os.path.join(d, f"{name}_pool.csv"))
    rare_lab = pl.concat([pl.concat([pl.DataFrame({"fileId": load_ids(s)}), build(s, load_ids(s), fc).drop("karuta_char")],
                                    how="horizontal") for s in RARE_SETS])
    rare_lab.write_csv(os.path.join(d, f"{name}_rare.csv"))
    counts = {"pool": {**{f: int((pool["format"] == f).sum()) for f in ["karuta", "anaume", "odaibun", "normal"]},
                       **{c: int(pool[c].sum()) for c in ATTRIBUTES + MEDIA}},
              "rare": {c: int(rare_lab[c].sum()) for c in ATTRIBUTES + MEDIA}}
    json.dump(counts, open(os.path.join(OUTPUT_DIR, f"teacher_label_counts_{name}.json"), "w"), indent=2)
    print(name, counts)


def predict_check(name: str) -> pl.DataFrame:
    """実データの目視ラベルの画像（real_check_v4）に対する教師の予測を返す．

    Args:
        name (str): 教師の名前．

    Returns:
        pl.DataFrame: 列 fileId, format と各項目を持つ表．
    """
    ids = load_ids("real_check_v4")
    return pl.concat([pl.DataFrame({"fileId": ids}), build("real_check_v4", ids, TEACHERS[name]).drop("karuta_char")],
                     how="horizontal")


def main() -> None:
    """教師を評価し，指定された教師の教師ラベルを作る．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", default=list(TEACHERS))
    parser.add_argument("--labels", nargs="*", default=[])
    args = parser.parse_args()
    labels, rare = load_eval_labels(), load_rare_labels()
    for name in args.teachers:
        fc = TEACHERS[name]
        pred = build("eval", labels["fileId"].to_list(), fc).drop("karuta_char")
        rp = pl.concat([build(s, rare.filter(pl.col("set") == s)["fileId"].to_list(), fc).select(["format"] + MEDIA)
                        for s in ["rare_eval", "rare_eval_v2"]])
        run_dir = os.path.join(OUTPUT_DIR, f"teacher_{name}", "rules", "0")
        m = save_run(run_dir, labels, pred, {"method": "teacher", "name": name, "spec": list(fc)},
                     rare_scores(rp, rare))
        predict_check(name).write_csv(os.path.join(run_dir, "check_predictions.csv"))
        print(f"    rare dev={m['rare_dev']['macro_f1']:.3f} test={m['rare_test']['macro_f1']:.3f}")
    for name in args.labels:
        make_labels(name)


if __name__ == "__main__":
    main()
