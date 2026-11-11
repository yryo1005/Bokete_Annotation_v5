"""方法ごとの予測を，検証データ620件の目視のラベル（発話の型と，定義書 第3版の画像外の知識）と比べる．

指標（調整用 dev，最終評価用 test のそれぞれ）：
- voice6：6区分の正解率とマクロ F1．voice6_noA：形式が穴埋め・お題文以外（回答を除く5区分）のマクロ F1．
- voice3：3区分（登場人物の言葉，外からの言葉，回答）のマクロ F1．
- outside：画像外の知識（はい）の F1，適合率，再現率．
比べる方法：
- ex014 の規則・教師（第3版の指示）・生徒（画像外の知識は第2版の4段階の O を「はい」とみなす）
- ex015 の教師（teacher_{モデル}/{方法}）と生徒（student/...）

出力：outputs/ex015_boke_labels_v3/figures/summary.csv，per_class.json

実行例:
    .venv_annotate/bin/python programs/ex015_boke_labels_v3/evaluate.py
"""
import glob
import json
import os
import sys

import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX13_DIR, EX14_OUTPUT, OUTPUT_DIR, VOICE3, _load, load_eval  # noqa: E402

R13 = _load("ex013_rules_for_v3", os.path.join(EX13_DIR, "rules.py"))
FIG = os.path.join(OUTPUT_DIR, "figures")
VOICES = ["S", "M", "T", "N", "L", "A"]
EX14_METHODS = ["teacher_q32_v3", "teacher_q30a3_v3", "student/aug_all_w__ruri_fmt", "student/aug_all_w__ruri_fmt__stack",
                "student/aug_all_w__ruri_fmt_img__stack"]


def scores(t: pl.DataFrame, p: pl.DataFrame) -> dict:
    """指標を求める．

    Args:
        t (pl.DataFrame): 目視のラベル（列 voice, outside, format）．
        p (pl.DataFrame): 予測（t と同じ順．列 voice, outside）．

    Returns:
        dict: 指標と，発話の型のクラスごとの F1．
    """
    y, q = t["voice"].to_list(), p["voice"].to_list()
    r = {"voice6_acc": accuracy_score(y, q), "voice6_f1": f1_score(y, q, labels=VOICES, average="macro", zero_division=0)}
    m = ~t["format"].is_in(["anaume", "odaibun"]).to_numpy()
    yn, qn = [a for a, k in zip(y, m) if k], [a for a, k in zip(q, m) if k]
    r["voice6_noA_f1"] = f1_score(yn, qn, labels=VOICES[:5], average="macro", zero_division=0)
    y3, q3 = [VOICE3[v] for v in y], [VOICE3[v] for v in q]
    r["voice3_acc"] = accuracy_score(y3, q3)
    r["voice3_f1"] = f1_score(y3, q3, labels=["C", "X", "A"], average="macro", zero_division=0)
    if p["outside"].null_count() == 0:
        yo, qo = t["outside"].to_numpy(), p["outside"].to_numpy().astype(bool)
        r.update({"outside_f1": f1_score(yo, qo, zero_division=0), "outside_prec": precision_score(yo, qo, zero_division=0),
                  "outside_rec": recall_score(yo, qo, zero_division=0), "outside_rate": float(qo.mean())})
    r["voice6_per_class"] = dict(zip(VOICES, f1_score(y, q, labels=VOICES, average=None, zero_division=0).round(3).tolist()))
    r["voice3_per_class"] = dict(zip(["C", "X", "A"], f1_score(y3, q3, labels=["C", "X", "A"], average=None, zero_division=0).round(3).tolist()))
    return r


def collect(ev: pl.DataFrame) -> dict:
    """全ての方法の予測を集める．

    Args:
        ev (pl.DataFrame): 検証データ．

    Returns:
        dict[str, pl.DataFrame]: 方法名 → 列 no, voice, outside の表（ev と同じ順）．
    """
    preds = {"ex014/rules": pl.DataFrame({"no": ev["no"], "voice": [R13.rule_voice(t, f) for t, f in zip(ev["text"], ev["format"])],
                                          "outside": [None] * ev.height}, schema_overrides={"outside": pl.Boolean})}

    def read(path: str) -> pl.DataFrame:
        """予測の表を読み，列 voice, outside にそろえる．

        Args:
            path (str): eval.csv のパス．

        Returns:
            pl.DataFrame: 予測の表．
        """
        d = pl.read_csv(path, schema_overrides={"knowledge": pl.String})
        if "outside" not in d.columns:
            d = d.with_columns((pl.col("knowledge") == "O").alias("outside"))
        return ev.select("no").join(d.select("no", "voice", "outside"), on="no", how="left", maintain_order="left")

    for m in EX14_METHODS:
        for path in sorted(glob.glob(os.path.join(EX14_OUTPUT, m, "**", "eval.csv"), recursive=True)):
            preds["ex014/" + os.path.relpath(os.path.dirname(path), EX14_OUTPUT)] = read(path)
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "eval.csv"), recursive=True)):
        preds[os.path.relpath(os.path.dirname(path), OUTPUT_DIR)] = read(path)
    return preds


def main() -> None:
    """全ての方法を評価して保存する．

    Returns:
        None
    """
    ev = load_eval()
    rows, detail = [], {}
    for name, p in collect(ev).items():
        for split in ["dev", "test"]:
            m = (ev["split"] == split).to_numpy()
            s = scores(ev.filter(m), p.filter(m))
            detail[f"{name}|{split}"] = s
            rows.append({"method": name, "split": split, **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    os.makedirs(FIG, exist_ok=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(os.path.join(FIG, "summary.csv"))
    json.dump(detail, open(os.path.join(FIG, "per_class.json"), "w"), indent=1, ensure_ascii=False)
    cols = ["voice6_f1", "voice6_noA_f1", "voice3_f1", "outside_f1", "outside_prec", "outside_rec"]
    g = df.with_columns(pl.col("method").str.replace(r"/\d+$", "").alias("m")).group_by("m", "split").agg([pl.col(c).mean() for c in cols])
    g = g.pivot(on="split", index="m", values=cols)
    g = g.select(["m"] + [f"{c}_{sp}" for c in cols for sp in ["dev", "test"]])
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=300, float_precision=3):
        print(g.sort("voice6_f1_dev", descending=True))


if __name__ == "__main__":
    main()
