"""ex017 の画像外の知識の判定を，検証データ（合意ラベル）で評価する．

対象：
- 教師（32B）の指示の変種（outputs/ex017_boke_outside/teacher/{方法}/labeled.csv）．検証データの項目のみを評価する．
- 生徒（outputs/ex017_boke_outside/ft/**/eval.csv）と，比較のための ex016 の生徒（二段階の微調整）．
- アンサンブル（--ensemble に指定した生徒の画像外の知識の確率の平均）．
指標：画像外の知識の F1・適合率・再現率・AUC（調整用 dev，最終評価用 test），参考として発話の型6区分のマクロ F1．
最終評価用の F1 の95%信頼区間（ブートストラップ1,000回）も求める．

出力：outputs/ex017_boke_outside/figures/summary.csv

実行例:
    .venv_annotate/bin/python programs/ex017_boke_outside/evaluate.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX16_OUTPUT, OUTPUT_DIR, VOICES, load_eval  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
EX16 = {"ex016/q4_stage2": "ft/q4/teacher_lr0.0001_ep2_bs16_r16__stage2/0", "ex016/iv4_stage2": "ft/iv4/teacher_lr0.0001_ep2_bs16_r16__stage2/0",
        "ex016/iv2_stage2": "ft/iv2/teacher_lr0.0001_ep1_bs16_r16__stage2/0", "ex016/q4_stage1": "ft/q4/teacher_lr0.0001_ep2_bs16_r16/0"}


def collect(ev: pl.DataFrame) -> dict:
    """全ての予測を集める．

    Args:
        ev (pl.DataFrame): 検証データ．

    Returns:
        dict[str, pl.DataFrame]: 方法名 → 列 no, voice（なければ null）, p_outside の表（ev と同じ順）．
    """
    out = {}
    for name, rel in EX16.items():
        out[name] = pl.read_csv(os.path.join(EX16_OUTPUT, rel, "eval.csv"))
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "ft", "**", "eval.csv"), recursive=True)):
        out[os.path.relpath(os.path.dirname(p), os.path.join(OUTPUT_DIR, "ft")).replace("/0", "")] = pl.read_csv(p)
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "teacher", "*", "labeled.csv"))):
        t = pl.read_csv(p).filter(pl.col("key").str.starts_with("e"))
        out["teacher/" + p.split(os.sep)[-2]] = t.select(pl.col("key").str.slice(1).cast(pl.Int64).alias("no"),
                                                         pl.lit(None, dtype=pl.String).alias("voice"), "p_outside")
    return {k: ev.select("no").join(v.select("no", "voice", "p_outside"), on="no", how="left", maintain_order="left") for k, v in out.items()}


def scores(y: np.ndarray, p: np.ndarray, rng: np.random.Generator = None) -> dict:
    """画像外の知識の指標を求める．

    Args:
        y (np.ndarray): 正解（bool）．
        p (np.ndarray): 確率．
        rng (np.random.Generator): ブートストラップの乱数（None なら信頼区間を求めない）．

    Returns:
        dict: f1, prec, rec, auc（と f1_lo, f1_hi）．
    """
    q = p > 0.5
    r = {"f1": f1_score(y, q, zero_division=0), "prec": precision_score(y, q, zero_division=0), "rec": recall_score(y, q, zero_division=0),
         "auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else float("nan")}
    if rng is not None:
        bs = [f1_score(y[b], q[b], zero_division=0) for b in (rng.choice(len(y), len(y)) for _ in range(1000))]
        r["f1_lo"], r["f1_hi"] = np.percentile(bs, [2.5, 97.5])
    return r


def main() -> None:
    """評価して保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", nargs="*", default=[], help="確率を平均する方法の名前（2つ以上）．複数の組は ; で区切る")
    args = parser.parse_args()
    ev = load_eval()
    preds = collect(ev)
    for spec in " ".join(args.ensemble).split(";"):
        names = spec.split()
        if len(names) >= 2:
            preds["ens:" + "+".join(names)] = preds[names[0]].with_columns(
                pl.Series("p_outside", np.mean([preds[n]["p_outside"].to_numpy() for n in names], 0)), pl.lit(None).alias("voice"))
    rows = []
    for name, p in preds.items():
        for split in ["dev", "test"]:
            m = (ev["split"] == split).to_numpy()
            s = scores(ev["outside"].to_numpy()[m], p["p_outside"].fill_null(0).to_numpy()[m],
                       np.random.default_rng(0) if split == "test" else None)
            if p["voice"].null_count() == 0:
                s["voice6_f1"] = f1_score(ev["voice"].to_numpy()[m], p["voice"].to_numpy()[m], labels=VOICES, average="macro", zero_division=0)
            rows.append({"method": name, "split": split, **s})
    os.makedirs(FIG, exist_ok=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(os.path.join(FIG, "summary.csv"))
    g = df.pivot(on="split", index="method", values=["f1", "prec", "rec", "auc", "voice6_f1"])
    g = g.join(df.filter(pl.col("split") == "test").select("method", "f1_lo", "f1_hi"), on="method")
    with pl.Config(tbl_rows=80, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=300, float_precision=3):
        print(g.sort("f1_dev", descending=True))


if __name__ == "__main__":
    main()
