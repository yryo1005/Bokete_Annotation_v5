"""お題文ありの判定の構成（生徒単独，生徒＋Qwen の段階的な構成）を評価し，全データでの処理時間を見積もる．

構成：
- current：現在のパッケージ（形式がお題文．穴埋め・カルタの画像のお題文は付かない）
- qwen：Qwen（QWEN）を全ての画像に問う
- student：学習し直した生徒（seed の平均の確率）．閾値は dev で F1 が最大になる値
- cascade：生徒の確率が hi 以上なら真，lo 未満なら偽，その間は Qwen の答え．lo と hi は dev で F1 が最大になる組
  （同じ F1 なら Qwen に問う割合が小さい組）
- hybrid：空欄・カルタ記号のない画像は現在のパッケージ（形式がお題文），穴埋め・カルタと予測した画像は学習し直した生徒（閾値は student と同じ）
- hybrid+qwen：hybrid の穴埋め・カルタの画像で，生徒の確率が 0.2 以上の画像に Qwen を問い，真の場合のみお題文ありとする

属性から逆算した形式：カルタ記号あり → カルタ，空欄あり → 穴埋め，お題文あり → お題文，それ以外 → 通常．
空欄あり・カルタ記号ありは現在のパッケージの値を用いる．

出力：outputs/ex024_odaibun_attr/pipeline/（metrics.csv，format.csv，config.json，student_all.parquet，fig_*.png）

実行例:
    .venv_annotate/bin/python programs/ex024_odaibun_attr/evaluate_pipeline.py --teacher Qwen3-VL-32B-Instruct-FP8__802816__P3
"""
import argparse
import glob
import itertools
import json
import os
import sys

import matplotlib
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ANN_DIR, ANNOTATION, OUTPUT_DIR, load_truth  # noqa: E402
from evaluate_prompts import prf  # noqa: E402
from model import OdaibunMLP  # noqa: E402
from train import PROB_COLS  # noqa: E402

QWEN = "Qwen3-VL-32B-Instruct-FP8__802816/P3"
GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
FONT = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
N_BOOT = 2000


def student_runs(teacher: str) -> dict:
    """学習した生徒の条件（手法と学習率）ごとに，seed の平均の確率を求める．

    Args:
        teacher (str): 教師の名前．

    Returns:
        dict[str, pl.DataFrame]: 条件名 → 列 fileId, prob の表．
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(OUTPUT_DIR, "student", teacher, "*", "*"))):
        files = glob.glob(os.path.join(d, "*", "eval_probs.csv"))
        if not files:
            continue
        df = pl.concat([pl.read_csv(f) for f in files]).group_by("fileId").agg(pl.col("prob").mean()).sort("fileId")
        out[os.path.relpath(d, os.path.join(OUTPUT_DIR, "student", teacher))] = df
    return out


def best_threshold(t: np.ndarray, p: np.ndarray) -> float:
    """F1 が最大になる閾値を GRID から選ぶ．

    Args:
        t (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の確率．

    Returns:
        float: 閾値．
    """
    w = np.ones(len(t))
    return max(GRID, key=lambda th: prf(t, p >= th, w)["f1"])


def best_band(t: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple:
    """段階的な構成の lo と hi を選ぶ（F1 が最大．同じなら Qwen に問う割合が小さい組）．

    Args:
        t (np.ndarray): 形状 (N,) の正解．
        p (np.ndarray): 形状 (N,) の生徒の確率．
        q (np.ndarray): 形状 (N,) の Qwen の答え．

    Returns:
        tuple[float, float]: (lo, hi)．
    """
    w = np.ones(len(t))
    cands = [(lo, hi) for lo, hi in itertools.product(GRID, GRID + [1.01]) if lo < hi]
    return max(cands, key=lambda b: (round(prf(t, cascade(p, q, *b), w)["f1"], 4), -((p >= b[0]) & (p < b[1])).mean()))


def cascade(p: np.ndarray, q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """段階的な構成の判定を求める．

    Args:
        p (np.ndarray): 形状 (N,) の生徒の確率．
        q (np.ndarray): 形状 (N,) の Qwen の答え．
        lo (float): これ未満は偽．
        hi (float): これ以上は真．

    Returns:
        np.ndarray: 形状 (N,) の判定．
    """
    return (p >= hi) | ((p >= lo) & (p < hi) & q)


def boot_ci(truth: pl.DataFrame, t: np.ndarray, pred: np.ndarray) -> tuple:
    """実データの F1 のブートストラップの95%区間（無作為抽出全体と層別抽出の層ごとに復元抽出）．

    Args:
        truth (pl.DataFrame): 正解の表．
        t (np.ndarray): 形状 (N,) の正解．
        pred (np.ndarray): 形状 (N,) の予測．

    Returns:
        tuple[float, float]: 2.5%点と97.5%点．
    """
    rng = np.random.default_rng(0)
    groups = truth.with_row_index("i").filter(pl.col("split") == "real").group_by("set", "stratum").agg("i")["i"].to_list()
    w = truth["w"].to_numpy()
    f = []
    for _ in range(N_BOOT):
        b = np.concatenate([rng.choice(np.asarray(g), len(g)) for g in groups])
        f.append(prf(t[b], pred[b], w[b])["f1"])
    return float(np.nanquantile(f, 0.025)), float(np.nanquantile(f, 0.975))


def predict_all(run_dir: str, method: str, out: str) -> pl.DataFrame:
    """全データの代表の画像に生徒を適用する（seed の平均）．

    Args:
        run_dir (str): 生徒の条件のディレクトリ（seed のディレクトリを含む）．
        method (str): 手法（emb / emb+probs）．
        out (str): 保存先（parquet）．

    Returns:
        pl.DataFrame: 列 rep_odaiId, prob を持つ表．
    """
    if os.path.exists(out):
        return pl.read_parquet(out)
    models = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*", "best_model.pt"))):
        m = OdaibunMLP(1152 if method == "emb" else 1159).cuda().eval()
        m.load_state_dict(torch.load(f))
        models.append(m)
    parts = []
    for path in tqdm(sorted(glob.glob(os.path.join(ANN_DIR, "predict", "part_*.parquet"))), desc="predict_all", leave=False):
        pr = pl.read_parquet(path)
        X = np.load(path.replace("predict", "embeddings").replace(".parquet", ".npy")).astype(np.float32)
        if method != "emb":
            X = np.concatenate([X, pr.select(PROB_COLS).to_numpy().astype(np.float32)], axis=1)
        probs = []
        with torch.no_grad():
            for s in range(0, len(X), 65536):
                x = torch.tensor(X[s:s + 65536]).cuda()
                probs.append(torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0).cpu().numpy())
        parts.append(pl.DataFrame({"rep_odaiId": pr["rep_odaiId"], "prob": np.concatenate(probs)}))
    df = pl.concat(parts)
    df.write_parquet(out)
    return df


def derive_format(blank: np.ndarray, karuta: np.ndarray, odai: np.ndarray) -> np.ndarray:
    """属性から形式を逆算する．

    Args:
        blank (np.ndarray): 形状 (N,) の空欄あり．
        karuta (np.ndarray): 形状 (N,) のカルタ記号あり．
        odai (np.ndarray): 形状 (N,) のお題文あり．

    Returns:
        np.ndarray: 形状 (N,) の形式（karuta / anaume / odaibun / normal）．
    """
    return np.where(karuta, "karuta", np.where(blank, "anaume", np.where(odai, "odaibun", "normal")))


def plot(rows: pl.DataFrame, out_dir: str) -> None:
    """構成ごとの F1 と適合率・再現率の図を作る．

    Args:
        rows (pl.DataFrame): 構成ごとの指標（metrics.csv）．
        out_dir (str): 保存先．

    Returns:
        None
    """
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    names = rows["label"].to_list()
    y = np.arange(len(names))[::-1]
    colors = ["#7a7a7a", "#2a78d6", "#eb6834", "#1baf7a"]
    for ax, s, title in zip(axes, ["dev", "test", "real"], ["調整用", "最終評価用", "実データ（推定）"]):
        for k, (m, mk) in enumerate([("precision", "o"), ("recall", "s"), ("f1", "D")]):
            ax.scatter(rows[f"{s}_{m}"], y + (0.2 - 0.2 * k), marker=mk, color=colors[k + 1], s=40, label=["適合率", "再現率", "F1"][k],
                       edgecolor="white", lw=0.8, zorder=3)
        if s == "real":
            lo, hi = rows["real_f1_lo"].to_numpy(), rows["real_f1_hi"].to_numpy()
            f = rows["real_f1"].to_numpy()
            ax.errorbar(f, y - 0.2, xerr=np.abs(np.stack([f - lo, hi - f])), fmt="none", ecolor=colors[3], elinewidth=1)
        ax.set_title(title)
        ax.set_xlim(0.3, 1.02)
        ax.grid(axis="x", color="#dddddd", lw=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out_dir, "fig_pipeline.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    """評価して保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen3-VL-32B-Instruct-FP8__802816__P3")
    args = parser.parse_args()
    out_dir = os.path.join(OUTPUT_DIR, "pipeline")
    os.makedirs(out_dir, exist_ok=True)
    truth = load_truth().sort("fileId")
    key = truth.select("fileId")
    t = truth["odaibun_attr"].to_numpy()
    w = truth["w"].to_numpy()
    split = truth["split"].to_numpy()
    dev = split == "dev"
    a = key.join(pl.scan_parquet(ANNOTATION).select(pl.col("odaiId").alias("fileId"), "format", "has_blank", "has_karuta_mark",
                                                       "p_format_odaibun").collect(), on="fileId", how="left")
    q = key.join(pl.read_csv(os.path.join(OUTPUT_DIR, "qwen", QWEN.split("/")[0], QWEN.split("/")[1] + ".csv")), on="fileId",
                 how="left")["answer"].fill_null(False).to_numpy()
    preds, config = {"現在のパッケージ": (a["format"] == "odaibun").to_numpy(), "Qwen（全画像）": q}, {}
    # 生徒の条件は dev の F1（閾値を選んだ後）で選ぶ
    runs = student_runs(args.teacher)
    scores = {}
    for name, df in runs.items():
        p = key.join(df, on="fileId", how="left")["prob"].to_numpy()
        th = best_threshold(t[dev], p[dev])
        scores[name] = (prf(t[dev], p[dev] >= th, np.ones(dev.sum()))["f1"], th, p)
    best = max(scores, key=lambda k: scores[k][0])
    f1_dev, th, p = scores[best]
    lo, hi = best_band(t[dev], p[dev], q[dev])
    preds["生徒（学習し直し）"] = p >= th
    preds["生徒＋Qwen"] = cascade(p, q, lo, hi)
    cur_od = (a["format"] == "odaibun").to_numpy()
    cur_bk = a["format"].is_in(["anaume", "karuta"]).to_numpy()
    preds["現在＋生徒（穴埋め・カルタ）"] = cur_od | (cur_bk & (p >= th))
    preds["現在＋生徒＋Qwen（穴埋め・カルタ）"] = cur_od | (cur_bk & (p >= 0.2) & q)
    config.update({"student_runs_dev_f1": {k: v[0] for k, v in scores.items()}, "student_best": best, "threshold": th, "lo": lo,
                   "hi": hi, "qwen": QWEN})
    # 全データで Qwen に問う割合と，お題文ありの割合
    method = best.split("/")[0]
    all_p = predict_all(os.path.join(OUTPUT_DIR, "student", args.teacher, best), method,
                        os.path.join(out_dir, f"student_all__{best.replace('/', '__')}.parquet"))["prob"].to_numpy()
    rep = pl.scan_parquet(ANNOTATION).filter(pl.col("odaiId") == pl.col("rep_odaiId")).select("rep_odaiId", "format").collect()
    all_df = pl.read_parquet(os.path.join(out_dir, f"student_all__{best.replace('/', '__')}.parquet")).join(rep, on="rep_odaiId")
    bk = all_df["format"].is_in(["anaume", "karuta"]).to_numpy()
    ap = all_df["prob"].to_numpy()
    od = (all_df["format"] == "odaibun").to_numpy()
    config["full_data_hybrid"] = {"n_rep": int(len(ap)), "odai_rate": float((od | (bk & (ap >= th))).mean()),
                                  "added_on_blank_karuta": float((bk & (ap >= th)).mean()),
                                  "qwen_ask_rate_hybrid_qwen": float((bk & (ap >= 0.2)).mean()),
                                  "current_odaibun_rate": float(od.mean())}
    config["full_data"] = {"n_rep": int(len(all_p)), "ask_rate": float(((all_p >= lo) & (all_p < hi)).mean()),
                           "student_positive_rate": float((all_p >= th).mean()),
                           "cascade_sure_positive_rate": float((all_p >= hi).mean())}
    rows = []
    for label, pred in preds.items():
        r = {"label": label}
        for s in ["dev", "test", "real"]:
            m = split == s
            for k, v in prf(t[m], pred[m], w[m]).items():
                r[f"{s}_{k}"] = v
        r["real_f1_lo"], r["real_f1_hi"] = boot_ci(truth, t, pred)
        r["dev_ask_rate"] = float(((p >= lo) & (p < hi))[dev].mean()) if label == "生徒＋Qwen" else (1.0 if label.startswith("Qwen") else 0.0)
        rows.append(r)
    rows = pl.DataFrame(rows)
    rows.write_csv(os.path.join(out_dir, "metrics.csv"))
    # 属性から逆算した形式の精度（定義書 v4 の形式との一致）
    fmt_rows = []
    blank, karuta = a["has_blank"].to_numpy(), a["has_karuta_mark"].to_numpy()
    for label, pred in [("現在のパッケージ", None), ("逆算（現在＋生徒）", preds["現在＋生徒（穴埋め・カルタ）"])]:
        f = a["format"].to_numpy() if pred is None else derive_format(blank, karuta, pred)
        for s in ["test", "real"]:
            m = split == s
            r = {"label": label, "set": s, "accuracy": float(np.average(f[m] == truth["format"].to_numpy()[m], weights=w[m]))}
            for c in ["karuta", "anaume", "odaibun", "normal"]:
                x = prf(truth["format"].to_numpy()[m] == c, f[m] == c, w[m])
                r[f"{c}_precision"], r[f"{c}_recall"] = x["precision"], x["recall"]
            fmt_rows.append(r)
    pl.DataFrame(fmt_rows).write_csv(os.path.join(out_dir, "format.csv"))
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=1, ensure_ascii=False)
    plot(rows, out_dir)
    with pl.Config(tbl_rows=50, tbl_cols=30, tbl_width_chars=300, float_precision=3, fmt_str_lengths=40):
        print(json.dumps(config, indent=1, ensure_ascii=False))
        print(rows)
        print(pl.DataFrame(fmt_rows))


if __name__ == "__main__":
    main()
