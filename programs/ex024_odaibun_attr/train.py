"""お題文ありの生徒（画像埋め込みに対する MLP）を，Qwen の教師ラベルで学習する．

- 教師ラベル：teacher.py の出力（4.5万枚．文字ありの確率が低い画像はお題文なし）．9割を学習用，1割を検証用とする．
- 入力（比べる手法）：
  - emb：SigLIP2-so400m の画像埋め込み（1,152次元）
  - emb+probs：埋め込みに，現在のパッケージの生徒の形式4・属性3の確率を加えたもの（1,159次元）
- 検証用の誤差が最小のエポックの重みを用い，目視のラベルの1,020枚の確率を保存する．

出力：outputs/ex024_odaibun_attr/student/{教師}/{手法}/lr{学習率}_ep{エポック数}/{seed}/
      （log.json，meta.json，best_model.pt，last_model.pt，last_optimizer.pt，eval_probs.csv）

実行例:
    .venv_annotate/bin/python programs/ex024_odaibun_attr/train.py --teacher Qwen3-VL-32B-Instruct-FP8__200704__P3
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import ANNOTATION, DATASET_DIR, OUTPUT_DIR, load_embeddings, load_truth_v4  # noqa: E402
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from model import OdaibunMLP  # noqa: E402

# 条件：手法，学習率，エポック数，seed（order_041）
METHODS = ["emb", "emb+probs"]
LRS = [1e-3, 3e-4]
EPOCHS = 40
SEEDS = [0, 1, 2]
BATCH = 256
PROB_COLS = [f"p_format_{f}" for f in ["karuta", "anaume", "odaibun", "normal"]] + ["p_has_blank", "p_has_karuta_mark", "p_has_text"]


def load_teacher(teacher: str) -> pl.DataFrame:
    """教師ラベルを読み込む．

    Args:
        teacher (str): 教師の名前（outputs/ex024_odaibun_attr/teacher/ のディレクトリ名）．

    Returns:
        pl.DataFrame: 列 fileId, y（bool）を持つ表．
    """
    d = os.path.join(OUTPUT_DIR, "teacher", teacher)
    df = pl.concat([pl.read_csv(p) for p in glob.glob(os.path.join(d, "*.csv"))])
    return df.select("fileId", pl.col("answer").alias("y")).unique("fileId").sort("fileId")


def features(ids: list, method: str, cache: str) -> np.ndarray:
    """入力の特徴を作る（埋め込みはキャッシュに保存する）．

    Args:
        ids (list[int]): お題IDのリスト．
        method (str): 手法（emb / emb+probs）．
        cache (str): 埋め込みのキャッシュのパス（.npy）．

    Returns:
        np.ndarray: 形状 (len(ids), 1152 または 1159) の float32 の配列．
    """
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        X = load_embeddings(ids)
        np.save(cache, X)
    if method == "emb":
        return X
    a = pl.scan_parquet(ANNOTATION).select(pl.col("odaiId").alias("fileId"), *PROB_COLS).collect()
    P = pl.DataFrame({"fileId": ids}).join(a, on="fileId", how="left").select(PROB_COLS).to_numpy().astype(np.float32)
    return np.concatenate([X, P], axis=1)


def iteration(model, X: torch.Tensor, y: torch.Tensor, opt=None) -> tuple:
    """1エポック分の学習または評価を行う．

    Args:
        model (OdaibunMLP): モデル．
        X (torch.Tensor): 形状 (N, D) の入力（GPU）．
        y (torch.Tensor): 形状 (N,) の教師ラベル（float，GPU）．
        opt (torch.optim.Optimizer): 最適化手法（None なら評価）．

    Returns:
        tuple[float, float, np.ndarray]: 平均の誤差，0.5を閾値とした F1，形状 (N,) の確率．
    """
    model.train(opt is not None)
    idx = torch.randperm(len(X), device=X.device) if opt is not None else torch.arange(len(X), device=X.device)
    loss_sum, probs = 0.0, torch.empty(len(X), device=X.device)
    for s in range(0, len(X), BATCH):
        b = idx[s:s + BATCH]
        with torch.set_grad_enabled(opt is not None):
            logit = model(X[b])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[b])
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_sum += loss.item() * len(b)
        probs[b] = torch.sigmoid(logit.detach())
    p = probs.cpu().numpy()
    t = y.cpu().numpy() > 0.5
    pred = p >= 0.5
    f1 = 2 * (t & pred).sum() / max(t.sum() + pred.sum(), 1)
    return loss_sum / len(X), float(f1), p


def train_one(run_dir: str, X_tr, y_tr, X_va, y_va, X_ev, ev_ids: list, lr: float, seed: int) -> None:
    """1つの条件で学習し，結果を保存する．

    Args:
        run_dir (str): 出力先．
        X_tr, y_tr, X_va, y_va (torch.Tensor): 学習用・検証用の入力と教師ラベル（GPU）．
        X_ev (torch.Tensor): 目視のラベルの画像の入力（GPU）．
        ev_ids (list[int]): 目視のラベルの画像のお題ID．
        lr (float): 学習率．
        seed (int): seed．

    Returns:
        None
    """
    set_seed(seed)
    model = OdaibunMLP(X_tr.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    log = ResultLogger()
    log.set_names("train_loss", "train_f1", "val_loss", "val_f1")
    best = float("inf")
    for _ in tqdm(range(EPOCHS), desc=os.path.relpath(run_dir, OUTPUT_DIR), leave=False):
        tl, tf, _ = iteration(model, X_tr, y_tr, opt)
        vl, vf, _ = iteration(model, X_va, y_va)
        log(tl, tf, vl, vf)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pt"))
    torch.save(opt.state_dict(), os.path.join(run_dir, "last_optimizer.pt"))
    log.save(os.path.join(run_dir, "log.json"))
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    _, _, p = iteration(model, X_ev, torch.zeros(len(X_ev), device=X_ev.device))
    pl.DataFrame({"fileId": ev_ids, "prob": p}).write_csv(os.path.join(run_dir, "eval_probs.csv"))
    json.dump({"lr": lr, "epochs": EPOCHS, "batch": BATCH, "seed": seed, "best_epoch": int(np.argmin(log["val_loss"])),
               "n_train": len(X_tr), "n_val": len(X_va), "train_pos_rate": float(y_tr.mean())},
              open(os.path.join(run_dir, "meta.json"), "w"), indent=1)


def main() -> None:
    """全ての条件で学習する（正常終了した条件は飛ばす）．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen3-VL-32B-Instruct-FP8__200704__P3")
    args = parser.parse_args()
    lab = load_teacher(args.teacher)
    ids = lab["fileId"].to_list()
    ev_ids = load_truth_v4()["fileId"].sort().to_list()
    os.makedirs(DATASET_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    va = rng.random(len(ids)) < 0.1
    y = torch.tensor(lab["y"].to_numpy().astype(np.float32)).cuda()
    for method, lr, seed in itertools.product(METHODS, LRS, SEEDS):
        run_dir = os.path.join(OUTPUT_DIR, "student", args.teacher, method, f"lr{lr}_ep{EPOCHS}", str(seed))
        if os.path.exists(os.path.join(run_dir, "eval_probs.csv")):
            continue
        os.makedirs(run_dir, exist_ok=True)
        X = torch.tensor(features(ids, method, os.path.join(DATASET_DIR, "teacher_emb.npy"))).cuda()
        X_ev = torch.tensor(features(ev_ids, method, os.path.join(DATASET_DIR, "eval_emb.npy"))).cuda()
        m = torch.tensor(va).cuda()
        train_one(run_dir, X[~m], y[~m], X[m], y[m], X_ev, ev_ids, lr, seed)


if __name__ == "__main__":
    main()
