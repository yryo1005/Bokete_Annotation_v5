"""目視のラベルの量と，小さな VLM の分類器の精度の関係を調べる（教師のラベルで学習済みの重みから，目視のラベルで微調整する）．

- curve：学習用の目視のラベル931件のうち 25%・50%・100% で微調整し，検証データの調整用・最終評価用で評価する．
- plus：学習用の931件に検証データの一方の分割（調整用または最終評価用，約310件）の合意ラベルを加えて微調整し，
  もう一方の分割を予測する（2分割で入れ替え，両方の予測を合わせて eval.csv とする．エポック数は4に固定）．

出力：outputs/ex016_boke_small_vlm/ft/{モデル}/{stage1 の条件}__curve{割合}/0/，__plus/0/（eval.csv，log.json）

実行例:
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/stage2_more.py --model q4 --stage1 teacher_lr0.0001_ep2_bs16_r16
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
import train as T  # noqa: E402
from data import OUTPUT_DIR, VOICES, load_eval, load_human  # noqa: E402
from machine_learning_utils import ResultLogger  # noqa: E402

COLS = ["odaiId", "text", "format", "voice", "outside"]


def finetune(key: str, stage1_dir: str, train_df: pl.DataFrame, pred_df: pl.DataFrame, seed: int = 0) -> tuple:
    """学習済みの重みから train_df で微調整し，各エポックの後に pred_df を予測する．

    Args:
        key (str): モデルのキー．
        stage1_dir (str): 教師のラベルで学習した結果のディレクトリ．
        train_df (pl.DataFrame): 学習データ（列 odaiId, text, format, voice, outside）．
        pred_df (pl.DataFrame): 予測するデータ．
        seed (int): 乱数のシード．

    Returns:
        tuple[list, list]: エポックごとの (発話の型の確率, 画像外の知識の確率) と学習の記録．
    """
    model = T.load_model(key, os.path.join(stage1_dir, "best_model.pt"), seed=seed).cuda()
    loss_func = T.LossFunc(*T.class_weights(train_df))
    tr_loader = T.make_loader(train_df, key, T.STAGE2["batch_size"], True, seed)
    pr_loader = T.make_loader(pred_df, key, 32, False, seed)
    opt, sched = T.make_optimizer(model, T.STAGE2["lr"], T.STAGE2["epochs"] * len(tr_loader))
    preds, logs = [], []
    for _ in range(T.STAGE2["epochs"]):
        logs.append(T.epoch(model, tr_loader, loss_func, opt, sched))
        pv, po, _ = T.predict(model, pr_loader)
        preds.append((pv, po))
    del model, opt
    torch.cuda.empty_cache()
    return preds, logs


def scores(ev: pl.DataFrame, pv: np.ndarray, po: np.ndarray, mask: np.ndarray) -> tuple:
    """6区分のマクロ F1 と画像外の知識の F1 を求める．

    Args:
        ev (pl.DataFrame): 検証データ．
        pv (np.ndarray): 発話の型の確率．
        po (np.ndarray): 画像外の知識の確率．
        mask (np.ndarray): 評価する項目．

    Returns:
        tuple[float, float]: 6区分のマクロ F1，画像外の知識の F1．
    """
    y = ev["voice"].to_numpy()[mask]
    return (f1_score(y, np.array(VOICES)[pv[mask].argmax(1)], labels=VOICES, average="macro", zero_division=0),
            f1_score(ev["outside"].to_numpy()[mask], po[mask] > 0.5, zero_division=0))


def main() -> None:
    """学習曲線と，検証データの一方の分割を加えた2分割の微調整を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage1", required=True)
    args = parser.parse_args()
    ev, hu = load_eval(), load_human()
    base = os.path.join(OUTPUT_DIR, "ft", args.model, args.stage1)
    s1 = os.path.join(base, "0")
    dev = (ev["split"] == "dev").to_numpy()
    for frac in [0.25, 0.5]:
        d = os.path.join(f"{base}__curve{frac}", "0")
        if os.path.exists(os.path.join(d, "eval.csv")):
            continue
        os.makedirs(d, exist_ok=True)
        sub = hu.sample(fraction=frac, seed=0)
        preds, logs = finetune(args.model, s1, sub.select(COLS), ev)
        logger = ResultLogger()
        logger.set_names("train_loss", "dev_voice6_f1", "dev_outside_f1", "test_voice6_f1", "test_outside_f1")
        best, best_ep = -1.0, 0
        for ep, ((pv, po), lg) in enumerate(zip(preds, logs)):
            dv, do = scores(ev, pv, po, dev)
            tv, to = scores(ev, pv, po, ~dev)
            logger(lg["loss"], dv, do, tv, to)
            if dv + do > best:
                best, best_ep = dv + do, ep
        logger.save(os.path.join(d, "log.json"))
        T.save_pred(os.path.join(d, "eval.csv"), ev, *preds[best_ep])
        json.dump({"n_human": sub.height, "best_epoch": best_ep + 1}, open(os.path.join(d, "meta.json"), "w"))
        print(args.model, "curve", frac, sub.height, logger.history, flush=True)
    d = os.path.join(f"{base}__plus", "0")
    if not os.path.exists(os.path.join(d, "eval.csv")):
        os.makedirs(d, exist_ok=True)
        pv_all, po_all = np.zeros((ev.height, len(VOICES))), np.zeros(ev.height)
        for tr_split in ["dev", "test"]:
            m = (ev["split"] == tr_split).to_numpy()
            train_df = pl.concat([hu.select(COLS), ev.filter(pl.Series(m)).select(COLS)])
            preds, _ = finetune(args.model, s1, train_df, ev.filter(pl.Series(~m)))
            pv_all[~m], po_all[~m] = preds[-1]
            print(args.model, "plus", tr_split, "->", scores(ev.filter(pl.Series(~m)), *preds[-1], np.ones((~m).sum(), bool)), flush=True)
        T.save_pred(os.path.join(d, "eval.csv"), ev, pv_all, po_all)
        json.dump({"n_human": hu.height, "plus_eval_half": True, "epochs": T.STAGE2["epochs"]}, open(os.path.join(d, "meta.json"), "w"))


if __name__ == "__main__":
    main()
