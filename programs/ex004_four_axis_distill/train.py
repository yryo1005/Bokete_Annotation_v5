"""SigLIP2-B の画像エンコーダを4軸の教師ラベルで微調整し，目視の評価用データで評価する．

結果は outputs/ex004_four_axis_distill/finetune_{モデル}/{教師}_lr{学習率}_ep{エポック数}/{seed}/ に保存する．
- log.json：エポックごとの学習用・検証用（教師ラベル）の誤差，形式の正解率，4軸のマクロ F1 の平均（0エポック目は学習前）
- best_model.pth：検証用の4軸のマクロ F1 の平均が最良のエポックの重み
- last_model.pth，last_optimizer.pth：最終エポックの重みと最適化手法の内部状態
- predictions.csv，metrics.json，meta.json：best_model.pth による評価用データの予測と指標，処理速度

実行例:
    .venv_annotate/bin/python programs/ex004_four_axis_distill/train.py
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from common import OUTPUT_DIR, evaluate, is_done, save_run, throughput  # noqa: E402
from data import FourAxisDataset, load_dataloader, load_eval_labels, load_pool  # noqa: E402
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from model import MODEL_CLASSES, load_model  # noqa: E402
from students import HEADS, labels_frame, to_frame  # noqa: E402

# 探索空間（order_006）：モデルと学習率の組，seed は3つ．学習率は ex002 で SigLIP2-B に用いた値とする
SEARCH_SPACE = {"model_lr": [("siglip2_base", 3e-5)]}
SEEDS = [0, 1, 2]
EPOCHS = 4
BATCH_SIZE = 128
TEACHER = "Qwen3-VL-8B-Instruct_v5_rules"


def loss_func(outputs: dict, teacher_signals: dict) -> torch.Tensor:
    """形式の交差エントロピーと，属性・媒体・事前知識の2値交差エントロピーの和を，1データあたりの平均で求める．

    Args:
        outputs (dict[str, torch.Tensor]): 出力層ごとのロジット．
        teacher_signals (dict[str, torch.Tensor]): キー format（形状 (B,)）と attr・media・know（形状 (B, K)）．

    Returns:
        torch.Tensor: スカラーの誤差．
    """
    loss = torch.nn.functional.cross_entropy(outputs["format"], teacher_signals["format"])
    for h in HEADS:
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(outputs[h], teacher_signals[h])
    return loss


def metrics_func(outputs: dict, teacher_signals: dict) -> dict:
    """形式の正解率と，属性・媒体・事前知識の項目ごとの正解率の平均を，1データあたりの平均で求める．

    Args:
        outputs (dict[str, torch.Tensor]): 出力層ごとのロジット．
        teacher_signals (dict[str, torch.Tensor]): 教師ラベル．

    Returns:
        dict[str, float]: format_accuracy，attr_accuracy，media_accuracy，know_accuracy．
    """
    out = {"format_accuracy": (outputs["format"].argmax(1) == teacher_signals["format"]).float().mean().item()}
    for h in HEADS:
        out[f"{h}_accuracy"] = ((outputs[h] > 0).float() == teacher_signals[h]).float().mean().item()
    return out


def iteration(model: torch.nn.Module, inputs: torch.Tensor, teacher_signals: dict,
              optimizer: torch.optim.Optimizer = None) -> dict:
    """1つのミニバッチを学習または検証する．

    Args:
        model (torch.nn.Module): モデル．
        inputs (torch.Tensor): 形状 (B, 3, 224, 224) の画像．
        teacher_signals (dict[str, torch.Tensor]): 教師ラベル．
        optimizer (torch.optim.Optimizer): 最適化手法．None の場合は勾配を計算しない．

    Returns:
        dict[str, float]: loss と metrics_func の指標（ミニバッチ内の平均）．
    """
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(optimizer is not None):
        outputs = {k: v.float() for k, v in model(inputs).items()}
        loss = loss_func(outputs, teacher_signals)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"loss": loss.item(), **metrics_func(outputs, teacher_signals)}


def epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer = None) -> dict:
    """データローダーの全データを1回学習または検証する．

    Args:
        model (torch.nn.Module): モデル．
        dataloader (DataLoader): データローダー．
        optimizer (torch.optim.Optimizer): 最適化手法．None の場合は検証のみ．

    Returns:
        dict[str, float]: 1データあたりの誤差と指標の平均．
    """
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    totals, n = {}, 0
    bar = tqdm(dataloader, leave=False, desc="train" if optimizer else "val")
    for inputs, teacher_signals in bar:
        inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        teacher_signals = {k: v.to(device, non_blocking=True) for k, v in teacher_signals.items()}
        m = iteration(model, inputs, teacher_signals, optimizer)
        b = len(inputs)
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + v * b
        n += b
        bar.set_postfix({k: f"{v / n:.3f}" for k, v in totals.items()})
    return {k: v / n for k, v in totals.items()}


def train(target_dir: str, ModelClass, load_dataloader, epochs: int, batch_size: int, seed: int = 0,
          hp: dict = None) -> None:
    """モデルを学習し，ログと重みを保存する．

    Args:
        target_dir (str): 結果の保存先のディレクトリ．
        ModelClass (type[torch.nn.Module]): モデルのクラス．
        load_dataloader (callable): データローダーを作る関数．
        epochs (int): エポック数．
        batch_size (int): バッチサイズ．
        seed (int): 乱数のシード．
        hp (dict): その他のハイパーパラメータ（lr，weight_decay，teacher）．

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    train_loader, val_loader = load_dataloader(hp["teacher"], ModelClass.transforms(), batch_size, seed=seed)
    model = load_model(ModelClass, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    names = ["loss", "format_accuracy", "attr_accuracy", "media_accuracy", "know_accuracy"]
    logger = ResultLogger()
    logger.set_names(*[f"train_{k}" for k in names], *[f"val_{k}" for k in names], "val_overall_macro_f1")
    best = -1.0
    for ep in range(epochs + 1):
        m_tr = epoch(model, train_loader, optimizer if ep > 0 else None)
        m_va = epoch(model, val_loader)
        f1 = validation_macro_f1(model, val_loader)
        logger(*[m_tr[k] for k in names], *[m_va[k] for k in names], f1)
        print(f"    epoch {ep}: train loss={m_tr['loss']:.3f} fmt={m_tr['format_accuracy']:.3f} | "
              f"val loss={m_va['loss']:.3f} fmt={m_va['format_accuracy']:.3f} overallF1={f1:.3f}")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
    logger.save(os.path.join(target_dir, "log.json"))
    torch.save(model.state_dict(), os.path.join(target_dir, "last_model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(target_dir, "last_optimizer.pth"))


@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader: DataLoader) -> tuple:
    """データローダーの全データの確率と教師ラベルを集める．

    Args:
        model (torch.nn.Module): モデル．
        loader (DataLoader): データローダー．

    Returns:
        tuple[dict, dict]: 出力層ごとの確率と，教師ラベルの配列（data.label_arrays と同じ形式）．
    """
    model.eval()
    probs, ys = {k: [] for k in ["format", *HEADS]}, {k: [] for k in ["format", *HEADS]}
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x.cuda(non_blocking=True).to(memory_format=torch.channels_last))
        probs["format"].append(torch.softmax(out["format"].float(), 1).cpu().numpy())
        for h in HEADS:
            probs[h].append(torch.sigmoid(out[h].float()).cpu().numpy())
        for k in ys:
            ys[k].append(y[k].numpy())
    return {k: np.concatenate(v) for k, v in probs.items()}, {k: np.concatenate(v) for k, v in ys.items()}


def validation_macro_f1(model: torch.nn.Module, loader: DataLoader) -> float:
    """検証用データの4軸のマクロ F1 の平均を求める．

    Args:
        model (torch.nn.Module): モデル．
        loader (DataLoader): 検証用のデータローダー．

    Returns:
        float: 4軸のマクロ F1 の平均．
    """
    probs, y = collect_probs(model, loader)
    return evaluate(labels_frame(y), to_frame(probs))["overall_macro_f1"]


def evaluate_run(target_dir: str, ModelClass, seed: int, hp: dict, measure_speed: bool) -> None:
    """best_model.pth で評価用データを予測し，指標と処理速度を保存する．

    Args:
        target_dir (str): 結果のディレクトリ．
        ModelClass (type[torch.nn.Module]): モデルのクラス．
        seed (int): 乱数のシード．
        hp (dict): ハイパーパラメータ．
        measure_speed (bool): 教師ラベル用の画像10,000枚で処理速度を測るか．

    Returns:
        None
    """
    labels = load_eval_labels()
    model = load_model(ModelClass, os.path.join(target_dir, "best_model.pth"), seed).cuda()
    model = model.to(memory_format=torch.channels_last)
    _, val_tf = ModelClass.transforms()
    dummy = {"format": np.zeros(len(labels), int), **{h: np.zeros((len(labels), 1), np.float32) for h in HEADS}}
    loader = DataLoader(FourAxisDataset(labels["fileId"].to_list(), dummy, val_tf), batch_size=256, num_workers=16)
    probs, _ = collect_probs(model, loader)
    meta = {"method": "finetune", "model": ModelClass.__name__, "seed": seed, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, **hp}
    if measure_speed:
        ids = load_pool()["fileId"].to_list()[:10000]
        d = {"format": np.zeros(len(ids), int), **{h: np.zeros((len(ids), 1), np.float32) for h in HEADS}}
        sp_loader = DataLoader(FourAxisDataset(ids, d, val_tf), batch_size=256, num_workers=16, pin_memory=True)
        start = time.perf_counter()
        collect_probs(model, sp_loader)
        torch.cuda.synchronize()
        meta["throughput"] = throughput(len(ids), time.perf_counter() - start)
    log = json.load(open(os.path.join(target_dir, "log.json")))
    best_ep = int(np.argmax(log["val_overall_macro_f1"]))
    extra = {"best_epoch": best_ep, "train": {k[6:]: v[best_ep] for k, v in log.items() if k.startswith("train_")},
             "val": {k[4:]: v[best_ep] for k, v in log.items() if k.startswith("val_")}}
    save_run(target_dir, labels, to_frame(probs), meta, extra)


def main() -> None:
    """探索空間と seed の全組み合わせで微調整と評価を行う．終了済みの条件は省略する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=TEACHER)
    args = parser.parse_args()
    for (model_name, lr), seed in itertools.product(SEARCH_SPACE["model_lr"], SEEDS):
        target_dir = os.path.join(OUTPUT_DIR, f"finetune_{model_name}", f"{args.teacher}_lr{lr}_ep{EPOCHS}", str(seed))
        if is_done(target_dir):
            continue
        print(f"[finetune] {model_name} lr={lr} seed={seed}")
        hp = {"lr": lr, "weight_decay": 0.05, "teacher": args.teacher}
        set_seed(seed)
        if not os.path.exists(os.path.join(target_dir, "log.json")):
            train(target_dir, MODEL_CLASSES[model_name], load_dataloader, EPOCHS, BATCH_SIZE, seed, hp)
        evaluate_run(target_dir, MODEL_CLASSES[model_name], seed, hp, measure_speed=(seed == 0))


if __name__ == "__main__":
    main()
