"""画像エンコーダを教師 VLM のラベルで微調整し，目視の評価用データで評価する．

結果は outputs/ex002_odai_type_distill/finetune_{モデル}/{教師}_lr{学習率}_ep{エポック数}/{seed}/ に保存する．
- log.json：エポックごとの学習用・検証用（教師ラベル）の誤差と正解率（0エポック目は学習前）
- best_model.pth：検証用の7クラスの正解率が最良のエポックの重み
- last_model.pth，last_optimizer.pth：最終エポックの重みと最適化手法の内部状態（追加の学習の再開用）
- predictions.csv，metrics.json，meta.json：best_model.pth による評価用データの予測と指標，処理速度

実行例:
    .venv_annotate/bin/python programs/ex002_odai_type_distill/train.py
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
from common import OUTPUT_DIR, is_done, save_run, throughput  # noqa: E402
from data import DOMAIN_LABELS, FORMAT_LABELS, OdaiImageDataset, load_dataloader, load_eval_labels, load_pool  # noqa: E402
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from model import MODEL_CLASSES, load_model  # noqa: E402

# 探索空間（order_002）：モデルと学習率の組，seed は3つ
# 計算時間（1条件あたり約20分）の制約から，学習率は CNN に 1e-4，ViT に 3e-5 の一般的な値を1つずつ用いる
SEARCH_SPACE = {"model_lr": [("convnext_tiny", 1e-4), ("siglip2_base", 3e-5)]}
SEEDS = [0, 1, 2]
EPOCHS = 5
BATCH_SIZE = 128
TEACHER = "Qwen3-VL-8B-Instruct"


def loss_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> torch.Tensor:
    """形式と画風の交差エントロピーの和を，1データあたりの平均として求める．

    Args:
        outputs (torch.Tensor): 形状 (B, 8) のロジット（前半4列が形式，後半4列が画風）．
        teacher_signals (torch.Tensor): 形状 (B, 2) のラベル番号 [形式, 画風]．

    Returns:
        torch.Tensor: スカラーの誤差．
    """
    ce = torch.nn.functional.cross_entropy
    return ce(outputs[:, :4], teacher_signals[:, 0]) + ce(outputs[:, 4:], teacher_signals[:, 1])


def metrics_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> dict:
    """形式，画風，7クラスの正解率を1データあたりの平均として求める．

    Args:
        outputs (torch.Tensor): 形状 (B, 8) のロジット．
        teacher_signals (torch.Tensor): 形状 (B, 2) のラベル番号．

    Returns:
        dict[str, float]: format_accuracy，domain_accuracy，flat_accuracy．
    """
    pf, pd_ = outputs[:, :4].argmax(1), outputs[:, 4:].argmax(1)
    tf, td = teacher_signals[:, 0], teacher_signals[:, 1]
    ok_f, ok_d = pf == tf, pd_ == td
    normal = FORMAT_LABELS.index("normal")
    # 7クラスの正解：形式が一致し，かつ形式が normal の場合は画風も一致する
    ok_flat = ok_f & ((tf != normal) | ok_d)
    return {"format_accuracy": ok_f.float().mean().item(), "domain_accuracy": ok_d.float().mean().item(),
            "flat_accuracy": ok_flat.float().mean().item()}


def iteration(model: torch.nn.Module, inputs: torch.Tensor, teacher_signals: torch.Tensor,
              optimizer: torch.optim.Optimizer = None) -> dict:
    """1つのミニバッチを学習または検証する．

    Args:
        model (torch.nn.Module): モデル．
        inputs (torch.Tensor): 形状 (B, 3, 224, 224) の画像．
        teacher_signals (torch.Tensor): 形状 (B, 2) のラベル番号．
        optimizer (torch.optim.Optimizer): 最適化手法．None の場合は勾配を計算しない．

    Returns:
        dict[str, float]: loss と metrics_func の指標（ミニバッチ内の平均）．
    """
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(optimizer is not None):
        outputs = model(inputs)
        loss = loss_func(outputs.float(), teacher_signals)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"loss": loss.item(), **metrics_func(outputs.float(), teacher_signals)}


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
        inputs, teacher_signals = inputs.to(device, non_blocking=True), teacher_signals.to(device, non_blocking=True)
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
    names = ["loss", "format_accuracy", "domain_accuracy", "flat_accuracy"]
    logger = ResultLogger()
    logger.set_names(*[f"train_{k}" for k in names], *[f"val_{k}" for k in names])
    best = -1.0
    for ep in range(epochs + 1):
        m_tr = epoch(model, train_loader, optimizer if ep > 0 else None)
        m_va = epoch(model, val_loader)
        logger(*[m_tr[k] for k in names], *[m_va[k] for k in names])
        print(f"    epoch {ep}: train loss={m_tr['loss']:.3f} acc={m_tr['flat_accuracy']:.3f} | "
              f"val loss={m_va['loss']:.3f} acc={m_va['flat_accuracy']:.3f}")
        if m_va["flat_accuracy"] > best:
            best = m_va["flat_accuracy"]
            torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
    logger.save(os.path.join(target_dir, "log.json"))
    torch.save(model.state_dict(), os.path.join(target_dir, "last_model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(target_dir, "last_optimizer.pth"))


@torch.no_grad()
def predict_ids(model: torch.nn.Module, transform, ids: list, batch_size: int = 256) -> tuple:
    """画像のリストの種別を予測し，読み込みを含む処理速度を測る．

    Args:
        model (torch.nn.Module): 学習済みのモデル．
        transform (callable): 検証用の画像の変換．
        ids (list[int]): お題IDのリスト（長さ N）．
        batch_size (int): バッチサイズ．

    Returns:
        tuple[np.ndarray, float]: 形状 (N, 8) の確率（前半が形式，後半が画風）と経過秒数．
    """
    model.eval()
    ds = OdaiImageDataset(ids, ["normal"] * len(ids), ["photo"] * len(ids), transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=16, pin_memory=True)
    out = []
    start = time.perf_counter()
    for x, _ in tqdm(loader, leave=False, desc="predict"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x.cuda(non_blocking=True).to(memory_format=torch.channels_last)).float()
        out.append(torch.cat([torch.softmax(z[:, :4], 1), torch.softmax(z[:, 4:], 1)], 1).cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(out), time.perf_counter() - start


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
    probs, _ = predict_ids(model, val_tf, labels["fileId"].to_list())
    meta = {"method": "finetune", "model": ModelClass.__name__, "seed": seed, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, **hp}
    if measure_speed:
        ids = load_pool()["fileId"].to_list()[:10000]
        _, sec = predict_ids(model, val_tf, ids)
        meta["throughput"] = throughput(len(ids), sec)
    log = json.load(open(os.path.join(target_dir, "log.json")))
    best_ep = int(np.argmax(log["val_flat_accuracy"]))
    extra = {"best_epoch": best_ep, "train": {k[6:]: v[best_ep] for k, v in log.items() if k.startswith("train_")},
             "val": {k[4:]: v[best_ep] for k, v in log.items() if k.startswith("val_")}}
    save_run(target_dir, labels, [FORMAT_LABELS[i] for i in probs[:, :4].argmax(1)],
             [DOMAIN_LABELS[i] for i in probs[:, 4:].argmax(1)], meta, extra,
             {"confidence": probs[:, :4].max(1) * probs[:, 4:].max(1)})


def remeasure_speed(teacher: str) -> None:
    """終了済みの seed 0 の条件について，他の処理と競合しない状態で処理速度を測り直し meta.json を更新する．

    Args:
        teacher (str): 教師ラベルの名前．

    Returns:
        None
    """
    for model_name, lr in SEARCH_SPACE["model_lr"]:
        target_dir = os.path.join(OUTPUT_DIR, f"finetune_{model_name}", f"{teacher}_lr{lr}_ep{EPOCHS}", "0")
        if not is_done(target_dir):
            continue
        ModelClass = MODEL_CLASSES[model_name]
        model = load_model(ModelClass, os.path.join(target_dir, "best_model.pth"), 0).cuda()
        model = model.to(memory_format=torch.channels_last)
        ids = load_pool()["fileId"].to_list()[:10000]
        _, sec = predict_ids(model, ModelClass.transforms()[1], ids)
        meta_path = os.path.join(target_dir, "meta.json")
        meta = json.load(open(meta_path))
        meta["throughput_during_other_jobs"] = meta.get("throughput")
        meta["throughput"] = throughput(len(ids), sec)
        json.dump(meta, open(meta_path, "w"), indent=4, ensure_ascii=False)
        print(f"{model_name}: {meta['throughput']['images_per_sec']:.1f} 枚/秒")


def main() -> None:
    """探索空間と seed の全組み合わせで微調整と評価を行う．終了済みの条件は省略する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=TEACHER)
    parser.add_argument("--speed_only", action="store_true", help="終了済みの seed 0 の条件の処理速度のみを測り直す")
    args = parser.parse_args()
    if args.speed_only:
        remeasure_speed(args.teacher)
        return
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
