"""SigLIP2-B の画像エンコーダを定義書 v4 の教師ラベル（T_v10oJXk）で微調整し，全ての評価用の画像の確率を保存する．

比較する条件（入力解像度）：
- b224：google/siglip2-base-patch16-224（224×224 画素，ex005 と同じ）
- b384：google/siglip2-base-patch16-384（384×384 画素．文字の細部を捉えるため）
学習データは基本の4万枚と少ない区分の追加分，損失は重みなし（order_012 で重み付き損失を廃止）とし，seed は3つとする．

結果は outputs/ex009_final/finetune/{モデル}/{教師}__base+rare/{seed}/ に保存する．
- log.json：エポックごとの学習用・検証用（基本の4万枚の検証用分割）の誤差と正解率，検証用の3軸のマクロ F1 の平均
- best_model.pth，last_model.pth，last_optimizer.pth
- probs_{eval,rare,check}.npz：評価用データ640枚，少ない区分の評価用データ800枚，実データの目視ラベル380枚の確率
- predictions.csv，metrics.json，meta.json（seed 0 は処理速度も記録する）

実行例:
    .venv_annotate/bin/python programs/ex009_final/train.py --models b224 b384
    .venv_annotate/bin/python programs/ex009_final/train.py --models b224 --teacher T_v11 --seeds 0
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from common import OUTPUT_DIR, evaluate, is_done, save_run, throughput  # noqa: E402
from data import (ATTRIBUTES, EX8_DIR, FORMATS, MEDIA, TEACHER, ThreeAxisDataset, _load, check_ids,  # noqa: E402
                  label_arrays, load_eval_labels, load_pool, load_rare_labels, load_teacher_labels)
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402

ST = _load("ex008_students_for_final", os.path.join(EX8_DIR, "students.py"))
HEADS, three_head_loss, to_frame, labels_frame = ST.HEADS, ST.three_head_loss, ST.to_frame, ST.labels_frame

MODELS = {"b224": ("google/siglip2-base-patch16-224", 224), "b384": ("google/siglip2-base-patch16-384", 384)}
SEEDS = [0, 1, 2]
EPOCHS = 5
BATCH_SIZE = 128
LR = 3e-5
WEIGHT_DECAY = 0.05


class SigLIP2ThreeHead(torch.nn.Module):
    """SigLIP2 の画像エンコーダのプーリング出力に，形式・属性・媒体の3つの線形の出力層を付けたモデル．"""

    def __init__(self, hf_name: str, size: int):
        """
        Args:
            hf_name (str): Hugging Face のモデル名．
            size (int): 入力画像の一辺の画素数．
        """
        super().__init__()
        from transformers import AutoModel

        self.size = size
        self.backbone = AutoModel.from_pretrained(hf_name).vision_model
        d = self.backbone.config.hidden_size
        self.heads = torch.nn.ModuleDict({"format": torch.nn.Linear(d, len(FORMATS)),
                                          "attr": torch.nn.Linear(d, len(ATTRIBUTES)),
                                          "media": torch.nn.Linear(d, len(MEDIA))})

    def forward(self, x: torch.Tensor) -> dict:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, 3, size, size) の画像．

        Returns:
            dict[str, torch.Tensor]: 出力層ごとのロジット（format は (B, 4)，attr は (B, 3)，media は (B, 7)）．
        """
        h = self.backbone(pixel_values=x).pooler_output
        return {k: m(h) for k, m in self.heads.items()}

    def transforms(self) -> tuple:
        """学習用と予測用の画像の変換を返す（左右反転は文字を壊すため用いず，色の揺らぎのみとする）．

        Returns:
            tuple: (学習用の変換, 予測用の変換)．
        """
        return make_transforms(self.size)


def make_transforms(size: int) -> tuple:
    """学習用と予測用の画像の変換を返す．

    Args:
        size (int): 入力画像の一辺の画素数．

    Returns:
        tuple: (学習用の変換, 予測用の変換)．
    """
    from torchvision import transforms as T

    norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return (T.Compose([T.Resize((size, size)), T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), norm]),
            T.Compose([T.Resize((size, size)), T.ToTensor(), norm]))


def load_model(model_key: str, weight_path: str = None, seed: int = 0) -> torch.nn.Module:
    """モデルをインスタンス化し，必要に応じて重みを読み込む．

    Args:
        model_key (str): MODELS のキー．
        weight_path (str): 重みのファイルのパス．
        seed (int): 出力層の初期値を固定する乱数のシード．

    Returns:
        torch.nn.Module: モデル．
    """
    set_seed(seed)
    model = SigLIP2ThreeHead(*MODELS[model_key])
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return model


def load_dataloader(transforms: tuple, batch_size: int, seed: int = 0, teacher: str = TEACHER) -> tuple:
    """学習用と検証用のデータローダーを作る（基本の4万枚と少ない区分の追加分）．

    Args:
        teacher (str): 教師の名前．
        transforms (tuple): (学習用の変換, 予測用の変換)．
        batch_size (int): バッチサイズ．
        seed (int): 乱数のシード．

    Returns:
        tuple[DataLoader, DataLoader]: 学習用と検証用のデータローダー．
    """
    set_seed(seed)
    lab = load_teacher_labels(f"{teacher}_pool", f"{teacher}_rare")
    tr = lab.filter(pl.col("split") == "train")
    va = lab.filter((pl.col("split") == "val") & (pl.col("source") == "base"))
    g = torch.Generator()
    g.manual_seed(seed)
    kw = {"batch_size": batch_size, "num_workers": 16, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2}
    return (DataLoader(ThreeAxisDataset(tr["fileId"].to_list(), label_arrays(tr), transforms[0]), shuffle=True, generator=g,
                       drop_last=True, **kw),
            DataLoader(ThreeAxisDataset(va["fileId"].to_list(), label_arrays(va), transforms[1]), shuffle=False, **kw))


def predict_loader(ids: list, transform, batch_size: int = 256, num_workers: int = 16) -> DataLoader:
    """お題IDのリストの予測用のデータローダーを作る．

    Args:
        ids (list[int]): お題IDのリスト．
        transform (callable): 予測用の変換．
        batch_size (int): バッチサイズ．
        num_workers (int): 読み込みのプロセス数．

    Returns:
        DataLoader: データローダー．
    """
    dummy = {"format": np.zeros(len(ids), int), "attr": np.zeros((len(ids), len(ATTRIBUTES)), np.float32),
             "media": np.zeros((len(ids), len(MEDIA)), np.float32)}
    return DataLoader(ThreeAxisDataset(ids, dummy, transform), batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True)


def loss_func(outputs: dict, teacher_signals: dict) -> torch.Tensor:
    """形式の交差エントロピーと属性・媒体の2値交差エントロピーの和を，1データあたりの平均で求める．

    Args:
        outputs (dict[str, torch.Tensor]): 出力層ごとのロジット．
        teacher_signals (dict[str, torch.Tensor]): 教師ラベル．

    Returns:
        torch.Tensor: スカラーの誤差．
    """
    return three_head_loss(outputs, teacher_signals, None)


def metrics_func(outputs: dict, teacher_signals: dict) -> dict:
    """形式の正解率と，属性・媒体の項目ごとの正解率の平均を求める．

    Args:
        outputs (dict[str, torch.Tensor]): 出力層ごとのロジット．
        teacher_signals (dict[str, torch.Tensor]): 教師ラベル．

    Returns:
        dict[str, float]: format_accuracy，attr_accuracy，media_accuracy．
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
        inputs (torch.Tensor): 形状 (B, 3, H, W) の画像．
        teacher_signals (dict[str, torch.Tensor]): 教師ラベル．
        optimizer (torch.optim.Optimizer): 最適化手法．None の場合は勾配を計算しない．

    Returns:
        dict[str, float]: loss と metrics_func の指標．
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


@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader: DataLoader) -> tuple:
    """データローダーの全データの確率とラベルを集める．

    Args:
        model (torch.nn.Module): モデル．
        loader (DataLoader): データローダー．

    Returns:
        tuple[dict, dict]: 出力層ごとの確率（format は (N, 4)，attr は (N, 3)，media は (N, 7)）と，ラベルの配列．
    """
    model.eval()
    probs, ys = {k: [] for k in ["format", *HEADS]}, {k: [] for k in ["format", *HEADS]}
    for x, y in tqdm(loader, leave=False, desc="predict"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x.cuda(non_blocking=True).to(memory_format=torch.channels_last))
        probs["format"].append(torch.softmax(out["format"].float(), 1).cpu().numpy())
        for h in HEADS:
            probs[h].append(torch.sigmoid(out[h].float()).cpu().numpy())
        for k in ys:
            ys[k].append(y[k].numpy())
    return {k: np.concatenate(v) for k, v in probs.items()}, {k: np.concatenate(v) for k, v in ys.items()}


def train(target_dir: str, model_key: str, seed: int, teacher: str = TEACHER) -> None:
    """モデルを学習し，ログと重みを保存する．

    Args:
        teacher (str): 教師の名前．
        target_dir (str): 結果の保存先のディレクトリ．
        model_key (str): MODELS のキー．
        seed (int): 乱数のシード．

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    model = load_model(model_key, seed=seed).cuda().to(memory_format=torch.channels_last)
    train_loader, val_loader = load_dataloader(model.transforms(), BATCH_SIZE, seed=seed, teacher=teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    names = ["loss", "format_accuracy", "attr_accuracy", "media_accuracy"]
    logger = ResultLogger()
    logger.set_names(*[f"train_{k}" for k in names], *[f"val_{k}" for k in names], "val_overall_macro_f1")
    best = -1.0
    for ep in range(EPOCHS + 1):
        m_tr = epoch(model, train_loader, optimizer if ep > 0 else None)
        m_va = epoch(model, val_loader, None)
        probs, y = collect_probs(model, val_loader)
        f1 = evaluate(labels_frame(y), to_frame(probs))["overall_macro_f1"]
        logger(*[m_tr[k] for k in names], *[m_va[k] for k in names], f1)
        print(f"    epoch {ep}: train loss={m_tr['loss']:.3f} | val loss={m_va['loss']:.3f} overallF1={f1:.3f}", flush=True)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
    logger.save(os.path.join(target_dir, "log.json"))
    torch.save(model.state_dict(), os.path.join(target_dir, "last_model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(target_dir, "last_optimizer.pth"))


def evaluate_run(target_dir: str, model_key: str, seed: int, teacher: str = TEACHER) -> None:
    """best_model.pth で全ての評価用の画像を予測し，確率，指標，処理速度を保存する．

    Args:
        teacher (str): 教師の名前．
        target_dir (str): 結果のディレクトリ．
        model_key (str): MODELS のキー．
        seed (int): 乱数のシード．

    Returns:
        None
    """
    labels = load_eval_labels()
    model = load_model(model_key, os.path.join(target_dir, "best_model.pth"), seed).cuda()
    model = model.to(memory_format=torch.channels_last)
    _, tf = model.transforms()
    sets = {"eval": labels["fileId"].to_list(), "rare": load_rare_labels()["fileId"].to_list(), "check": check_ids()}
    probs = {}
    for name, ids in sets.items():
        probs[name], _ = collect_probs(model, predict_loader(ids, tf))
        np.savez(os.path.join(target_dir, f"probs_{name}.npz"), **probs[name])
    meta = {"method": "finetune", "model": MODELS[model_key][0], "size": MODELS[model_key][1], "teacher": teacher,
            "use_rare": True, "weighted": False, "seed": seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY}
    if seed == 0:
        ids = load_pool()["fileId"].to_list()[:10000]
        start = time.perf_counter()
        collect_probs(model, predict_loader(ids, tf))
        torch.cuda.synchronize()
        meta["throughput"] = throughput(len(ids), time.perf_counter() - start)
    log = json.load(open(os.path.join(target_dir, "log.json")))
    best_ep = int(np.argmax(log["val_overall_macro_f1"]))
    extra = {"best_epoch": best_ep, "train": {k[6:]: v[best_ep] for k, v in log.items() if k.startswith("train_")},
             "val": {k[4:]: v[best_ep] for k, v in log.items() if k.startswith("val_")}}
    save_run(target_dir, labels, to_frame(probs["eval"]), meta, extra)


def main() -> None:
    """モデルと seed の全組み合わせで微調整と評価を行う．終了済みの条件は省略する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--teacher", default=TEACHER)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    for model_key, seed in itertools.product(args.models, args.seeds):
        target_dir = os.path.join(OUTPUT_DIR, "finetune", model_key, f"{args.teacher}__base+rare", str(seed))
        if is_done(target_dir):
            continue
        print(f"[finetune] {model_key} seed={seed}", flush=True)
        if not os.path.exists(os.path.join(target_dir, "log.json")):
            train(target_dir, model_key, seed, args.teacher)
        evaluate_run(target_dir, model_key, seed, args.teacher)


if __name__ == "__main__":
    main()
