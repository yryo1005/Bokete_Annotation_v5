"""SigLIP2-B の画像エンコーダを3軸の教師ラベルで微調整し，評価用データで評価する．

少ない区分の追加分の有無と，重み付き損失の有無を比較する．重みは students.py の class_weights と同じである．
結果は outputs/ex005_rare_weighted_distill/finetune_siglip2_base/{教師}__{データ}__{損失}/{seed}/ に保存する．
- log.json：エポックごとの学習用・検証用（基本の4万枚の検証用分割）の誤差と形式の正解率，検証用の3軸のマクロ F1 の平均
- best_model.pth，last_model.pth，last_optimizer.pth
- predictions.csv，metrics.json，meta.json

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/train.py --teacher T_v2_J6M7
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
from data import (ATTRIBUTES, FORMATS, MEDIA, ThreeAxisDataset, label_arrays, load_eval_labels, load_pool,  # noqa: E402
                  load_teacher_labels)
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from students import HEADS, labels_frame, three_head_loss, to_frame, weights_for  # noqa: E402

# 探索空間（order_007）：データ × 損失，seed は3つ．学習率は ex002・ex004 と同じ値とする
# order_009：損失に weighted_sel（合成画像とその他の文字を重み付けの対象から外す）を追加（--weight_modes で指定）
SEARCH_SPACE = {"use_rare": [False, True], "weighted": [False, True]}
SEEDS = [0, 1, 2]
EPOCHS = 5
BATCH_SIZE = 128
LR = 3e-5
HF_NAME = "google/siglip2-base-patch16-224"


class SigLIP2BaseThreeHead(torch.nn.Module):
    """SigLIP2-B の画像エンコーダのプーリング出力に，形式・属性・媒体の3つの線形の出力層を付けたモデル．"""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(HF_NAME).vision_model
        d = self.backbone.config.hidden_size
        self.heads = torch.nn.ModuleDict({"format": torch.nn.Linear(d, len(FORMATS)),
                                          "attr": torch.nn.Linear(d, len(ATTRIBUTES)),
                                          "media": torch.nn.Linear(d, len(MEDIA))})

    def forward(self, x: torch.Tensor) -> dict:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, 3, 224, 224) の画像．

        Returns:
            dict[str, torch.Tensor]: 出力層ごとのロジット（format は (B, 4)，attr は (B, 5)，media は (B, 8)）．
        """
        h = self.backbone(pixel_values=x).pooler_output
        return {k: m(h) for k, m in self.heads.items()}

    @staticmethod
    def transforms() -> tuple:
        """学習用と検証用の画像の変換を返す（左右反転は用いず，色の揺らぎのみとする）．

        Returns:
            tuple: (学習用の変換, 検証用の変換)．
        """
        from torchvision import transforms as T

        norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        return (T.Compose([T.Resize((224, 224)), T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), norm]),
                T.Compose([T.Resize((224, 224)), T.ToTensor(), norm]))


def load_model(ModelClass, weight_path: str = None, seed: int = 0) -> torch.nn.Module:
    """モデルをインスタンス化し，必要に応じて重みを読み込む．

    Args:
        ModelClass (type[torch.nn.Module]): モデルのクラス．
        weight_path (str): 重みのファイルのパス．
        seed (int): 出力層の初期値を固定する乱数のシード．

    Returns:
        torch.nn.Module: モデル．
    """
    set_seed(seed)
    model = ModelClass()
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return model


def load_dataloader(teacher: str, use_rare: bool, transforms: tuple, batch_size: int, seed: int = 0) -> tuple:
    """学習用と検証用のデータローダーと，重み付き損失の重みの計算に用いる学習用のラベルを作る．

    Args:
        teacher (str): 教師の構成の名前．
        use_rare (bool): 少ない区分の追加分を学習用に加えるか．
        transforms (tuple): (学習用の変換, 検証用の変換)．
        batch_size (int): バッチサイズ．
        seed (int): 乱数のシード．

    Returns:
        tuple[DataLoader, DataLoader, dict]: 学習用と検証用のデータローダー，学習用のラベルの配列．
    """
    set_seed(seed)
    lab = load_teacher_labels(f"{teacher}_pool", f"{teacher}_rare" if use_rare else None)
    tr = lab.filter(pl.col("split") == "train")
    va = lab.filter((pl.col("split") == "val") & (pl.col("source") == "base"))
    g = torch.Generator()
    g.manual_seed(seed)
    y_tr = label_arrays(tr)
    kw = {"batch_size": batch_size, "num_workers": 16, "pin_memory": True, "persistent_workers": True}
    return (DataLoader(ThreeAxisDataset(tr["fileId"].to_list(), y_tr, transforms[0]), shuffle=True, generator=g,
                       drop_last=True, **kw),
            DataLoader(ThreeAxisDataset(va["fileId"].to_list(), label_arrays(va), transforms[1]), shuffle=False, **kw),
            y_tr)


def loss_func(outputs: dict, teacher_signals: dict, weights: dict = None) -> torch.Tensor:
    """形式の交差エントロピーと属性・媒体の2値交差エントロピーの和を，1データあたりの平均で求める．

    Args:
        outputs (dict[str, torch.Tensor]): 出力層ごとのロジット．
        teacher_signals (dict[str, torch.Tensor]): 教師ラベル．
        weights (dict): 重み付き損失の重み（None で重みなし）．

    Returns:
        torch.Tensor: スカラーの誤差．
    """
    return three_head_loss(outputs, teacher_signals, weights)


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
              optimizer: torch.optim.Optimizer = None, weights: dict = None) -> dict:
    """1つのミニバッチを学習または検証する．

    Args:
        model (torch.nn.Module): モデル．
        inputs (torch.Tensor): 形状 (B, 3, 224, 224) の画像．
        teacher_signals (dict[str, torch.Tensor]): 教師ラベル．
        optimizer (torch.optim.Optimizer): 最適化手法．None の場合は勾配を計算しない．
        weights (dict): 重み付き損失の重み．

    Returns:
        dict[str, float]: loss と metrics_func の指標．
    """
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(optimizer is not None):
        outputs = {k: v.float() for k, v in model(inputs).items()}
        loss = loss_func(outputs, teacher_signals, weights)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"loss": loss.item(), **metrics_func(outputs, teacher_signals)}


def epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer = None,
          weights: dict = None) -> dict:
    """データローダーの全データを1回学習または検証する．

    Args:
        model (torch.nn.Module): モデル．
        dataloader (DataLoader): データローダー．
        optimizer (torch.optim.Optimizer): 最適化手法．None の場合は検証のみ．
        weights (dict): 重み付き損失の重み．

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
        m = iteration(model, inputs, teacher_signals, optimizer, weights)
        b = len(inputs)
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + v * b
        n += b
        bar.set_postfix({k: f"{v / n:.3f}" for k, v in totals.items()})
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader: DataLoader) -> tuple:
    """データローダーの全データの確率と教師ラベルを集める．

    Args:
        model (torch.nn.Module): モデル．
        loader (DataLoader): データローダー．

    Returns:
        tuple[dict, dict]: 出力層ごとの確率と，ラベルの配列．
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
        hp (dict): その他のハイパーパラメータ（lr，weight_decay，teacher，use_rare，weighted）．

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    train_loader, val_loader, y_tr = load_dataloader(hp["teacher"], hp["use_rare"], ModelClass.transforms(),
                                                     batch_size, seed=seed)
    model = load_model(ModelClass, seed=seed).cuda().to(memory_format=torch.channels_last)
    weights = weights_for(hp["weighted"], y_tr, "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    names = ["loss", "format_accuracy", "attr_accuracy", "media_accuracy"]
    logger = ResultLogger()
    logger.set_names(*[f"train_{k}" for k in names], *[f"val_{k}" for k in names], "val_overall_macro_f1")
    best = -1.0
    for ep in range(epochs + 1):
        m_tr = epoch(model, train_loader, optimizer if ep > 0 else None, weights)
        m_va = epoch(model, val_loader, None, weights)
        probs, y = collect_probs(model, val_loader)
        f1 = evaluate(labels_frame(y), to_frame(probs))["overall_macro_f1"]
        logger(*[m_tr[k] for k in names], *[m_va[k] for k in names], f1)
        print(f"    epoch {ep}: train loss={m_tr['loss']:.3f} fmt={m_tr['format_accuracy']:.3f} | "
              f"val loss={m_va['loss']:.3f} fmt={m_va['format_accuracy']:.3f} overallF1={f1:.3f}", flush=True)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
    logger.save(os.path.join(target_dir, "log.json"))
    torch.save(model.state_dict(), os.path.join(target_dir, "last_model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(target_dir, "last_optimizer.pth"))


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

    def loader(ids: list) -> DataLoader:
        """お題IDのリストの予測用のデータローダーを作る．

        Args:
            ids (list[int]): お題IDのリスト．

        Returns:
            DataLoader: データローダー．
        """
        dummy = {"format": np.zeros(len(ids), int), "attr": np.zeros((len(ids), len(ATTRIBUTES)), np.float32),
                 "media": np.zeros((len(ids), len(MEDIA)), np.float32)}
        return DataLoader(ThreeAxisDataset(ids, dummy, val_tf), batch_size=256, num_workers=16, pin_memory=True)

    probs, _ = collect_probs(model, loader(labels["fileId"].to_list()))
    meta = {"method": "finetune", "model": HF_NAME, "seed": seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE, **hp}
    if measure_speed:
        ids = load_pool()["fileId"].to_list()[:10000]
        start = time.perf_counter()
        collect_probs(model, loader(ids))
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
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--no_rare", action="store_true", help="追加分を用いる条件を省略する")
    parser.add_argument("--weight_modes", nargs="+", default=None,
                        choices=["unweighted", "weighted", "weighted_sel"])
    parser.add_argument("--only_rare", action="store_true", help="追加分を用いる条件のみ実行する")
    args = parser.parse_args()
    rare_opts = [False] if args.no_rare else ([True] if args.only_rare else SEARCH_SPACE["use_rare"])
    modes = args.weight_modes or SEARCH_SPACE["weighted"]
    for use_rare, weighted, seed in itertools.product(rare_opts, modes, SEEDS):
        wname = {False: "unweighted", True: "weighted"}.get(weighted, weighted)
        tag = f"{args.teacher}__{'base+rare' if use_rare else 'base'}__{wname}"
        target_dir = os.path.join(OUTPUT_DIR, "finetune_siglip2_base", tag, str(seed))
        if is_done(target_dir):
            continue
        print(f"[finetune] {tag} seed={seed}", flush=True)
        hp = {"lr": LR, "weight_decay": 0.05, "teacher": args.teacher, "use_rare": use_rare, "weighted": weighted}
        set_seed(seed)
        if not os.path.exists(os.path.join(target_dir, "log.json")):
            train(target_dir, SigLIP2BaseThreeHead, load_dataloader, EPOCHS, BATCH_SIZE, seed, hp)
        evaluate_run(target_dir, SigLIP2BaseThreeHead, seed, hp, measure_speed=(seed == 0))


if __name__ == "__main__":
    main()
