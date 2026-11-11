"""小さな VLM の分類器（model.py）を，教師のラベルで微調整し，検証データで評価する．

学習データ：data.load_train（教師 Qwen3-VL-32B one_kt_cap のラベル，34,634件）を 9:1 で学習用と検証用に分ける．
誤差：発話の型は出現の逆数の平方根で重みを付けた交差エントロピー，画像外の知識は正例の比で重みを付けた2値交差エントロピーの和．
--correction forward で，教師のラベルの誤り方（調整用の合意ラベルで推定した遷移行列）による forward correction を行う．
目視のラベルは合意ラベル（data.load_eval の既定）を用いる．
各エポックの後に，教師のラベルの学習用・検証用の誤差と一致率，目視のラベルの調整用（dev）・最終評価用（test）の F1 を記録し，
調整用の「6区分のマクロ F1＋画像外の知識の F1」が最も高いエポックの重み（LoRA と出力層）と予測を保存する．
最後に，検証データ620件の推論の速度（1秒あたりの件数）を測る．

--stage2 を指定すると，学習済みの重みから，学習用の目視のラベル（human_labels.txt，931件）でさらに微調整する（{条件}__stage2）．
--human_only を指定すると，VLM の初期値から目視のラベルのみで学習する（human_only/…，対照）．

ハイパーパラメータの探索空間と seed は CONFIGS と SEEDS で指定する．学習済みの条件（eval.csv がある）は省略する．

出力：outputs/ex016_boke_small_vlm/ft/{モデル}/{条件}/{seed}/（log.json，eval.csv，speed.json，best_model.pt，final_model.pt，
final_optimizer.pt，meta.json）

実行例:
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/train.py --models q2 iv2
    .venv_annotate/bin/python programs/ex016_boke_small_vlm/train.py --models q2 --stage2
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import OUTPUT_DIR, VOICES, image_path, load_eval, load_human, load_train, noise_transition  # noqa: E402
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from model import build_prompt, encode, load_model, load_processor, small_image  # noqa: E402

# 条件：学習率（LoRA），エポック数，バッチサイズ，LoRA のランク
# order_023：教師のラベル34,634件で1エポック（q4・iv4 は2エポックで実行し，1エポック目が最良だったため以後は1エポック），目視のラベル931件で4エポック
CONFIGS = {"lr": [1e-4], "epochs": [1], "batch_size": [16], "lora_r": [16]}
SEEDS = [0]
STAGE2 = {"lr": 5e-5, "epochs": 4, "batch_size": 8}


class BokeDataset(torch.utils.data.Dataset):
    """お題画像・ボケの文・ラベルの組のデータセット．"""

    def __init__(self, df: pl.DataFrame):
        """
        Args:
            df (pl.DataFrame): 列 odaiId, text, format, voice, outside を持つ表．
        """
        self.rows = df.select("odaiId", "text", "format", "voice", "outside").rows()

    def __len__(self) -> int:
        """件数を返す．

        Returns:
            int: 件数．
        """
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple:
        """1件を返す．

        Args:
            i (int): 番号．

        Returns:
            tuple[PIL.Image.Image, str, int, float]: 画像，入力の文，発話の型の番号，画像外の知識（0/1）．
        """
        fid, text, fmt, voice, outside = self.rows[i]
        return small_image(image_path(fid)), build_prompt(fmt, text), VOICES.index(voice), float(bool(outside))


class Collate:
    """processor でミニバッチを作る関数オブジェクト．"""

    def __init__(self, key: str):
        """
        Args:
            key (str): モデルのキー．
        """
        self.key = key
        self.proc = None

    def __call__(self, batch: list) -> tuple:
        """ミニバッチを作る．

        Args:
            batch (list[tuple]): BokeDataset の要素のリスト．

        Returns:
            tuple[dict, dict]: モデルの入力と，教師信号（voice: (B,) の番号，outside: (B,) の 0/1）．
        """
        if self.proc is None:
            self.proc = load_processor(self.key)
        images, prompts, v, o = zip(*batch)
        return encode(self.proc, list(images), list(prompts)), {"voice": torch.tensor(v), "outside": torch.tensor(o)}


def make_loader(df: pl.DataFrame, key: str, batch_size: int, shuffle: bool, seed: int) -> torch.utils.data.DataLoader:
    """データローダーを作る．

    Args:
        df (pl.DataFrame): データの表．
        key (str): モデルのキー．
        batch_size (int): バッチサイズ．
        shuffle (bool): 並べ替えるか．
        seed (int): 乱数のシード．

    Returns:
        torch.utils.data.DataLoader: データローダー．
    """
    return torch.utils.data.DataLoader(BokeDataset(df), batch_size=batch_size, shuffle=shuffle, num_workers=8, collate_fn=Collate(key),
                                       generator=torch.Generator().manual_seed(seed), persistent_workers=False)


def load_dataloader(key: str, seed: int = 0, batch_size: int = 16) -> tuple:
    """教師のラベルの学習用・検証用と，目視のラベルの検証データのデータローダーを作る．

    Args:
        key (str): モデルのキー．
        seed (int): 乱数のシード．
        batch_size (int): バッチサイズ．

    Returns:
        tuple: 学習用，検証用，目視の検証データ（620件）のデータローダーと，目視の検証データの表．
    """
    set_seed(seed)
    df = load_train()
    tr, va = train_test_split(np.arange(df.height), test_size=0.1, random_state=seed)
    ev = load_eval()
    return (make_loader(df[tr.tolist()], key, batch_size, True, seed), make_loader(df[va.tolist()], key, 32, False, seed),
            make_loader(ev, key, 32, False, seed), ev)


class LossFunc:
    """重み付きの誤差（教師のラベルの誤り方で補正する forward correction を選べる）．

    forward correction（Patrini et al., 2017）：モデルの出力を真のラベルの確率 p とみなし，教師のラベルの確率を p T（T は真のラベル →
    教師のラベルの遷移行列）として教師のラベルとの誤差を求める．予測には補正前の p を用いる．
    """

    def __init__(self, voice_weight: torch.Tensor, pos_weight: torch.Tensor, transition: tuple = None):
        """
        Args:
            voice_weight (torch.Tensor): 発話の型の重み (6,)．
            pos_weight (torch.Tensor): 画像外の知識の正例の重み（スカラー）．
            transition (tuple[np.ndarray, np.ndarray]): 発話の型 (6, 6) と画像外の知識 (2, 2) の遷移行列（None で補正なし）．
        """
        self.voice_weight, self.pos_weight = voice_weight, pos_weight
        self.tv = self.to = None
        if transition is not None:
            self.tv, self.to = (torch.tensor(t, dtype=torch.float32) for t in transition)

    def __call__(self, outputs: tuple, teacher_signals: dict) -> torch.Tensor:
        """1データあたりの誤差の平均を求める．

        Args:
            outputs (tuple[torch.Tensor, torch.Tensor]): 発話の型のロジット (B, 6) と画像外の知識のロジット (B,)．
            teacher_signals (dict[str, torch.Tensor]): 教師信号．

        Returns:
            torch.Tensor: スカラーの誤差．
        """
        lv, lo = outputs[0].float(), outputs[1].float()
        yv, yo = teacher_signals["voice"], teacher_signals["outside"]
        w, pw = self.voice_weight.to(lv.device), self.pos_weight.to(lo.device)
        if self.tv is None:
            return (torch.nn.functional.cross_entropy(lv, yv, weight=w)
                    + torch.nn.functional.binary_cross_entropy_with_logits(lo, yo, pos_weight=pw))
        pn = torch.softmax(lv, 1) @ self.tv.to(lv.device)
        q = torch.sigmoid(lo)
        to = self.to.to(lo.device)
        qn = (q * to[1, 1] + (1 - q) * to[0, 1]).clamp(1e-6, 1 - 1e-6)
        loss_o = -(pw * yo * torch.log(qn) + (1 - yo) * torch.log(1 - qn)).mean()
        return torch.nn.functional.nll_loss(torch.log(pn.clamp_min(1e-8)), yv, weight=w) + loss_o


def metrics_func(outputs: tuple, teacher_signals: dict) -> dict:
    """教師信号との一致率を求める．

    Args:
        outputs (tuple[torch.Tensor, torch.Tensor]): ロジット．
        teacher_signals (dict[str, torch.Tensor]): 教師信号．

    Returns:
        dict[str, float]: voice_acc（6区分の一致率），outside_acc（画像外の知識の一致率）．
    """
    lv, lo = outputs
    return {"voice_acc": (lv.argmax(1) == teacher_signals["voice"]).float().mean().item(),
            "outside_acc": ((lo > 0).float() == teacher_signals["outside"]).float().mean().item()}


def to_device(inputs: dict, device) -> dict:
    """入力を GPU に移す（浮動小数の画像は bfloat16 にする）．

    Args:
        inputs (dict[str, torch.Tensor]): 入力．
        device (torch.device): 移動先．

    Returns:
        dict[str, torch.Tensor]: 移動した入力．
    """
    return {k: (v.to(device, torch.bfloat16) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}


def iteration(model, inputs: dict, teacher_signals: dict, loss_func, optimizer=None, scheduler=None) -> dict:
    """1つのミニバッチを学習または検証する．

    Args:
        model (VLMClassifier): 分類器．
        inputs (dict[str, torch.Tensor]): 入力（GPU 上）．
        teacher_signals (dict[str, torch.Tensor]): 教師信号（GPU 上）．
        loss_func (LossFunc): 誤差関数．
        optimizer (torch.optim.Optimizer): 最適化手法（None で検証）．
        scheduler (torch.optim.lr_scheduler.LRScheduler): 学習率のスケジューラ．

    Returns:
        dict[str, float]: 誤差と一致率．
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
    loss = loss_func(outputs, teacher_signals)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return {"loss": loss.item(), **metrics_func(outputs, teacher_signals)}


def epoch(model, dataloader, loss_func, optimizer=None, scheduler=None) -> dict:
    """1つのデータローダーの全データを学習または検証する．

    Args:
        model (VLMClassifier): 分類器．
        dataloader (torch.utils.data.DataLoader): データローダー．
        loss_func (LossFunc): 誤差関数．
        optimizer (torch.optim.Optimizer): 最適化手法（None で検証）．
        scheduler (torch.optim.lr_scheduler.LRScheduler): 学習率のスケジューラ．

    Returns:
        dict[str, float]: 1データあたりの誤差と一致率の平均．
    """
    device = next(p for p in model.parameters() if p.requires_grad).device
    model.train(optimizer is not None)
    total, n = {}, 0
    bar = tqdm(dataloader, leave=False)
    for inputs, signals in bar:
        inputs, signals = to_device(inputs, device), {k: v.to(device) for k, v in signals.items()}
        with torch.set_grad_enabled(optimizer is not None):
            m = iteration(model, inputs, signals, loss_func, optimizer, scheduler)
        b = len(signals["voice"])
        total = {k: total.get(k, 0.0) + v * b for k, v in m.items()}
        n += b
        bar.set_postfix({k: round(v / n, 4) for k, v in total.items()})
    return {k: v / n for k, v in total.items()}


@torch.no_grad()
def predict(model, dataloader) -> tuple:
    """確率を求める．

    Args:
        model (VLMClassifier): 分類器．
        dataloader (torch.utils.data.DataLoader): データローダー（並べ替えない）．

    Returns:
        tuple[np.ndarray, np.ndarray, float]: 発話の型の確率 (N, 6)，画像外の知識の確率 (N,)，モデルの順伝播の秒数．
    """
    device = next(p for p in model.parameters() if p.requires_grad).device
    model.eval()
    pv, po, sec = [], [], 0.0
    for inputs, _ in tqdm(dataloader, leave=False):
        inputs = to_device(inputs, device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lv, lo = model(inputs)
        torch.cuda.synchronize()
        sec += time.perf_counter() - start
        pv.append(torch.softmax(lv, 1).float().cpu().numpy())
        po.append(torch.sigmoid(lo).float().cpu().numpy())
    return np.concatenate(pv), np.concatenate(po), sec


def human_scores(ev: pl.DataFrame, pv: np.ndarray, po: np.ndarray) -> dict:
    """目視のラベルとの F1 を，調整用と最終評価用で求める．

    Args:
        ev (pl.DataFrame): 検証データ．
        pv (np.ndarray): 発話の型の確率．
        po (np.ndarray): 画像外の知識の確率．

    Returns:
        dict[str, float]: {dev,test}_{voice6_f1,outside_f1}．
    """
    out = {}
    for split in ["dev", "test"]:
        m = (ev["split"] == split).to_numpy()
        y = ev["voice"].to_numpy()[m]
        q = np.array(VOICES)[pv[m].argmax(1)]
        out[f"{split}_voice6_f1"] = f1_score(y, q, labels=VOICES, average="macro", zero_division=0)
        out[f"{split}_outside_f1"] = f1_score(ev["outside"].to_numpy()[m], po[m] > 0.5, zero_division=0)
    return out


def save_pred(path: str, ev: pl.DataFrame, pv: np.ndarray, po: np.ndarray) -> None:
    """予測を保存する．

    Args:
        path (str): 保存先．
        ev (pl.DataFrame): 検証データ．
        pv (np.ndarray): 発話の型の確率．
        po (np.ndarray): 画像外の知識の確率．

    Returns:
        None
    """
    d = {"no": ev["no"], "voice": np.array(VOICES)[pv.argmax(1)], "outside": po > 0.5, "p_outside": po}
    d.update({f"p_{v}": pv[:, i] for i, v in enumerate(VOICES)})
    pl.DataFrame(d).write_csv(path)


def class_weights(df: pl.DataFrame) -> tuple:
    """発話の型の重み（出現の逆数の平方根）と画像外の知識の正例の重みを求める．

    Args:
        df (pl.DataFrame): 学習データ．

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 重み (6,) と正例の重み．
    """
    cnt = np.array([(df["voice"] == v).sum() for v in VOICES], dtype=float)
    w = np.sqrt(cnt.sum() / (len(VOICES) * np.maximum(cnt, 1)))
    pos = float(df["outside"].cast(pl.Boolean).sum())
    return torch.tensor(w, dtype=torch.float32), torch.tensor((df.height - pos) / max(pos, 1.0))


def make_optimizer(model, lr: float, steps: int) -> tuple:
    """最適化手法（AdamW．出力層の学習率は10倍）と，線形の warmup（5%）の後に cosine で下げるスケジューラを作る．

    Args:
        model (VLMClassifier): 分類器．
        lr (float): LoRA の学習率．
        steps (int): 全ステップ数．

    Returns:
        tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]: 最適化手法とスケジューラ．
    """
    heads = [p for n, p in model.named_parameters() if p.requires_grad and "_head" in n]
    body = [p for n, p in model.named_parameters() if p.requires_grad and "_head" not in n]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": heads, "lr": lr * 10}], weight_decay=0.01)
    warm = max(1, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    return opt, sched


def train(target_dir: str, key: str, params: dict, seed: int = 0, correction: str = "none") -> None:
    """教師のラベルで分類器を学習し，記録・重み・予測・速度を保存する．

    Args:
        target_dir (str): 保存先のディレクトリ．
        key (str): モデルのキー．
        params (dict): lr, epochs, batch_size, lora_r．
        seed (int): 乱数のシード．
        correction (str): none（補正なし）または forward（調整用の合意ラベルで推定した遷移行列による forward correction）．

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    tr_loader, va_loader, ev_loader, ev = load_dataloader(key, seed, params["batch_size"])
    model = load_model(key, seed=seed, lora_r=params["lora_r"]).cuda()
    loss_func = LossFunc(*class_weights(load_train()), transition=noise_transition() if correction == "forward" else None)
    opt, sched = make_optimizer(model, params["lr"], params["epochs"] * len(tr_loader))
    logger = ResultLogger()
    names = [f"{s}_{m}" for s in ["train", "val"] for m in ["loss", "voice_acc", "outside_acc"]]
    names += [f"{s}_{m}" for s in ["dev", "test"] for m in ["voice6_f1", "outside_f1"]]
    logger.set_names(*names)
    best = -1.0
    for ep in range(params["epochs"] + 1):
        tr = epoch(model, tr_loader, loss_func, opt, sched) if ep > 0 else epoch(model, tr_loader, loss_func)
        va = epoch(model, va_loader, loss_func)
        pv, po, sec = predict(model, ev_loader)
        hs = human_scores(ev, pv, po)
        logger(*[tr[m] for m in ["loss", "voice_acc", "outside_acc"]], *[va[m] for m in ["loss", "voice_acc", "outside_acc"]],
               *[hs[k] for k in ["dev_voice6_f1", "dev_outside_f1", "test_voice6_f1", "test_outside_f1"]])
        logger.save(os.path.join(target_dir, "log.json"))
        print(key, "epoch", ep, {k: round(v, 3) for k, v in {**{"train_" + k: v for k, v in tr.items()},
                                                              **{"val_" + k: v for k, v in va.items()}, **hs}.items()}, flush=True)
        score = hs["dev_voice6_f1"] + hs["dev_outside_f1"]
        if ep > 0 and score > best:
            best = score
            torch.save(model.trainable_state_dict(), os.path.join(target_dir, "best_model.pt"))
            save_pred(os.path.join(target_dir, "eval.csv"), ev, pv, po)
            json.dump({"best_epoch": ep, "model_sec": sec, "model_per_sec": ev.height / sec}, open(os.path.join(target_dir, "best.json"), "w"))
    torch.save(model.trainable_state_dict(), os.path.join(target_dir, "final_model.pt"))
    torch.save(opt.state_dict(), os.path.join(target_dir, "final_optimizer.pt"))
    measure_speed(model, key, ev, target_dir)


def measure_speed(model, key: str, ev: pl.DataFrame, target_dir: str) -> None:
    """検証データ620件の推論の速度を，前処理込みとモデルのみで測る（1回目の空の実行の後）．

    Args:
        model (VLMClassifier): 分類器．
        key (str): モデルのキー．
        ev (pl.DataFrame): 検証データ．
        target_dir (str): 保存先．

    Returns:
        None
    """
    out = {}
    for bs in [32, 64]:
        loader = make_loader(ev, key, bs, False, 0)
        predict(model, loader)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, _, sec = predict(model, loader)
        total = time.perf_counter() - start
        out[f"bs{bs}"] = {"model_per_sec": ev.height / sec, "end_to_end_per_sec": ev.height / total,
                          "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
    json.dump(out, open(os.path.join(target_dir, "speed.json"), "w"), indent=1)


def train_stage2(target_dir: str, key: str, stage1_dir: str, seed: int = 0) -> None:
    """教師のラベルで学習した分類器を，学習用の目視のラベル（931件，検証データのお題は含まない）でさらに微調整する．

    各エポックの後に検証データ（調整用・最終評価用）の F1 を記録し，調整用の「6区分のマクロ F1＋画像外の知識の F1」が
    最も高いエポックの予測と重みを保存する．stage1_dir が None なら，VLM の初期値から目視のラベルのみで学習する（対照）．

    Args:
        target_dir (str): 保存先．
        key (str): モデルのキー．
        stage1_dir (str): 教師のラベルで学習した結果のディレクトリ（best_model.pt を読む）．None で初期値から．
        seed (int): 乱数のシード．

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    ev, hu = load_eval(), load_human()
    model = load_model(key, None if stage1_dir is None else os.path.join(stage1_dir, "best_model.pt"), seed=seed).cuda()
    loss_func = LossFunc(*class_weights(hu))
    tr_loader = make_loader(hu, key, STAGE2["batch_size"], True, seed)
    ev_loader = make_loader(ev, key, 32, False, seed)
    epochs = STAGE2["epochs"] if stage1_dir is not None else STAGE2["epochs"] * 2
    opt, sched = make_optimizer(model, STAGE2["lr"] if stage1_dir is not None else 1e-4, epochs * len(tr_loader))
    logger = ResultLogger()
    logger.set_names("train_loss", "train_voice_acc", "train_outside_acc", "dev_voice6_f1", "dev_outside_f1", "test_voice6_f1",
                     "test_outside_f1")
    best = -1.0
    for ep in range(1, epochs + 1):
        tr = epoch(model, tr_loader, loss_func, opt, sched)
        pv, po, sec = predict(model, ev_loader)
        hs = human_scores(ev, pv, po)
        logger(tr["loss"], tr["voice_acc"], tr["outside_acc"], hs["dev_voice6_f1"], hs["dev_outside_f1"], hs["test_voice6_f1"],
               hs["test_outside_f1"])
        logger.save(os.path.join(target_dir, "log.json"))
        print(key, "stage2 epoch", ep, {k: round(v, 3) for k, v in hs.items()}, flush=True)
        score = hs["dev_voice6_f1"] + hs["dev_outside_f1"]
        if score > best:
            best = score
            save_pred(os.path.join(target_dir, "eval.csv"), ev, pv, po)
            torch.save(model.trainable_state_dict(), os.path.join(target_dir, "best_model.pt"))
            json.dump({"best_epoch": ep, "model_per_sec": ev.height / sec}, open(os.path.join(target_dir, "best.json"), "w"))
    torch.save(model.trainable_state_dict(), os.path.join(target_dir, "final_model.pt"))
    torch.save(opt.state_dict(), os.path.join(target_dir, "final_optimizer.pt"))


def main() -> None:
    """指定したモデルと全ての条件で学習する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--correction", default="none", choices=["none", "forward"])
    parser.add_argument("--human_only", action="store_true")
    args = parser.parse_args()
    if args.human_only:
        for key, seed in itertools.product(args.models, SEEDS):
            d = os.path.join(OUTPUT_DIR, "ft", key, "human_only", str(seed))
            if not os.path.exists(os.path.join(d, "final_model.pt")):
                train_stage2(d, key, None, seed)
            torch.cuda.empty_cache()
        return
    for key, lr, ep, bs, r, seed in itertools.product(args.models, CONFIGS["lr"], CONFIGS["epochs"], CONFIGS["batch_size"],
                                                     CONFIGS["lora_r"], SEEDS):
        cond = f"teacher_lr{lr}_ep{ep}_bs{bs}_r{r}" + ("_fc" if args.correction == "forward" else "")
        d1 = os.path.join(OUTPUT_DIR, "ft", key, cond, str(seed))
        if not os.path.exists(os.path.join(d1, "speed.json")):
            json.dump({"model": key, "lr": lr, "epochs": ep, "batch_size": bs, "lora_r": r, "seed": seed, "labels": "q32 one_kt_cap",
                       "correction": args.correction, "gold": "consensus_labels.txt"},
                      open(os.path.join(os.makedirs(d1, exist_ok=True) or d1, "meta.json"), "w"))
            train(d1, key, {"lr": lr, "epochs": ep, "batch_size": bs, "lora_r": r}, seed, args.correction)
        if args.stage2:
            d2 = os.path.join(OUTPUT_DIR, "ft", key, cond + "__stage2", str(seed))
            if not os.path.exists(os.path.join(d2, "final_model.pt")):
                train_stage2(d2, key, d1, seed)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
