"""画像外の知識の判定を重視して，小さな VLM の分類器（ex016 model.py）を微調整する．

ex016 の二段階の微調整（教師のラベル → 目視のラベル931件）を基に，次を変えられるようにする．
- 教師のラベル（--teacher）：画像外の知識を ex017 teacher.py の方法（kt, src, src_fs, src_sc）のラベルに差し替える．
  発話の型は ex015 の 32B one_kt_cap のまま．src_sc は5回の生成の「はい」の割合を確率の目標（ソフトラベル）にできる（--soft）．
- 損失の重み（--w_outside）：画像外の知識の2値交差エントロピーに掛ける重み（発話の型は1）．
- モデル（--models）：ex016 model.py の MODELS に InternVL3.5-8B（iv8）を加える．
評価・保存・速度の測定は ex016 train.py と同じで，各エポックの後に検証データ（合意ラベル）の調整用・最終評価用の F1 を記録し，
調整用の「画像外の知識の F1」が最も高いエポックの重みと予測を保存する（ex016 は6区分＋画像外の和で選んだ）．

ハイパーパラメータの探索空間と seed は CONFIGS と SEEDS で指定する．学習済みの条件（final_model.pt がある）は省略する．

出力：outputs/ex017_boke_outside/ft/{モデル}/{条件}[__stage2]/{seed}/（log.json，eval.csv，speed.json，best_model.pt，final_model.pt，
final_optimizer.pt，meta.json）

実行例:
    .venv_annotate/bin/python programs/ex017_boke_outside/train.py --models iv2 --teacher src_sc --soft --w_outside 3
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import D16, EX16_DIR, OUTPUT_DIR, VOICES, _load, image_path, load_eval, load_human, load_train  # noqa: E402
from machine_learning_utils import ResultLogger, set_seed  # noqa: E402

_own = sys.modules["data"]
sys.modules["data"] = D16
M16 = _load("ex016_model_for_v5", os.path.join(EX16_DIR, "model.py"))
sys.modules["model"] = M16
T16 = _load("ex016_train_for_v5", os.path.join(EX16_DIR, "train.py"))
sys.modules["data"] = _own
M16.MODELS["iv8"] = "OpenGVLab/InternVL3_5-8B-HF"

# order_024：教師のラベルで1エポック，目視のラベルで4エポック
CONFIGS = {"lr": [1e-4], "epochs": [1], "batch_size": [16], "lora_r": [16]}
SEEDS = [0]
STAGE2 = {"lr": 5e-5, "epochs": 4, "batch_size": 8}


class SoftDataset(T16.BokeDataset):
    """画像外の知識の目標を確率（0〜1）のまま返すデータセット．"""

    def __getitem__(self, i: int) -> tuple:
        """1件を返す．

        Args:
            i (int): 番号．

        Returns:
            tuple[PIL.Image.Image, str, int, float]: 画像，入力の文，発話の型の番号，画像外の知識の目標．
        """
        fid, text, fmt, voice, outside = self.rows[i]
        return M16.small_image(image_path(fid)), M16.build_prompt(fmt, text), VOICES.index(voice), float(outside)


def make_loader(df: pl.DataFrame, key: str, batch_size: int, shuffle: bool, seed: int) -> torch.utils.data.DataLoader:
    """データローダーを作る（画像外の知識は確率のまま）．

    Args:
        df (pl.DataFrame): 列 odaiId, text, format, voice, outside（float）を持つ表．
        key (str): モデルのキー．
        batch_size (int): バッチサイズ．
        shuffle (bool): 並べ替えるか．
        seed (int): 乱数のシード．

    Returns:
        torch.utils.data.DataLoader: データローダー．
    """
    return torch.utils.data.DataLoader(SoftDataset(df.with_columns(pl.col("outside").cast(pl.Float64))), batch_size=batch_size,
                                       shuffle=shuffle, num_workers=8, collate_fn=T16.Collate(key),
                                       generator=torch.Generator().manual_seed(seed))


class LossFunc(T16.LossFunc):
    """画像外の知識の誤差に重みを掛けた誤差．"""

    def __init__(self, voice_weight: torch.Tensor, pos_weight: torch.Tensor, w_outside: float = 1.0):
        """
        Args:
            voice_weight (torch.Tensor): 発話の型の重み (6,)．
            pos_weight (torch.Tensor): 画像外の知識の正例の重み．
            w_outside (float): 画像外の知識の誤差に掛ける重み．
        """
        super().__init__(voice_weight, pos_weight)
        self.w_outside = w_outside

    def __call__(self, outputs: tuple, teacher_signals: dict) -> torch.Tensor:
        """1データあたりの誤差の平均を求める．

        Args:
            outputs (tuple[torch.Tensor, torch.Tensor]): 発話の型と画像外の知識のロジット．
            teacher_signals (dict[str, torch.Tensor]): 教師信号（outside は確率）．

        Returns:
            torch.Tensor: スカラーの誤差．
        """
        lv, lo = outputs[0].float(), outputs[1].float()
        return (torch.nn.functional.cross_entropy(lv, teacher_signals["voice"], weight=self.voice_weight.to(lv.device))
                + self.w_outside * torch.nn.functional.binary_cross_entropy_with_logits(
                    lo, teacher_signals["outside"].float(), pos_weight=self.pos_weight.to(lo.device)))


def teacher_df(teacher: str, soft: bool) -> pl.DataFrame:
    """教師のラベルの学習データを作る（画像外の知識を指定した方法のラベルに差し替える）．

    Args:
        teacher (str): ex015（one_kt_cap のまま）または ex017 teacher.py の方法．
        soft (bool): 画像外の知識を確率（p_outside）にするか．

    Returns:
        pl.DataFrame: 列 odaiId, text, format, voice, outside（float）を持つ表．
    """
    df = load_train()
    if teacher != "ex015":
        lab = pl.read_csv(os.path.join(OUTPUT_DIR, "teacher", teacher, "train.csv")).sort("row")
        assert lab.height == df.height
        df = df.with_columns((lab["p_outside"] if soft else lab["outside"].cast(pl.Float64)).alias("outside"))
    return df.select("odaiId", "text", "format", "voice", pl.col("outside").cast(pl.Float64))


def class_weights(df: pl.DataFrame) -> tuple:
    """発話の型の重みと画像外の知識の正例の重みを求める（確率の目標にも対応）．

    Args:
        df (pl.DataFrame): 学習データ．

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 重み (6,) と正例の重み．
    """
    cnt = np.array([(df["voice"] == v).sum() for v in VOICES], dtype=float)
    w = np.sqrt(cnt.sum() / (len(VOICES) * np.maximum(cnt, 1)))
    pos = float(df["outside"].cast(pl.Float64).sum())
    return torch.tensor(w, dtype=torch.float32), torch.tensor((df.height - pos) / max(pos, 1.0))


def run(target_dir: str, key: str, train_df: pl.DataFrame, params: dict, w_outside: float, init: str = None, val_df=None,
        seed: int = 0) -> None:
    """学習し，各エポックの後に検証データで評価して，調整用の画像外の知識の F1 が最良のエポックを保存する．

    Args:
        target_dir (str): 保存先．
        key (str): モデルのキー．
        train_df (pl.DataFrame): 学習データ．
        params (dict): lr, epochs, batch_size, lora_r．
        w_outside (float): 画像外の知識の誤差の重み．
        init (str): 初期の重み（best_model.pt．None で VLM の初期値）．
        val_df (pl.DataFrame): 教師のラベルの検証用（None なら記録しない）．
        seed (int): 乱数のシード．

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    set_seed(seed)
    ev = load_eval()
    model = M16.load_model(key, init, seed=seed, lora_r=params["lora_r"]).cuda()
    loss_func = LossFunc(*class_weights(train_df), w_outside=w_outside)
    tr_loader = make_loader(train_df, key, params["batch_size"], True, seed)
    ev_loader = make_loader(ev.select("odaiId", "text", "format", "voice", "outside"), key, 32, False, seed)
    va_loader = make_loader(val_df, key, 32, False, seed) if val_df is not None else None
    opt, sched = T16.make_optimizer(model, params["lr"], params["epochs"] * len(tr_loader))
    logger = ResultLogger()
    logger.set_names("train_loss", "train_voice_acc", "train_outside_acc", "val_loss", "val_voice_acc", "val_outside_acc",
                     "dev_voice6_f1", "dev_outside_f1", "test_voice6_f1", "test_outside_f1")
    best = -1.0
    for ep in range(params["epochs"] + 1):
        tr = T16.epoch(model, tr_loader, loss_func, opt if ep > 0 else None, sched if ep > 0 else None) if ep > 0 else \
            {"loss": float("nan"), "voice_acc": float("nan"), "outside_acc": float("nan")}
        va = T16.epoch(model, va_loader, loss_func) if va_loader is not None else {"loss": float("nan"), "voice_acc": float("nan"),
                                                                                  "outside_acc": float("nan")}
        pv, po, sec = T16.predict(model, ev_loader)
        hs = T16.human_scores(ev, pv, po)
        logger(tr["loss"], tr["voice_acc"], tr["outside_acc"], va["loss"], va["voice_acc"], va["outside_acc"],
               hs["dev_voice6_f1"], hs["dev_outside_f1"], hs["test_voice6_f1"], hs["test_outside_f1"])
        logger.save(os.path.join(target_dir, "log.json"))
        print(key, os.path.basename(os.path.dirname(target_dir)), "epoch", ep, {k: round(v, 3) for k, v in hs.items()}, flush=True)
        if ep > 0 and hs["dev_outside_f1"] > best:
            best = hs["dev_outside_f1"]
            torch.save(model.trainable_state_dict(), os.path.join(target_dir, "best_model.pt"))
            T16.save_pred(os.path.join(target_dir, "eval.csv"), ev, pv, po)
            json.dump({"best_epoch": ep, "model_per_sec": ev.height / sec}, open(os.path.join(target_dir, "best.json"), "w"))
    torch.save(model.trainable_state_dict(), os.path.join(target_dir, "final_model.pt"))
    torch.save(opt.state_dict(), os.path.join(target_dir, "final_optimizer.pt"))
    del model, opt
    torch.cuda.empty_cache()


def main() -> None:
    """指定したモデル・教師のラベル・重みで，二段階の微調整を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--teacher", default="ex015")
    parser.add_argument("--soft", action="store_true")
    parser.add_argument("--w_outside", type=float, default=1.0)
    parser.add_argument("--no_stage2", action="store_true")
    args = parser.parse_args()
    full = teacher_df(args.teacher, args.soft)
    hu = load_human().select("odaiId", "text", "format", "voice", pl.col("outside").cast(pl.Float64))
    for key, lr, ep, bs, r, seed in itertools.product(args.models, CONFIGS["lr"], CONFIGS["epochs"], CONFIGS["batch_size"],
                                                     CONFIGS["lora_r"], SEEDS):
        cond = f"{args.teacher}{'_soft' if args.soft else ''}_wo{args.w_outside:g}_lr{lr}_ep{ep}_bs{bs}_r{r}"
        d1 = os.path.join(OUTPUT_DIR, "ft", key, cond, str(seed))
        if not os.path.exists(os.path.join(d1, "final_model.pt")):
            tr, va = train_test_split(np.arange(full.height), test_size=0.1, random_state=seed)
            json.dump({"model": M16.MODELS[key], "teacher": args.teacher, "soft": args.soft, "w_outside": args.w_outside, "lr": lr,
                       "epochs": ep, "batch_size": bs, "lora_r": r, "seed": seed}, open(os.path.join(os.makedirs(d1, exist_ok=True) or d1,
                                                                                                        "meta.json"), "w"))
            run(d1, key, full[tr.tolist()], {"lr": lr, "epochs": ep, "batch_size": bs, "lora_r": r}, args.w_outside,
                val_df=full[va.tolist()], seed=seed)
            T16.measure_speed(M16.load_model(key, os.path.join(d1, "best_model.pt"), seed=seed, gradient_checkpointing=False).cuda(),
                              key, load_eval(), d1)
            torch.cuda.empty_cache()
        if args.no_stage2:
            continue
        d2 = os.path.join(OUTPUT_DIR, "ft", key, cond + "__stage2", str(seed))
        if not os.path.exists(os.path.join(d2, "final_model.pt")):
            run(d2, key, hu, {"lr": STAGE2["lr"], "epochs": STAGE2["epochs"], "batch_size": STAGE2["batch_size"], "lora_r": r},
                args.w_outside, init=os.path.join(d1, "best_model.pt"), seed=seed)


if __name__ == "__main__":
    main()
