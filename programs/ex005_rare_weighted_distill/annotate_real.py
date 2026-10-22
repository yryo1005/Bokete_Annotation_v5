"""蒸留モデルで実データにアノテーションし，少ない区分の評価用データ（第2版）を層別に抽出する．

1. アノテーション：ex005 の最良の蒸留モデル（微調整した SigLIP2-B，追加分あり・重み付き，検証用で最良の seed 1）で，
   収集の候補30万枚から既に使った画像（少ない区分の追加分と少ない区分の評価用データ第1版）を除いた画像に
   3軸の確率を付与する（datasets/ex005_rare_weighted_distill/real_annotation.parquet）．
2. 層別の抽出：少ない区分 c（ゲーム，CG，絵画，カルタ）ごとに，
   - 層 pos：確率が0.5以上（モデルがその区分と判定）から無作為に N_POS 枚
   - 層 border：確率が0.05以上0.5未満から無作為に N_BORDER 枚
   を抽出し，さらに全体から無作為に N_RANDOM 枚（層 random）を抽出する．
   各画像の層と抽出確率（層の画像数に対する抽出数の比）を記録し，適合率と再現率の重み付きの推定に用いる．
3. 目視のラベル付け用の一覧画像を作る．ラベルは rare_eval_labels_v2.txt に「お題ID 形式（K/N） 媒体」で記録する．

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/annotate_real.py --annotate --sample
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, plot_image_grid  # noqa: E402
from data import ATTRIBUTES, DATASET_DIR, FORMATS, MEDIA, ThreeAxisDataset  # noqa: E402

RUN = os.path.join(OUTPUT_DIR, "finetune_siglip2_base", "T_v2_J6M7__base+rare__weighted", "1")
ANN_PATH = os.path.join(DATASET_DIR, "real_annotation.parquet")
SAMPLE_PATH = os.path.join(DATASET_DIR, "rare_eval_v2_ids.csv")
SHEET_DIR = os.path.join(DATASET_DIR, "rare_eval_v2_sheets")
CLASSES = ["game", "cg", "painting", "karuta"]
N_POS, N_BORDER, N_RANDOM = 60, 60, 120
BORDER_LOW = 0.05


def _ex005_module(name: str):
    """ex005 のモジュールをファイルから読み込む（他の実験の同名のモジュールとの衝突を避ける）．

    Args:
        name (str): モジュール名．

    Returns:
        module: 読み込んだモジュール．
    """
    import importlib.util

    if f"ex005_{name}" in sys.modules:
        return sys.modules[f"ex005_{name}"]
    spec = importlib.util.spec_from_file_location(f"ex005_{name}", os.path.join(os.path.dirname(__file__), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ex005_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def annotate() -> pl.DataFrame:
    """未使用の候補の画像に，蒸留モデルで3軸の確率を付与して保存する．

    Returns:
        pl.DataFrame: 列 fileId，p_format_{形式}，p_{属性}，p_{媒体} を持つ表．
    """
    T = _ex005_module("train")
    cand = pl.read_csv(os.path.join(DATASET_DIR, "candidates", "cand.csv"))["fileId"].to_list()
    used = set(pl.read_csv(os.path.join(DATASET_DIR, "teacher_labels", "T_v2_J6M7_rare.csv"))["fileId"].to_list())
    used |= set(pl.read_csv(os.path.join(DATASET_DIR, "rare_eval_ids.csv"))["fileId"].to_list())
    ids = [f for f in cand if f not in used]
    model = T.load_model(T.SigLIP2BaseThreeHead, os.path.join(RUN, "best_model.pth")).cuda()
    model = model.to(memory_format=torch.channels_last)
    dummy = {"format": np.zeros(len(ids), int), "attr": np.zeros((len(ids), len(ATTRIBUTES)), np.float32),
             "media": np.zeros((len(ids), len(MEDIA)), np.float32)}
    loader = torch.utils.data.DataLoader(ThreeAxisDataset(ids, dummy, T.SigLIP2BaseThreeHead.transforms()[1]),
                                         batch_size=256, num_workers=16, pin_memory=True)
    probs, _ = T.collect_probs(model, loader)
    cols = {"fileId": ids}
    cols.update({f"p_format_{c}": probs["format"][:, i] for i, c in enumerate(FORMATS)})
    cols.update({f"p_{c}": probs["attr"][:, i] for i, c in enumerate(ATTRIBUTES)})
    cols.update({f"p_{c}": probs["media"][:, i] for i, c in enumerate(MEDIA)})
    df = pl.DataFrame(cols)
    df.write_parquet(ANN_PATH)
    return df


def class_prob(df: pl.DataFrame, cls: str) -> pl.Series:
    """区分の確率の列を返す（カルタは形式の確率）．

    Args:
        df (pl.DataFrame): annotate の戻り値．
        cls (str): 区分の名前．

    Returns:
        pl.Series: 確率．
    """
    return df["p_format_karuta"] if cls == "karuta" else df[f"p_{cls}"]


def sample(df: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """層別に評価用の画像を抽出し，層と抽出確率を記録する．

    Args:
        df (pl.DataFrame): annotate の戻り値．
        seed (int): 乱数のシード．

    Returns:
        pl.DataFrame: 列 fileId, stratum（"{区分}_pos"，"{区分}_border"，"random"），n_stratum（層の画像数），
        n_drawn（層から抽出した数）を持つ表．
    """
    rng = np.random.default_rng(seed)
    chosen, rows = set(), []
    for c in CLASSES:
        p = class_prob(df, c).to_numpy()
        for name, mask, n in [("pos", p >= 0.5, N_POS), ("border", (p >= BORDER_LOW) & (p < 0.5), N_BORDER)]:
            pool = [f for f in df["fileId"].to_numpy()[mask] if f not in chosen]
            sel = rng.choice(len(pool), min(n, len(pool)), replace=False)
            for i in sel:
                chosen.add(int(pool[i]))
                rows.append({"fileId": int(pool[i]), "stratum": f"{c}_{name}", "n_stratum": int(mask.sum()),
                             "n_drawn": int(len(sel))})
    pool = [f for f in df["fileId"].to_list() if f not in chosen]
    sel = rng.choice(len(pool), N_RANDOM, replace=False)
    rows += [{"fileId": int(pool[i]), "stratum": "random", "n_stratum": len(df), "n_drawn": N_RANDOM} for i in sel]
    out = pl.DataFrame(rows)
    order = rng.permutation(len(out))  # 目視の際に層が分からないよう並びを混ぜる
    out = out[order.tolist()]
    out.write_csv(SAMPLE_PATH)
    os.makedirs(SHEET_DIR, exist_ok=True)
    for s in range(0, len(out), 20):
        part = out[s:s + 20]
        plot_image_grid(part["fileId"].to_list(), [f"{k:02d}:{f}" for k, f in enumerate(part["fileId"])],
                        f"rare eval v2 sheet {s // 20}", os.path.join(SHEET_DIR, f"sheet_{s // 20:02d}.png"), ncols=5)
    return out


def main() -> None:
    """コマンドライン引数に従ってアノテーションと抽出を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    df = annotate() if args.annotate or not os.path.exists(ANN_PATH) else pl.read_parquet(ANN_PATH)
    print(f"アノテーション済み: {len(df)} 枚")
    for c in CLASSES:
        p = class_prob(df, c)
        print(f"  {c}: 判定 {int((p >= 0.5).sum())} 枚（{float((p >= 0.5).mean()):.2%}），"
              f"しきい値付近 {int(((p >= BORDER_LOW) & (p < 0.5)).sum())} 枚")
    if args.sample:
        out = sample(df)
        print(f"評価用に {len(out)} 枚を抽出した")


if __name__ == "__main__":
    main()
