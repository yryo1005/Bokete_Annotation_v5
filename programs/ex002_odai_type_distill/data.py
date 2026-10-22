"""ex002 で用いるデータを扱うモジュール．

- 評価用データの拡充：全お題から無作為に200枚を追加で抽出し，目視のラベルを付与する（ex001 の440枚と合わせて640枚）．
- 教師ラベル用の画像：全お題から無作為に40,000枚を抽出し，学習用36,000枚と検証用4,000枚に分ける．
  検証用は VLM の付けたラベルでハイパーパラメータを選ぶために用い，目視の評価用データは最終評価のみに用いる．
- 学習・検証用のデータローダー（画像エンコーダの微調整用）を定義する．
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import polars as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

EX001_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ex001_odai_image_type")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
# ex001 の data.py は同名のモジュールであるため，別名 ex001_data として読み込む
_spec = importlib.util.spec_from_file_location("ex001_data", os.path.join(EX001_DIR, "data.py"))
ex001_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex001_data)
sys.modules["ex001_data"] = ex001_data
from machine_learning_utils import set_seed  # noqa: E402

DOMAIN_LABELS, FLAT_LABELS, FORMAT_LABELS = ex001_data.DOMAIN_LABELS, ex001_data.FLAT_LABELS, ex001_data.FORMAT_LABELS
PROJECT_DIR, image_path, to_flat_label = ex001_data.PROJECT_DIR, ex001_data.image_path, ex001_data.to_flat_label
load_odai_meta, make_contact_sheets = ex001_data.load_odai_meta, ex001_data.make_contact_sheets

DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex002_odai_type_distill")
EXTRA_LABEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_labels_extra.csv")
N_EXTRA_EVAL = 200
N_POOL = 40_000
N_VAL = 4_000

__all__ = ["DOMAIN_LABELS", "FLAT_LABELS", "FORMAT_LABELS", "image_path", "to_flat_label"]


def load_eval_labels() -> pl.DataFrame:
    """ex001 の評価用ラベル（440枚）と追加の評価用ラベル（200枚，pool は random2）を結合して読み込む．

    Returns:
        pl.DataFrame: 列 fileId, pool, format, domain, flat を持つ表（640行）．
    """
    base = ex001_data.load_eval_labels()
    extra = pl.read_csv(EXTRA_LABEL_PATH, comment_prefix="#").with_columns(
        pl.struct("format", "domain").map_elements(
            lambda r: to_flat_label(r["format"], r["domain"]), return_dtype=pl.String).alias("flat"))
    return pl.concat([base.select(extra.columns), extra])


def load_pool() -> pl.DataFrame:
    """教師ラベル用の画像の一覧（学習用と検証用の区別を含む）を読み込む．

    Returns:
        pl.DataFrame: 列 fileId (Int64), split (str: "train" または "val") を持つ表（40,000行）．
    """
    return pl.read_csv(os.path.join(DATASET_DIR, "pool.csv"))


def load_teacher_labels(teacher: str) -> pl.DataFrame:
    """教師の VLM が付けたラベルを，教師ラベル用の画像の一覧と結合して読み込む．

    Args:
        teacher (str): datasets/ex002_odai_type_distill/teacher_labels/ 内の CSV のファイル名（拡張子なし）．

    Returns:
        pl.DataFrame: 列 fileId, split, format, domain, flat を持つ表．
    """
    lab = pl.read_csv(os.path.join(DATASET_DIR, "teacher_labels", f"{teacher}.csv"))
    lab = lab.with_columns(pl.struct("format", "domain").map_elements(
        lambda r: to_flat_label(r["format"], r["domain"]), return_dtype=pl.String).alias("flat"))
    return load_pool().join(lab, on="fileId", how="inner", maintain_order="left")


class OdaiImageDataset(Dataset):
    """お題画像と2軸のラベルの番号を返すデータセット（画像エンコーダの微調整用）．"""

    def __init__(self, ids: list, fmt: list, dom: list, transform):
        """
        Args:
            ids (list[int]): お題IDのリスト（長さ N）．
            fmt (list[str]): 長さ N の形式のラベル．
            dom (list[str]): 長さ N の画風のラベル．
            transform (callable): PIL 画像を形状 (3, H, W) のテンソルに変換する関数．
        """
        self.ids = ids
        self.fmt = [FORMAT_LABELS.index(f) for f in fmt]
        self.dom = [DOMAIN_LABELS.index(d) for d in dom]
        self.transform = transform

    def __len__(self) -> int:
        """データ数を返す．

        Returns:
            int: 画像の数．
        """
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple:
        """i 番目の画像とラベルを返す．

        Args:
            i (int): インデックス．

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 形状 (3, H, W) の画像と，形状 (2,) の [形式の番号, 画風の番号]．
        """
        with Image.open(image_path(self.ids[i])) as im:
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor([self.fmt[i], self.dom[i]])


def load_dataloader(teacher: str, transforms: tuple, batch_size: int, n_train: int = None, seed: int = 0,
                    num_workers: int = 16) -> tuple:
    """教師ラベルによる学習用と，検証用（教師ラベルの検証用分割）のデータローダーを作る．

    学習用と検証用の分割は data.py の main で固定した pool.csv の split に従う（9:1）．

    Args:
        teacher (str): 教師ラベルの名前．
        transforms (tuple): (学習用の変換, 検証用の変換)．
        batch_size (int): バッチサイズ．
        n_train (int): 学習に用いる枚数．None の場合は全て用いる．
        seed (int): 乱数のシード．
        num_workers (int): ワーカー数．

    Returns:
        tuple[DataLoader, DataLoader]: 学習用と検証用のデータローダー．
    """
    set_seed(seed)
    lab = load_teacher_labels(teacher)
    tr, va = lab.filter(pl.col("split") == "train"), lab.filter(pl.col("split") == "val")
    if n_train is not None and n_train < len(tr):
        tr = tr.sample(n=n_train, seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    ds_tr = OdaiImageDataset(tr["fileId"].to_list(), tr["format"].to_list(), tr["domain"].to_list(), transforms[0])
    ds_va = OdaiImageDataset(va["fileId"].to_list(), va["format"].to_list(), va["domain"].to_list(), transforms[1])
    kw = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True, "persistent_workers": True}
    return DataLoader(ds_tr, shuffle=True, generator=g, drop_last=True, **kw), DataLoader(ds_va, shuffle=False, **kw)


def main() -> None:
    """追加の評価用画像と教師ラベル用の画像を抽出し，評価用画像のコンタクトシートを作る．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(DATASET_DIR, exist_ok=True)
    meta = load_odai_meta()
    used = set(ex001_data.load_eval_labels()["fileId"].to_list())
    weak_path = os.path.join(PROJECT_DIR, "datasets", "ex001_odai_image_type", "weak_train.csv")
    used |= set(pl.read_csv(weak_path)["fileId"].to_list())
    cand = meta.filter(~pl.col("fileId").is_in(list(used)))
    picked = cand.sample(n=N_EXTRA_EVAL + N_POOL, seed=args.seed + 100)["fileId"].to_list()
    extra, pool = picked[:N_EXTRA_EVAL], picked[N_EXTRA_EVAL:]
    pl.DataFrame({"fileId": extra, "pool": ["random2"] * len(extra)}).write_csv(
        os.path.join(DATASET_DIR, "eval_extra_candidates.csv"))
    make_contact_sheets(extra, os.path.join(DATASET_DIR, "contact_sheets"))
    tr, va = train_test_split(np.array(pool), test_size=N_VAL, random_state=args.seed)
    pl.DataFrame({"fileId": np.concatenate([tr, va]), "split": ["train"] * len(tr) + ["val"] * len(va)}).write_csv(
        os.path.join(DATASET_DIR, "pool.csv"))
    print(f"追加の評価用 {len(extra)} 枚，教師ラベル用 {len(pool)} 枚（学習 {len(tr)}，検証 {len(va)}）")


if __name__ == "__main__":
    main()
