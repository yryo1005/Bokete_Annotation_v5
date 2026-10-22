"""ex004（4軸のラベルによる蒸留）で用いるデータを扱うモジュール．

- 4軸のラベルの定義（項目の一覧）と，目視ラベルの読み込み
- 目視ラベル付与用のコンタクトシートの作成（以前の目視ラベルを表示し，VLM の予測は表示しない）
- 教師ラベルの読み込みと，学習用・検証用の分割（ex002 の pool.csv を共有する）

目視ラベルは eval_labels_v1.txt に1行1画像で記録する．書式は「お題ID 形式 属性 媒体 事前知識」であり，
形式は K（カルタ）/ A（穴埋め）/ O（お題文）/ N（通常），属性は b（空欄）p（お題文）k（カルタ記号）t（その他の文字）
c（合成）u（判定不能），媒体は P（写真）A（アニメ）G（ゲーム）C（CG）M（マンガ）T（テキストグラフィック）E（絵画）
I（イラスト），事前知識は p（著名人）w（特定の作品・キャラクター）x（作品不明）の文字の並びで表し，該当なしは "-" とする．
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import polars as pl
import torch
from PIL import Image, ImageDraw

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EX002_DIR = os.path.join(PROJECT_DIR, "programs", "ex002_odai_type_distill")
sys.path.insert(0, os.path.join(PROJECT_DIR, ".ai", "ai-dev-kit", "src"))
_spec = importlib.util.spec_from_file_location("ex002_data", os.path.join(EX002_DIR, "data.py"))
ex002_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex002_data)

DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex004_four_axis_distill")
LABEL_PATH = os.path.join(THIS_DIR, "eval_labels_v1.txt")
image_path, load_pool = ex002_data.image_path, ex002_data.load_pool
load_old_eval_labels = ex002_data.load_eval_labels

FORMATS = ["karuta", "anaume", "odaibun", "normal"]
ATTRIBUTES = ["has_blank", "has_prompt_text", "has_karuta_mark", "has_other_text", "composite"]
MEDIA = ["photo", "anime", "game", "cg", "manga", "text_graphic", "painting", "illustration"]
KNOWLEDGE = ["known_person", "known_work", "work_unknown"]
FORMAT_CODE = {"K": "karuta", "A": "anaume", "O": "odaibun", "N": "normal"}
ATTR_CODE = {"b": "has_blank", "p": "has_prompt_text", "k": "has_karuta_mark", "t": "has_other_text",
             "c": "composite", "u": "unsure"}
MEDIA_CODE = {"P": "photo", "A": "anime", "G": "game", "C": "cg", "M": "manga", "T": "text_graphic",
              "E": "painting", "I": "illustration"}
KNOW_CODE = {"p": "known_person", "w": "known_work", "x": "work_unknown"}
# 形式から必ず真になる属性
FORMAT_IMPLIES = {"karuta": "has_karuta_mark", "anaume": "has_blank", "odaibun": "has_prompt_text"}


def parse_label_line(line: str) -> dict:
    """目視ラベルの1行を解釈する．

    Args:
        line (str): 「お題ID 形式 属性 媒体 事前知識」の行．

    Returns:
        dict: キー fileId，format，ATTRIBUTES・MEDIA・KNOWLEDGE の各項目（bool），unsure（bool）を持つ辞書．
    """
    fid, f, a, m, k = line.split()
    rec = {"fileId": int(fid), "format": FORMAT_CODE[f]}
    attrs = {ATTR_CODE[c] for c in a if c != "-"}
    if rec["format"] in FORMAT_IMPLIES:
        attrs.add(FORMAT_IMPLIES[rec["format"]])
    for name in ATTRIBUTES + ["unsure"]:
        rec[name] = name in attrs
    media = {MEDIA_CODE[c] for c in m if c != "-"}
    for name in MEDIA:
        rec[name] = name in media
    know = {KNOW_CODE[c] for c in k if c != "-"}
    for name in KNOWLEDGE:
        rec[name] = name in know
    return rec


def load_eval_labels() -> pl.DataFrame:
    """確定版の定義による目視ラベル（640枚）を，以前のラベルの pool の列と結合して読み込む．

    Returns:
        pl.DataFrame: 列 fileId, pool, format と，属性・媒体・事前知識の各項目（bool），unsure を持つ表．
    """
    lines = [l for l in open(LABEL_PATH, encoding="utf-8") if l.strip() and not l.startswith("#")]
    df = pl.DataFrame([parse_label_line(l) for l in lines])
    old = load_old_eval_labels().select("fileId", "pool")
    out = old.join(df, on="fileId", how="inner", maintain_order="left")
    assert len(out) == len(old), "目視ラベルが全ての評価用画像に付いていない"
    return out


def make_label_sheets(out_dir: str, cols: int = 5, rows: int = 4, tile: int = 320) -> list:
    """目視ラベル付与用のコンタクトシートを作る．各タイルに番号，お題ID，以前の目視ラベルを表示する．

    Args:
        out_dir (str): 出力先のディレクトリ．
        cols (int): 1シートの列数．
        rows (int): 1シートの行数．
        tile (int): 1タイルの一辺の画素数．

    Returns:
        list[str]: 作成したシートのパスのリスト．
    """
    os.makedirs(out_dir, exist_ok=True)
    old = load_old_eval_labels()
    font = ex002_data.ex001_data._load_font(18)
    per = cols * rows
    paths = []
    recs = old.to_dicts()
    for s in range(0, len(recs), per):
        sheet = Image.new("RGB", (cols * tile, rows * tile), "white")
        draw = ImageDraw.Draw(sheet)
        for k, r in enumerate(recs[s:s + per]):
            x, y = (k % cols) * tile, (k // cols) * tile
            sheet.paste(ex002_data.ex001_data._fit_tile(image_path(r["fileId"]), tile - 6), (x + 3, y + 3))
            draw.rectangle([x + 3, y + 3, x + 230, y + 26], fill="black")
            draw.text((x + 5, y + 4), f"{k:02d}:{r['fileId']} {r['format'][:2]}/{r['domain'][:2]}", fill="yellow",
                      font=font)
        path = os.path.join(out_dir, f"sheet_{s // per:03d}.jpg")
        sheet.save(path, quality=90)
        paths.append(path)
    return paths


def load_teacher_labels(name: str) -> pl.DataFrame:
    """教師ラベル（4軸）を教師ラベル用の画像の一覧と結合して読み込む．

    Args:
        name (str): datasets/ex004_four_axis_distill/teacher_labels/ 内の CSV のファイル名（拡張子なし）．

    Returns:
        pl.DataFrame: 列 fileId, split, format と，各項目（bool）を持つ表．
    """
    lab = pl.read_csv(os.path.join(DATASET_DIR, "teacher_labels", f"{name}.csv"))
    return load_pool().join(lab, on="fileId", how="inner", maintain_order="left")


def label_arrays(df: pl.DataFrame) -> dict:
    """ラベルの表を学習用の配列に変換する．

    Args:
        df (pl.DataFrame): 列 format と各項目（bool）を持つ表（行数 N）．

    Returns:
        dict: キー format（形状 (N,) の番号），attr（形状 (N, 5)），media（形状 (N, 8)），know（形状 (N, 3)）の配列．
    """
    return {"format": np.array([FORMATS.index(f) for f in df["format"]]),
            "attr": df.select(ATTRIBUTES).to_numpy().astype(np.float32),
            "media": df.select(MEDIA).to_numpy().astype(np.float32),
            "know": df.select(KNOWLEDGE).to_numpy().astype(np.float32)}


def main() -> None:
    """目視ラベル付与用のコンタクトシートを作る．

    Returns:
        None
    """
    argparse.ArgumentParser().parse_args()
    paths = make_label_sheets(os.path.join(DATASET_DIR, "label_sheets"))
    print(f"{len(paths)} 枚のシートを作成した")


if __name__ == "__main__":
    main()


class FourAxisDataset(torch.utils.data.Dataset):
    """お題画像と4軸のラベルを返すデータセット（画像エンコーダの微調整用）．"""

    def __init__(self, ids: list, y: dict, transform):
        """
        Args:
            ids (list[int]): お題IDのリスト（長さ N）．
            y (dict): label_arrays の辞書．
            transform (callable): PIL 画像を形状 (3, H, W) のテンソルに変換する関数．
        """
        self.ids, self.y, self.transform = ids, y, transform

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
            tuple[torch.Tensor, dict]: 形状 (3, H, W) の画像と，キー format（スカラーの番号）・attr・media・know
            （0/1 のベクトル）のラベル．
        """
        with Image.open(image_path(self.ids[i])) as im:
            x = self.transform(im.convert("RGB"))
        return x, {"format": int(self.y["format"][i]), "attr": torch.tensor(self.y["attr"][i]),
                   "media": torch.tensor(self.y["media"][i]), "know": torch.tensor(self.y["know"][i])}


def load_dataloader(teacher: str, transforms: tuple, batch_size: int, seed: int = 0, num_workers: int = 16) -> tuple:
    """教師ラベルによる学習用と検証用のデータローダーを作る（分割は pool.csv の split に従う．9:1）．

    Args:
        teacher (str): 教師ラベルの名前．
        transforms (tuple): (学習用の変換, 検証用の変換)．
        batch_size (int): バッチサイズ．
        seed (int): 乱数のシード．
        num_workers (int): ワーカー数．

    Returns:
        tuple[DataLoader, DataLoader]: 学習用と検証用のデータローダー．
    """
    from machine_learning_utils import set_seed

    set_seed(seed)
    lab = load_teacher_labels(teacher)
    tr, va = lab.filter(pl.col("split") == "train"), lab.filter(pl.col("split") == "val")
    g = torch.Generator()
    g.manual_seed(seed)
    ds_tr = FourAxisDataset(tr["fileId"].to_list(), label_arrays(tr), transforms[0])
    ds_va = FourAxisDataset(va["fileId"].to_list(), label_arrays(va), transforms[1])
    kw = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True, "persistent_workers": True}
    return (torch.utils.data.DataLoader(ds_tr, shuffle=True, generator=g, drop_last=True, **kw),
            torch.utils.data.DataLoader(ds_va, shuffle=False, **kw))
