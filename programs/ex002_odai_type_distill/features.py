"""生徒の分類器の入力となる特徴量を計算して保存する．

- 画像埋め込み：CLIP-B/32，SigLIP2-B，SigLIP2-so400m の画像特徴（L2 正規化，float16）
- 空欄の矩形の特徴：OpenCV で輪郭を抽出し，内部が一様な四角形（穴埋めの空欄の候補）を数える手作りの特徴

保存先は datasets/ex002_odai_type_distill/features/{特徴名}/{集合名}.npy であり，
行の順は集合名 "pool" が pool.csv，"eval" が load_eval_labels の順である．

実行例:
    .venv_annotate/bin/python programs/ex002_odai_type_distill/features.py --kinds embedding blankbox
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, image_path, load_eval_labels, load_pool  # noqa: E402

FEATURE_DIR = os.path.join(DATASET_DIR, "features")
EMBEDDING_MODELS = ["openai/clip-vit-base-patch32", "google/siglip2-base-patch16-224",
                    "google/siglip2-so400m-patch14-384"]
BLANKBOX_DIM = 8


class _PathDataset(Dataset):
    """お題IDから画像を読み込み，前処理器で画素値に変換するデータセット．"""

    def __init__(self, ids: list, processor):
        """
        Args:
            ids (list[int]): お題IDのリスト．
            processor: Hugging Face の画像の前処理器．
        """
        self.ids, self.processor = ids, processor

    def __len__(self) -> int:
        """データ数を返す．

        Returns:
            int: 画像の数．
        """
        return len(self.ids)

    def __getitem__(self, i: int) -> torch.Tensor:
        """i 番目の画像を前処理する．

        Args:
            i (int): インデックス．

        Returns:
            torch.Tensor: 形状 (3, H, W) の float16 の画素値．
        """
        with Image.open(image_path(self.ids[i])) as im:
            return self.processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0].half()


def extract_embeddings(model_name: str, ids: list, batch_size: int = 128, num_workers: int = 16) -> np.ndarray:
    """画像の埋め込みを計算する．

    Args:
        model_name (str): Hugging Face Hub 上のモデル名．
        ids (list[int]): お題IDのリスト（長さ N）．
        batch_size (int): バッチサイズ．
        num_workers (int): DataLoader のワーカー数．

    Returns:
        np.ndarray: 形状 (N, D) の float16 の配列（L2 正規化済み）．
    """
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name).image_processor
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).cuda().eval()
    loader = DataLoader(_PathDataset(ids, processor), batch_size=batch_size, num_workers=num_workers, prefetch_factor=2)
    out = []
    with torch.no_grad():
        for x in tqdm(loader, desc=model_name.split("/")[-1], leave=False):
            f = model.get_image_features(pixel_values=x.cuda(non_blocking=True))
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).half().cpu().numpy())
    return np.concatenate(out)


def blankbox_features(odai_id: int) -> np.ndarray:
    """OpenCV で穴埋めの空欄の候補となる四角形を検出し，その量を特徴量にする．

    二値化した画像の輪郭を多角形で近似し，4頂点で凸，面積が画像の0.1%〜40%，
    内部の画素の標準偏差が小さい（一様に塗られた）ものを空欄の候補とする．

    Args:
        odai_id (int): お題ID．

    Returns:
        np.ndarray: 形状 (8,) の float32 の配列．[候補の数, 白い候補の数, 候補の面積比の最大値, 合計,
        白い候補の面積比の最大値, 合計, 縦長の候補の数, 輪郭の総数の対数]．
    """
    img = cv2.imread(image_path(odai_id), cv2.IMREAD_GRAYSCALE)
    if img is None or min(img.shape) < 16:
        return np.zeros(BLANKBOX_DIM, dtype=np.float32)
    h, w = img.shape
    area_img = float(h * w)
    edges = cv2.Canny(img, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas, white_areas, n_tall = [], [], 0
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        ratio = bw * bh / area_img
        if not (0.001 <= ratio <= 0.4) or bw < 8 or bh < 8:
            continue
        inner = img[y + bh // 6: y + bh - bh // 6, x + bw // 6: x + bw - bw // 6]
        if inner.size == 0 or inner.std() > 12:
            continue
        areas.append(ratio)
        if inner.mean() > 200:
            white_areas.append(ratio)
        n_tall += int(bh > 1.8 * bw)
    a, wa = np.array(areas or [0.0]), np.array(white_areas or [0.0])
    return np.array([len(areas), len(white_areas), a.max(), a.sum(), wa.max(), wa.sum(), n_tall,
                     np.log1p(len(contours))], dtype=np.float32)


def compute_blankbox(ids: list, processes: int = 48) -> tuple:
    """空欄の矩形の特徴量をマルチプロセスで計算する．

    Args:
        ids (list[int]): お題IDのリスト（長さ N）．
        processes (int): プロセス数．

    Returns:
        tuple[np.ndarray, float]: 形状 (N, 8) の特徴量と，経過秒数．
    """
    start = time.perf_counter()
    with Pool(processes) as p:
        feats = list(tqdm(p.imap(blankbox_features, ids, chunksize=64), total=len(ids), desc="blankbox", leave=False))
    return np.stack(feats), time.perf_counter() - start


def save(name: str, split: str, arr: np.ndarray) -> None:
    """特徴量を保存する．

    Args:
        name (str): 特徴名．
        split (str): 集合名（"pool" または "eval"）．
        arr (np.ndarray): 特徴量．

    Returns:
        None
    """
    d = os.path.join(FEATURE_DIR, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{split}.npy"), arr)


def load_features(name: str, split: str) -> np.ndarray:
    """保存した特徴量を float32 で読み込む．

    Args:
        name (str): 特徴名（埋め込みはモデル名の末尾，例 "siglip2-so400m-patch14-384"，または "blankbox"）．
        split (str): 集合名（"pool" または "eval"）．

    Returns:
        np.ndarray: 形状 (N, D) の float32 の配列．
    """
    return np.load(os.path.join(FEATURE_DIR, name, f"{split}.npy")).astype(np.float32)


def main() -> None:
    """指定した種類の特徴量を教師ラベル用の画像と評価用の画像について計算する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", default=["embedding", "blankbox"])
    args = parser.parse_args()
    sets = {"pool": load_pool()["fileId"].to_list(), "eval": load_eval_labels()["fileId"].to_list()}
    if "blankbox" in args.kinds:
        for split, ids in sets.items():
            if os.path.exists(os.path.join(FEATURE_DIR, "blankbox", f"{split}.npy")):
                continue
            arr, sec = compute_blankbox(ids)
            save("blankbox", split, arr)
            print(f"blankbox {split}: {len(ids) / sec:.1f} 枚/秒")
    for model_name in EMBEDDING_MODELS if "embedding" in args.kinds else []:
        for split, ids in sets.items():
            name = model_name.split("/")[-1]
            if os.path.exists(os.path.join(FEATURE_DIR, name, f"{split}.npy")):
                continue
            save(name, split, extract_embeddings(model_name, ids))
            print(f"{name} {split}: 完了")


if __name__ == "__main__":
    main()
