"""埋め込みモデルの処理速度を，実運用に近い条件（DataLoader の複数ワーカーで読み込みと前処理）で測定する．

train.py の処理速度は前処理を推論と同じスレッドで行うため CPU 律速となる．
本スクリプトでは，読み込みと前処理を num_workers 個のプロセスで並列に行い，GPU の推論と重ねた場合の速度を測る．
結果は outputs/ex001_odai_image_type/benchmark/{モデル}/bs{バッチサイズ}_w{ワーカー数}/0/meta.json に保存する．

実行例:
    .venv_annotate/bin/python programs/ex001_odai_image_type/benchmark.py
"""
import itertools
import json
import os
import sys
import time

import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, image_path  # noqa: E402
from train import EMBEDDING_MODELS, OUTPUT_DIR, TOTAL_IMAGES, short_name  # noqa: E402

BATCH_SIZES = [128]
NUM_WORKERS = [16]
N_IMAGES = 10_000


class ImageDataset(Dataset):
    """お題IDのリストから画像を読み込むデータセット．"""

    def __init__(self, ids: list):
        """
        Args:
            ids (list[int]): お題IDのリスト．
        """
        self.ids = ids

    def __len__(self) -> int:
        """データ数を返す．

        Returns:
            int: お題IDの数．
        """
        return len(self.ids)

    def __getitem__(self, i: int) -> Image.Image:
        """i 番目の画像を読み込む．

        Args:
            i (int): インデックス．

        Returns:
            PIL.Image.Image: RGB 画像．
        """
        with Image.open(image_path(self.ids[i])) as im:
            return im.convert("RGB")


class Collate:
    """ワーカープロセス内で画像のリストをモデルの入力テンソルに変換する collate 関数．"""

    def __init__(self, processor):
        """
        Args:
            processor: Hugging Face の画像の前処理器．
        """
        self.processor = processor

    def __call__(self, images: list) -> torch.Tensor:
        """画像のリストを前処理する．

        Args:
            images (list[PIL.Image.Image]): 長さ B の画像のリスト．

        Returns:
            torch.Tensor: 形状 (B, 3, H, W) の float16 の画素値．
        """
        return self.processor(images=images, return_tensors="pt")["pixel_values"].half()


def benchmark(model_name: str, ids: list, batch_size: int, num_workers: int) -> dict:
    """1つのモデルと条件で，読み込みから埋め込みの計算までの処理速度を測る．

    Args:
        model_name (str): Hugging Face Hub 上のモデル名．
        ids (list[int]): お題IDのリスト．
        batch_size (int): バッチサイズ．
        num_workers (int): DataLoader のワーカー数．

    Returns:
        dict: 画像数，経過秒数，1秒あたりの画像数，全画像の処理に要する時間の見積もり，最大の VRAM 使用量．
    """
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name).image_processor
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).cuda().eval()
    loader = DataLoader(ImageDataset(ids), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=Collate(processor), prefetch_factor=2, persistent_workers=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for pixel_values in tqdm(loader, desc=short_name(model_name), leave=False):
            model.get_image_features(pixel_values=pixel_values.cuda(non_blocking=True).half())
    torch.cuda.synchronize()
    sec = time.perf_counter() - start
    ips = len(ids) / sec
    return {"model": model_name, "batch_size": batch_size, "num_workers": num_workers, "n_images": len(ids),
            "elapsed_sec": sec, "images_per_sec": ips, "est_hours_all": TOTAL_IMAGES / ips / 3600,
            "max_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}


def main() -> None:
    """全ての埋め込みモデルと条件の組み合わせで処理速度を測定する．

    Returns:
        None
    """
    ids = pl.read_csv(os.path.join(DATASET_DIR, "weak_train.csv"))["fileId"].to_list()[:N_IMAGES]
    for model_name, bs, nw in itertools.product(EMBEDDING_MODELS, BATCH_SIZES, NUM_WORKERS):
        run_dir = os.path.join(OUTPUT_DIR, "benchmark", short_name(model_name), f"bs{bs}_w{nw}", "0")
        if os.path.exists(os.path.join(run_dir, "meta.json")):
            continue
        result = benchmark(model_name, ids, bs, nw)
        os.makedirs(run_dir, exist_ok=True)
        json.dump(result, open(os.path.join(run_dir, "meta.json"), "w"), indent=4)
        print(f"{short_name(model_name)} bs={bs} w={nw}: {result['images_per_sec']:.1f} img/s, "
              f"全件 {result['est_hours_all']:.1f} 時間, VRAM {result['max_vram_gb']:.1f} GB")


if __name__ == "__main__":
    main()
