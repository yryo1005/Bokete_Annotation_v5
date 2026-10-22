"""4軸の出力を持つ画像エンコーダ（SigLIP2-B）の微調整用のモデルを定義するモジュール．"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import ATTRIBUTES, FORMATS, KNOWLEDGE, MEDIA  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

HEAD_SIZES = {"format": len(FORMATS), "attr": len(ATTRIBUTES), "media": len(MEDIA), "know": len(KNOWLEDGE)}


class SigLIP2BaseFourHead(torch.nn.Module):
    """SigLIP2-B の画像エンコーダのプーリング出力に，形式・属性・媒体・事前知識の4つの線形の出力層を付けたモデル．"""

    hf_name = "google/siglip2-base-patch16-224"

    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(self.hf_name).vision_model
        d = self.backbone.config.hidden_size
        self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(d, n) for k, n in HEAD_SIZES.items()})

    def forward(self, x: torch.Tensor) -> dict:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, 3, 224, 224) の画像．

        Returns:
            dict[str, torch.Tensor]: 出力層ごとのロジット．format は形状 (B, 4)，attr は (B, 5)，media は (B, 8)，
            know は (B, 3)．
        """
        h = self.backbone(pixel_values=x).pooler_output
        return {k: m(h) for k, m in self.heads.items()}

    @staticmethod
    def transforms() -> tuple:
        """学習用と検証用の画像の変換を返す（SigLIP 2 の正規化は平均0.5，標準偏差0.5）．

        Returns:
            tuple: (学習用の変換, 検証用の変換)．学習用は文字の向きを保つため左右反転を用いず，色の揺らぎのみとする．
        """
        from torchvision import transforms as T

        norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        train = T.Compose([T.Resize((224, 224)), T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), norm])
        val = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
        return train, val


MODEL_CLASSES = {"siglip2_base": SigLIP2BaseFourHead}


def load_model(ModelClass, weight_path: str = None, seed: int = 0) -> torch.nn.Module:
    """モデルをインスタンス化し，必要に応じて重みを読み込む．

    Args:
        ModelClass (type[torch.nn.Module]): モデルのクラス．
        weight_path (str): 重みのファイルのパス．None の場合は事前学習の重みのみを用いる．
        seed (int): 出力層の初期値を固定する乱数のシード．

    Returns:
        torch.nn.Module: モデル．
    """
    set_seed(seed)
    model = ModelClass()
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return model
