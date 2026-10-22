"""画像エンコーダを微調整する分類器のモデルを定義するモジュール．

形式（4クラス）と画風（4クラス）の2つの出力層を持つ．エンコーダは以下の2種類を比較する．
- ConvNeXt-Tiny（timm，ImageNet-12k で事前学習し ImageNet-1k で微調整したもの）
- SigLIP2-B の画像エンコーダ（google/siglip2-base-patch16-224）
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from machine_learning_utils import set_seed  # noqa: E402

N_FORMAT, N_DOMAIN = 4, 4


class ConvNeXtTinyTwoHead(torch.nn.Module):
    """ConvNeXt-Tiny の特徴に2つの線形の出力層を付けたモデル．入力は 224×224．"""

    timm_name = "convnext_tiny.in12k_ft_in1k"

    def __init__(self):
        super().__init__()
        import timm

        self.backbone = timm.create_model(self.timm_name, pretrained=True, num_classes=0)
        d = self.backbone.num_features
        self.head_f = torch.nn.Linear(d, N_FORMAT)
        self.head_d = torch.nn.Linear(d, N_DOMAIN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, 3, 224, 224) の画像．

        Returns:
            torch.Tensor: 形状 (B, 8) のロジット．前半4列が形式，後半4列が画風．
        """
        h = self.backbone(x)
        return torch.cat([self.head_f(h), self.head_d(h)], dim=1)

    @staticmethod
    def transforms() -> tuple:
        """学習用と検証用の画像の変換を返す．

        Returns:
            tuple: (学習用の変換, 検証用の変換)．学習用は左右反転を除き，文字の向きを保つ弱い拡張のみとする．
        """
        from torchvision import transforms as T

        norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        train = T.Compose([T.Resize((224, 224)), T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), norm])
        val = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
        return train, val


class SigLIP2BaseTwoHead(torch.nn.Module):
    """SigLIP2-B の画像エンコーダのプーリング出力に2つの線形の出力層を付けたモデル．入力は 224×224．"""

    hf_name = "google/siglip2-base-patch16-224"

    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(self.hf_name).vision_model
        d = self.backbone.config.hidden_size
        self.head_f = torch.nn.Linear(d, N_FORMAT)
        self.head_d = torch.nn.Linear(d, N_DOMAIN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, 3, 224, 224) の画像．

        Returns:
            torch.Tensor: 形状 (B, 8) のロジット．前半4列が形式，後半4列が画風．
        """
        h = self.backbone(pixel_values=x).pooler_output
        return torch.cat([self.head_f(h), self.head_d(h)], dim=1)

    @staticmethod
    def transforms() -> tuple:
        """学習用と検証用の画像の変換を返す（SigLIP 2 の正規化は平均0.5，標準偏差0.5）．

        Returns:
            tuple: (学習用の変換, 検証用の変換)．
        """
        from torchvision import transforms as T

        norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        train = T.Compose([T.Resize((224, 224)), T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), norm])
        val = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
        return train, val


MODEL_CLASSES = {"convnext_tiny": ConvNeXtTinyTwoHead, "siglip2_base": SigLIP2BaseTwoHead}


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
