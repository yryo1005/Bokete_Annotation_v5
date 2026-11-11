"""お題文ありを予測する生徒（画像埋め込みに対する MLP）．

パッケージの生徒の MLP（LayerNorm → 全結合 1,024 → GELU → Dropout → 出力層）と同じ構成で，出力はお題文ありの1つのロジットとする．
"""
import torch
from torch import nn


class OdaibunMLP(nn.Module):
    """お題文ありの2値分類の MLP．"""

    def __init__(self, d_in: int, hidden: int = 1024, dropout: float = 0.3):
        """層を作る．

        Args:
            d_in (int): 入力の次元（埋め込みは1,152．現在の生徒の確率を加える場合は1,159）．
            hidden (int): 中間層の次元．
            dropout (float): Dropout の確率．

        Returns:
            None
        """
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ロジットを求める．

        Args:
            x (torch.Tensor): 形状 (B, d_in) の入力．

        Returns:
            torch.Tensor: 形状 (B,) のロジット．
        """
        return self.net(x).squeeze(1)
