"""小さな VLM を分類器にするモデルの定義．

VLM（Hugging Face の AutoModelForImageTextToText）の言語モデルの最後の層の，入力の最後のトークン（生成の直前の位置）の隠れ状態に，
発話の型6区分と画像外の知識の2つの出力層を付ける．VLM には LoRA（全ての線形層）を付けて微調整する．
生成を行わず1回の順伝播で判定するため，生成による判定より速い．

お題画像は長辺 448 画素以下に縮小したうえで，モデルごとに画像の分割（タイル）を止め，画像のトークン数を200〜300程度にそろえる．
"""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import VOICES  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

# モデル名 → Hugging Face の ID
MODELS = {
    "q2": "Qwen/Qwen3-VL-2B-Instruct",
    "q4": "Qwen/Qwen3-VL-4B-Instruct",
    "q8": "Qwen/Qwen3-VL-8B-Instruct",
    "iv1": "OpenGVLab/InternVL3_5-1B-HF",
    "iv2": "OpenGVLab/InternVL3_5-2B-HF",
    "iv4": "OpenGVLab/InternVL3_5-4B-HF",
    "smol2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "lfm16": "LiquidAI/LFM2-VL-1.6B",
    "lfm3": "LiquidAI/LFM2-VL-3B",
    "gemma4": "unsloth/gemma-3-4b-it",
}
FORMAT_JA = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常"}
IMAGE_SIDE = 448


def load_processor(key: str):
    """モデルの processor を読み込み，画像の分割を止めて画像のトークン数をそろえる．

    Args:
        key (str): MODELS のキー．

    Returns:
        transformers.ProcessorMixin: processor（左詰めの padding）．
    """
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODELS[key])
    ip = proc.image_processor
    if key.startswith("q"):
        ip.size = {"shortest_edge": 32 * 32, "longest_edge": IMAGE_SIDE * IMAGE_SIDE}
    if hasattr(ip, "crop_to_patches"):
        ip.crop_to_patches = False
    if hasattr(ip, "do_image_splitting"):
        ip.do_image_splitting = False
    if key.startswith("smol"):
        ip.size = {"longest_edge": 384}
        ip.max_image_size = {"longest_edge": 384}
    if key.startswith("lfm"):
        ip.max_image_tokens = 256
    if hasattr(ip, "do_pan_and_scan"):
        ip.do_pan_and_scan = False
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    return proc


def build_prompt(fmt: str, text: str) -> str:
    """分類器への入力の文を作る（判定の指示は微調整で学ぶため短くする）．

    Args:
        fmt (str): お題の形式．
        text (str): ボケの文．

    Returns:
        str: 入力の文．
    """
    return f"大喜利サイト「ボケて」のお題画像（形式：{FORMAT_JA.get(fmt, '通常')}）に投稿されたボケ：「{(text or '').replace(chr(10), ' ')[:200]}」"


def encode(proc, images: list, prompts: list) -> dict:
    """画像と文の組をモデルの入力にする．

    Args:
        proc (transformers.ProcessorMixin): load_processor の結果．
        images (list[PIL.Image.Image]): お題画像．
        prompts (list[str]): build_prompt の結果．

    Returns:
        dict[str, torch.Tensor]: モデルの入力（input_ids, attention_mask, pixel_values など）．
    """
    texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                                      add_generation_prompt=True, tokenize=False) for p in prompts]
    extra = {"crop_to_patches": False} if "InternVL" in type(proc).__name__ else {}
    return dict(proc(text=texts, images=[[im] for im in images], return_tensors="pt", padding=True, **extra))


def small_image(path: str) -> Image.Image:
    """画像を RGB で読み込み，長辺を IMAGE_SIDE 画素以下にする（極端に細い画像は余白を足す）．

    Args:
        path (str): 画像のパス．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
    im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
    if min(im.size) < 32:
        canvas = Image.new("RGB", (max(im.width, 32), max(im.height, 32)), "white")
        canvas.paste(im, (0, 0))
        im = canvas
    return im


class VLMClassifier(torch.nn.Module):
    """VLM の最後のトークンの隠れ状態に，発話の型と画像外の知識の出力層を付けた分類器．"""

    def __init__(self, key: str, lora_r: int = 16, gradient_checkpointing: bool = True):
        """
        Args:
            key (str): MODELS のキー．
            lora_r (int): LoRA のランク（0 で VLM を固定し出力層のみ学習）．
            gradient_checkpointing (bool): 勾配のチェックポイントを使うか．
        """
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText

        vlm = AutoModelForImageTextToText.from_pretrained(MODELS[key], dtype=torch.bfloat16, attn_implementation="sdpa")
        hidden = vlm.config.get_text_config().hidden_size
        backbone = vlm.model
        if gradient_checkpointing:
            vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.config.use_cache = False
        if lora_r > 0:
            backbone = get_peft_model(backbone, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules="all-linear"))
        else:
            backbone.requires_grad_(False)
        self.backbone = backbone
        self.voice_head = torch.nn.Linear(hidden, len(VOICES))
        self.outside_head = torch.nn.Linear(hidden, 1)

    def forward(self, inputs: dict) -> tuple:
        """順伝播．

        Args:
            inputs (dict[str, torch.Tensor]): encode の結果（左詰めの padding）．

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 発話の型のロジット (B, 6) と画像外の知識のロジット (B,)．
        """
        out = self.backbone(**inputs, use_cache=False)
        h = out.last_hidden_state[:, -1].float()
        return self.voice_head(h), self.outside_head(h)[:, 0]

    def trainable_state_dict(self) -> dict:
        """学習したパラメータ（LoRA と出力層）だけの state_dict を返す．

        Returns:
            dict[str, torch.Tensor]: パラメータの辞書．
        """
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}


def load_model(key: str, weight_path: str = None, seed: int = 0, **kwargs) -> VLMClassifier:
    """分類器をインスタンス化する．

    Args:
        key (str): MODELS のキー．
        weight_path (str): 学習したパラメータ（trainable_state_dict）のパス（None で初期値）．
        seed (int): 乱数のシード．
        **kwargs: VLMClassifier の引数．

    Returns:
        VLMClassifier: 分類器．
    """
    set_seed(seed)
    # VLM の一部の演算（画像の位置埋め込みなど）は決定論的な実装を持たないため，警告にとどめる
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = VLMClassifier(key, **kwargs)
    if weight_path is not None:
        missing = model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    return model
