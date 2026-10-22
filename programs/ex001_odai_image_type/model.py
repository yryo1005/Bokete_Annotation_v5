"""お題画像の種別判定に用いるモデルを定義するモジュール．

比較する手法は以下の3系統である．
- 画像とテキストの埋め込みモデル（CLIP，SigLIP 2）によるゼロショット分類
- 埋め込みモデルの画像特徴量に対するロジスティック回帰（線形プローブ）
- 視覚言語モデル（Vision Language Model，VLM）である Qwen3-VL への多肢選択の質問
いずれの手法も，形式（normal/text/anaume/karuta）と画風（photo/anime/manga/illust）の2軸を出力する．
"""
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from data import DOMAIN_LABELS, FLAT_LABELS, FORMAT_LABELS

# ゼロショット分類のプロンプト．各クラスの複数のプロンプトの埋め込みを平均して用いる
FORMAT_PROMPTS = {
    "normal": ["a photo", "a picture without any text", "an image of a scene"],
    "text": ["an image of Japanese text written on a plain background", "a slide with a written question in Japanese",
             "a text-only image"],
    "anaume": ["an image with an empty white blank box to fill in", "a fill-in-the-blank puzzle with an empty rectangle",
               "an image with an empty speech bubble"],
    "karuta": ["a Japanese karuta card with a hiragana character in a circle",
               "an image with a single hiragana letter inside a circle in the corner", "a karuta playing card"],
}
DOMAIN_PROMPTS = {
    "photo": ["a real life photograph", "a photo of real people", "a screenshot of a live-action TV show"],
    "anime": ["an anime screenshot", "a frame from a Japanese animated TV series", "a cartoon animation still"],
    "manga": ["a black and white manga panel with speech bubbles", "a page of a Japanese comic book",
              "a manga drawing with screentone"],
    "illust": ["a digital illustration", "a video game screenshot", "a painting", "a simple graphic design"],
}

_VLM_PROMPT_V1 = """This image is a prompt image ("odai") from the Japanese comedy site Bokete. Classify it on two axes.

Q1 "format":
- "karuta": a single hiragana character is written inside a circle or square (like a karuta card). Users write a line starting with that hiragana.
- "anaume": there are blank placeholders to fill in, such as empty white boxes, empty speech bubbles, ○○, □, or ？ replacing words.
- "text": the image is mainly written text that poses a topic or question (no blank placeholder).
- "normal": none of the above (incidental text such as subtitles or signs does not count).
Q2 "domain" (the visual style of the main picture):
- "photo": real-world photograph or live-action video frame (including TV, movies, photographed toys).
- "anime": screenshot of an animated TV series or film.
- "manga": panel(s) of a manga or comic (line drawing, screentone, speech bubbles).
- "illust": other illustrations, CG, game screens, paintings, or plain text graphics.

Answer with JSON only: {"format": "...", "domain": "..."}"""
_VLM_PROMPT_V2 = """This image is a prompt image ("odai") from the Japanese comedy site Bokete. Classify it on two axes.

Q1 "format":
- "karuta": a single hiragana character is written inside a circle or square (like a karuta card). Users write a line starting with that hiragana.
- "anaume": there are blank placeholders to fill in, such as empty white boxes, empty speech bubbles, ○○, □, or ？ replacing words.
- "text": the image consists almost entirely of written text (typically a topic or question on a plain or colored background) and has no main photo, anime or manga picture.
- "normal": an ordinary photo, anime, manga or illustration. Pictures that contain captions, subtitles, signs, titles or speech bubbles with words are still "normal".
Q2 "domain" (the visual style of the main picture):
- "photo": real-world photograph or live-action video frame (including TV, movies, photographed toys).
- "anime": screenshot of an animated TV series or film.
- "manga": panel(s) of a manga or comic (line drawing, screentone, speech bubbles).
- "illust": other illustrations, CG, game screens, paintings, or plain text graphics.

Answer with JSON only: {"format": "...", "domain": "..."}"""

# VLM のプロンプトの版．v2 は v1 で多かった「字幕などを含む画像を text とする誤り」を抑えるよう text の定義を厳しくした
VLM_PROMPTS = {"v1": _VLM_PROMPT_V1, "v2": _VLM_PROMPT_V2}


def get_device() -> str:
    """利用できるデバイスを返す．

    Returns:
        str: "cuda" または "cpu"．
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    """Hugging Face の CLIP 系モデルで画像とテキストを L2 正規化した埋め込みに変換する．"""

    def __init__(self, model_name: str, device: str = None):
        """
        Args:
            model_name (str): Hugging Face Hub 上のモデル名（例: "openai/clip-vit-base-patch32"）．
            device (str): 推論に用いるデバイス．None の場合は自動で選ぶ．
        """
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = device or get_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to(self.device).eval()
        self.is_siglip = "siglip" in model_name.lower()

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        """get_*_features の戻り値を特徴量のテンソルに揃える．

        Args:
            out (torch.Tensor | ModelOutput): get_image_features または get_text_features の戻り値．

        Returns:
            torch.Tensor: 形状 (N, D) の特徴量．
        """
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        """画像のリストを埋め込みに変換する．

        Args:
            images (list[PIL.Image.Image]): RGB 画像のリスト（長さ N）．

        Returns:
            np.ndarray: 形状 (N, D) の float32 配列．各行は L2 正規化されている．
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        feats = self._as_tensor(self.model.get_image_features(**inputs)).float()
        return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: list) -> np.ndarray:
        """テキストのリストを埋め込みに変換する．

        Args:
            texts (list[str]): テキストのリスト（長さ M）．

        Returns:
            np.ndarray: 形状 (M, D) の float32 配列．各行は L2 正規化されている．
        """
        kwargs = {"padding": "max_length", "max_length": 64} if self.is_siglip else {"padding": True}
        inputs = self.processor(text=texts, return_tensors="pt", truncation=True, **kwargs).to(self.device)
        feats = self._as_tensor(self.model.get_text_features(**inputs)).float()
        return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()


class ZeroShotClassifier:
    """クラスごとのプロンプトの埋め込みとのコサイン類似度で分類するゼロショット分類器．"""

    def __init__(self, embedder: EmbeddingModel, class_prompts: dict):
        """
        Args:
            embedder (EmbeddingModel): テキストの埋め込みに用いるモデル．
            class_prompts (dict[str, list[str]]): クラス名をキー，プロンプトのリストを値とする辞書．
        """
        self.classes = list(class_prompts.keys())
        protos = []
        for name in self.classes:
            emb = embedder.encode_texts(class_prompts[name]).mean(axis=0)
            protos.append(emb / np.linalg.norm(emb))
        self.prototypes = np.stack(protos)  # (C, D)

    def predict_proba(self, image_emb: np.ndarray) -> np.ndarray:
        """画像の埋め込みからクラスの確率を求める．

        Args:
            image_emb (np.ndarray): 形状 (N, D) の画像の埋め込み．

        Returns:
            np.ndarray: 形状 (N, C) の確率．温度 0.01 のソフトマックスで求める．
        """
        logits = image_emb @ self.prototypes.T / 0.01
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, image_emb: np.ndarray) -> list:
        """画像の埋め込みからクラスを予測する．

        Args:
            image_emb (np.ndarray): 形状 (N, D) の画像の埋め込み．

        Returns:
            list[str]: 長さ N の予測クラス名のリスト．
        """
        return [self.classes[i] for i in self.predict_proba(image_emb).argmax(axis=1)]


class LinearProbe:
    """画像の埋め込みに対するロジスティック回帰の分類器．"""

    def __init__(self, C: float = 1.0, seed: int = 0):
        """
        Args:
            C (float): L2 正則化の強さの逆数．
            seed (int): 乱数のシード．
        """
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=seed))

    def fit(self, emb: np.ndarray, labels: list) -> "LinearProbe":
        """分類器を学習する．

        Args:
            emb (np.ndarray): 形状 (N, D) の画像の埋め込み．
            labels (list[str]): 長さ N のラベル．

        Returns:
            LinearProbe: 学習済みの自身．
        """
        self.clf.fit(emb, labels)
        return self

    def predict(self, emb: np.ndarray) -> list:
        """クラスを予測する．

        Args:
            emb (np.ndarray): 形状 (N, D) の画像の埋め込み．

        Returns:
            list[str]: 長さ N の予測クラス名のリスト．
        """
        return list(self.clf.predict(emb))

    def predict_proba(self, emb: np.ndarray) -> np.ndarray:
        """クラスの確率を求める．

        Args:
            emb (np.ndarray): 形状 (N, D) の画像の埋め込み．

        Returns:
            np.ndarray: 形状 (N, C) の確率．列の順は self.classes に従う．
        """
        return self.clf.predict_proba(emb)

    @property
    def classes(self) -> list:
        """学習したクラス名のリスト．

        Returns:
            list[str]: クラス名のリスト．
        """
        return list(self.clf.classes_)


class VLMClassifier:
    """vLLM 上の Qwen3-VL に2軸の種別を JSON で回答させる分類器．"""

    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.85, max_pixels: int = 448 * 448,
                 prompt_version: str = "v2"):
        """
        Args:
            model_name (str): Hugging Face Hub 上のモデル名（例: "Qwen/Qwen3-VL-8B-Instruct"）．
            gpu_memory_utilization (float): vLLM が確保する VRAM の割合．
            max_pixels (int): 画像を入力する際の最大画素数．処理速度に直接影響する．
            prompt_version (str): VLM_PROMPTS のキー．
        """
        self.prompt = VLM_PROMPTS[prompt_version]
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # vLLM の JIT コンパイルが仮想環境内の ninja を見つけられるよう，仮想環境の bin を PATH に加える
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs={"max_pixels": max_pixels},
            seed=0,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=32, **self._structured_kwargs())

    @staticmethod
    def _structured_kwargs() -> dict:
        """出力を2軸のラベルの JSON に制約する vLLM の設定を作る．

        Returns:
            dict: SamplingParams に渡すキーワード引数．制約付き生成に対応しない版では空の辞書．
        """
        pattern = (r'\{"format": "(' + "|".join(FORMAT_LABELS) + r')", "domain": "(' + "|".join(DOMAIN_LABELS) + r')"\}')
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return {"structured_outputs": StructuredOutputsParams(regex=pattern)}
        except ImportError:
            return {}

    def _build_input(self, image: Image.Image) -> dict:
        """画像1枚に対する vLLM の入力を作成する．

        Args:
            image (PIL.Image.Image): RGB 画像．

        Returns:
            dict: プロンプトと画像を含む vLLM の入力．
        """
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": image}}

    def predict(self, images: list) -> list:
        """画像のリストの種別を予測する．

        Args:
            images (list[PIL.Image.Image]): RGB 画像のリスト（長さ N）．

        Returns:
            list[dict]: 長さ N のリスト．各要素はキー format，domain，raw（生成した文字列）を持つ辞書．
        """
        outputs = self.llm.generate([self._build_input(im) for im in images], sampling_params=self.sampling_params,
                                    use_tqdm=False)
        return [parse_vlm_output(o.outputs[0].text) for o in outputs]


class OcrTextDetector:
    """YomiToku の OCR で文字を検出し，文字の量から「文章のお題」かを判定する規則ベースの分類器．

    穴埋めとカルタの記号や画風は判定できないため，形式を normal と text の2値で出力する．
    """

    def __init__(self, area_threshold: float = 0.08, char_threshold: int = 8, device: str = None):
        """
        Args:
            area_threshold (float): 文字領域の外接矩形の面積の総和が画像に占める割合のしきい値．
            char_threshold (int): 読み取った文字数のしきい値．
            device (str): 推論に用いるデバイス．None の場合は自動で選ぶ．
        """
        from yomitoku import OCR

        self.area_threshold = area_threshold
        self.char_threshold = char_threshold
        self.analyzer = OCR(visualize=False, device=device or get_device())

    def features(self, image: Image.Image) -> dict:
        """画像1枚の OCR を行い，文字の量の特徴量を求める．

        Args:
            image (PIL.Image.Image): RGB 画像．

        Returns:
            dict: キー n_chars（文字数，int），area_ratio（文字領域の面積の割合，float），text（読み取った文字列，str）．
        """
        arr = np.asarray(image)[:, :, ::-1].copy()  # YomiToku は OpenCV と同じ BGR の配列を受け取る
        h, w = arr.shape[:2]
        if min(h, w) < 32:
            return {"n_chars": 0, "area_ratio": 0.0, "text": ""}
        result, _ = self.analyzer(arr)
        words = result.words
        area = 0.0
        for word in words:
            pts = np.array(word.points)
            area += (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        text = " ".join(word.content for word in words)
        return {"n_chars": len(text.replace(" ", "")), "area_ratio": float(area / (h * w)), "text": text}

    def predict(self, feats: dict) -> str:
        """特徴量から形式（normal または text）を判定する．

        Args:
            feats (dict): features の戻り値．

        Returns:
            str: "text" または "normal"．
        """
        is_text = feats["area_ratio"] >= self.area_threshold and feats["n_chars"] >= self.char_threshold
        return "text" if is_text else "normal"


def parse_vlm_output(text: str) -> dict:
    """VLM の出力文字列から2軸のラベルを取り出す．

    JSON として読めない場合は，ラベル名の出現で判定し，見つからない場合は normal と illust とする．

    Args:
        text (str): VLM が生成した文字列．

    Returns:
        dict: キー format (str)，domain (str)，raw (str) を持つ辞書．
    """
    try:
        obj = json.loads(text)
        fmt, dom = obj.get("format"), obj.get("domain")
    except (json.JSONDecodeError, AttributeError):
        fmt = next((f for f in FORMAT_LABELS if re.search(rf"\b{f}\b", text)), None)
        dom = next((d for d in DOMAIN_LABELS if re.search(rf"\b{d}\b", text)), None)
    fmt = fmt if fmt in FORMAT_LABELS else "normal"
    dom = dom if dom in DOMAIN_LABELS else "illust"
    return {"format": fmt, "domain": dom, "raw": text}


def flat_to_axes(flat_label: str) -> tuple:
    """7クラスのラベルを形式と画風の2軸に戻す．形式が normal 以外の場合，画風は不明（None）とする．

    Args:
        flat_label (str): FLAT_LABELS のいずれか．

    Returns:
        tuple[str, str | None]: (形式, 画風)．
    """
    if flat_label in FORMAT_LABELS:
        return flat_label, None
    assert flat_label in FLAT_LABELS
    return "normal", flat_label
