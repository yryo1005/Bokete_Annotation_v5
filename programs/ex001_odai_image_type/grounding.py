"""Qwen3-VL による穴埋めの記号とカルタのひらがなの位置の検出（グラウンディング）を試行する．

種別の判定の後段で必要となる「記号の位置」「ひらがなの読み」を，同じ VLM で取得できるかを確かめる．
検出結果は outputs/ex001_odai_image_type/grounding/{モデル}/0/ に，JSON と矩形を描画した画像として保存する．

実行例:
    .venv_annotate/bin/python programs/ex001_odai_image_type/grounding.py
"""
import json
import os
import re
import sys

import polars as pl
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_eval_labels  # noqa: E402
from train import OUTPUT_DIR, load_image, short_name  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
N_PER_FORMAT = 8
PROMPTS = {
    "anaume": ('Locate every blank placeholder in this image that a user must fill in with words '
               '(empty boxes, empty speech bubbles, ○○, □ or ？). '
               'Output a JSON list only: [{"bbox_2d": [x1, y1, x2, y2], "label": "blank"}]'),
    "karuta": ('Locate the hiragana character written inside a circle or square (the karuta mark). '
               'Output a JSON list only: [{"bbox_2d": [x1, y1, x2, y2], "label": "<the hiragana character>"}]'),
}


def parse_boxes(text: str) -> list:
    """VLM の出力から矩形のリストを取り出す．

    Args:
        text (str): VLM が生成した文字列（JSON のリストを含む）．

    Returns:
        list[dict]: 各要素はキー bbox_2d（0〜1000 に正規化した [x1, y1, x2, y2]）と label を持つ辞書．
    """
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        boxes = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [b for b in boxes if isinstance(b, dict) and len(b.get("bbox_2d", [])) == 4]


def draw_boxes(image: Image.Image, boxes: list, font: ImageFont.ImageFont) -> Image.Image:
    """画像に検出した矩形とラベルを描画する．

    Args:
        image (PIL.Image.Image): RGB 画像．
        boxes (list[dict]): parse_boxes の戻り値．
        font (ImageFont.ImageFont): ラベルの描画に用いるフォント．

    Returns:
        PIL.Image.Image: 矩形を描画した画像の複製．
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for b in boxes:
        x1, y1, x2, y2 = [v / 1000 for v in b["bbox_2d"]]
        rect = [x1 * w, y1 * h, x2 * w, y2 * h]
        draw.rectangle(rect, outline=(255, 0, 0), width=4)
        draw.text((rect[0] + 2, max(rect[1] - 30, 0)), str(b.get("label", "")), fill=(255, 0, 0), font=font)
    return out


def main() -> None:
    """評価用の穴埋めとカルタの画像に対して位置の検出を行い，結果を保存する．

    Returns:
        None
    """
    from vllm import SamplingParams

    from model import VLMClassifier

    labels = load_eval_labels()
    targets = []
    for fmt in PROMPTS:
        ids = labels.filter((pl.col("format") == fmt) & pl.col("pool").str.starts_with(
            "type_" if fmt == "anaume" else "tag_"))["fileId"].to_list()[:N_PER_FORMAT]
        targets += [(fid, fmt) for fid in ids]

    out_dir = os.path.join(OUTPUT_DIR, "grounding", short_name(MODEL_NAME), "0")
    os.makedirs(out_dir, exist_ok=True)
    clf = VLMClassifier(MODEL_NAME, max_pixels=1024 * 1024)
    params = SamplingParams(temperature=0, max_tokens=256)
    images = [load_image(fid) for fid, _ in targets]
    inputs = []
    for (fid, fmt), im in zip(targets, images):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[fmt]}]}]
        prompt = clf.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": im}})
    outputs = clf.llm.generate(inputs, sampling_params=params, use_tqdm=False)

    font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    font = ImageFont.truetype(font_path, 28) if os.path.exists(font_path) else ImageFont.load_default()
    records = []
    for (fid, fmt), im, o in zip(targets, images, outputs):
        boxes = parse_boxes(o.outputs[0].text)
        draw_boxes(im, boxes, font).save(os.path.join(out_dir, f"{fmt}_{fid}.jpg"), quality=90)
        records.append({"fileId": fid, "format": fmt, "raw": o.outputs[0].text, "boxes": boxes})
    json.dump(records, open(os.path.join(out_dir, "results.json"), "w"), indent=2, ensure_ascii=False)
    json.dump({"model": MODEL_NAME, "max_pixels": 1024 * 1024, "prompts": PROMPTS},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"{len(records)} 枚の検出結果を {out_dir} に保存した")


if __name__ == "__main__":
    main()
