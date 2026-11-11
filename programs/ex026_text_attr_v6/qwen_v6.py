"""Qwen3-VL に，定義書 v6 の草案の文の属性（文字の層と文の形）と，重ねた文字・余白の文字の書き起こしを問う．

指示：
- V6：v6 の属性の定義を与え，層ごとの有無，重ねた文字・余白の文字の書き起こしと文の形を JSON で答えさせる．
- OCR：重ねた文字と余白の文字の書き起こしだけを答えさせる（小さなモデルで書き起こしの速さを測るためのもの）．

出力：outputs/ex026_text_attr_v6/qwen/{モデル}__{max_pixels}/{指示}.csv（列 rep_odaiId と属性の列，text_o, text_m, raw）と speed.json

実行例:
    .venv_annotate/bin/python programs/ex026_text_attr_v6/qwen_v6.py --model Qwen/Qwen3-VL-32B-Instruct-FP8 --prompt V6
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FORMS, OUTPUT_DIR, PROJECT_DIR, TARGETS, image_path  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0"))
from bokete_annotation import verify  # noqa: E402
from bokete_annotation.data import load_rgb  # noqa: E402

FORM_ENUM = {"blank_sentence": "k", "question": "q", "instruction": "i", "topic": "t", "other": "x"}
V6 = """この画像は大喜利サイト「ボケて」のお題画像です．画像の中の読める文字を，次の4つの層に分けて答えてください．誰が書いたか，元の素材の文字か投稿者が加えた文字かは問いません．見た目だけで判断してください．

層：
- scene（場面の文字）：写っている世界の中の文字．看板，ポスター，商品や衣服の表示，本の表紙，黒板，写真に写った画面，ウェブ・SNS・アプリ・ゲームのスクリーンショットの画面の中の文字，絵に描き込まれた効果音．透かし，ロゴ，署名，著作権表示，日付，ボケてのテンプレートに印刷された定型の見出し（「写真で一言ボケて」）もここに含めます．
- bubble（吹き出し）：漫画やイラストの吹き出しと語りの枠の中の文字．吹き出しのない，絵の中の人物の脇に描き込まれた台詞も含めます．
- overlay（重ねた文字）：写真や絵の上に重ねた文字．テレビの字幕やテロップ，写真への書き込み，半透明の帯の文字．
- margin（余白の文字）：写真や絵の外にある文字．上下左右の帯や枠，無地の背景，テンプレートの文字欄の文字．文字が主体の画像（スクリーンショットを除く）の文字もここに含めます．

overlay と margin については，文字を書き起こし（text．80文字以内に省略してよい），その層にある文の形を全て選んでください（forms）．
文の形：
- blank_sentence：空欄（□，○○，＿，語の代わりの？，空の枠など）を含む文，または空欄を含む文章のまとまり（段落，説明文，字幕，歌詞，台詞，台本）．まとまりの中の空欄のない文もこれに含めます．
- question：疑問の形の文（〜とは？，なぜ〜？，どっち？），「もし〜たら」「〜したら」の条件の文．
- instruction：依頼や命令の形の文（〜してください，〜を教えて，一言ボケて）．
- topic：大喜利の題となる句（〜あるある，〜な事，〜の共通点，こんな〜は嫌だ，〜する方法，〜ランキング）．
- other：上のどれでもない文や語句（平叙文，台詞，感想，状況の説明，題名，名前，名札だけの見出し）．
空欄を含むまとまりの中の問いは question としません．空欄を含むまとまりとは別に問いの文があれば，blank_sentence と question の両方を選びます．
文字がない層は present を false にします．JSON のみを出力してください．"""
OCR = """この画像は大喜利サイト「ボケて」のお題画像です．写真や絵の上に重ねた文字（字幕，テロップ，書き込み）と，写真や絵の外にある文字（上下の帯，枠，無地の背景，テンプレートの文字欄）をそれぞれそのまま書き起こしてください．看板や商品などの写っている世界の中の文字，吹き出しの文字，ロゴや透かしは除きます．ない場合は空の文字列にします．JSON のみを出力してください．"""
LAYER_SCHEMA = {"type": "object", "properties": {"present": {"type": "boolean"}, "text": {"type": "string", "maxLength": 80},
                                                  "forms": {"type": "array", "items": {"type": "string", "enum": list(FORM_ENUM)}}},
                "required": ["present", "text", "forms"], "additionalProperties": False}
V6_SCHEMA = {"type": "object", "properties": {"scene": {"type": "boolean"}, "bubble": {"type": "boolean"}, "overlay": LAYER_SCHEMA,
                                              "margin": LAYER_SCHEMA},
             "required": ["scene", "bubble", "overlay", "margin"], "additionalProperties": False}
OCR_SCHEMA = {"type": "object", "properties": {"overlay": {"type": "string", "maxLength": 120}, "margin": {"type": "string", "maxLength": 120}},
              "required": ["overlay", "margin"], "additionalProperties": False}
PROMPTS = {"V6": (V6, V6_SCHEMA), "OCR": (OCR, OCR_SCHEMA)}


def parse_v6(text: str) -> dict:
    """V6 の出力を属性の真偽と書き起こしに変える．

    Args:
        text (str): モデルの出力（JSON の文字列）．

    Returns:
        dict: キー ok（解釈できたか），L_{s,b,o,m}，F_{o,m}_{形}，text_o, text_m．
    """
    rec = {"ok": False, "L_s": False, "L_b": False, "L_o": False, "L_m": False, "text_o": "", "text_m": ""}
    rec.update({f"F_{l}_{f}": False for l in ["o", "m"] for f in FORMS})
    try:
        j = json.loads(text)
    except json.JSONDecodeError:
        return rec
    rec.update(ok=True, L_s=bool(j["scene"]), L_b=bool(j["bubble"]))
    for key, l in [("overlay", "o"), ("margin", "m")]:
        layer = j[key]
        present = bool(layer["present"]) and (len(layer["text"].strip()) > 0 or len(layer["forms"]) > 0)
        rec[f"L_{l}"] = present
        rec[f"text_{l}"] = layer["text"] if present else ""
        for f in layer["forms"] if present else []:
            rec[f"F_{l}_{FORM_ENUM[f]}"] = True
    return rec


def parse_ocr(text: str) -> dict:
    """OCR の出力を書き起こしに変える．

    Args:
        text (str): モデルの出力（JSON の文字列）．

    Returns:
        dict: キー ok, text_o, text_m．
    """
    try:
        j = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "text_o": "", "text_m": ""}
    return {"ok": True, "text_o": j["overlay"], "text_m": j["margin"]}


def run(llm, name: str, ids: list, max_pixels: int) -> tuple:
    """1つの指示で全ての画像に問う．

    Args:
        llm (vllm.LLM): モデル．
        name (str): 指示の名前（V6 / OCR）．
        ids (list[int]): 代表の画像のお題IDのリスト．
        max_pixels (int): 入力画像の最大画素数．

    Returns:
        tuple[pl.DataFrame, float, int]: 結果の表，推論の秒数，生成したトークンの総数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompt, schema = PROMPTS[name]
    params = SamplingParams(temperature=0, max_tokens=768, structured_outputs=StructuredOutputsParams(json=schema))
    with ThreadPoolExecutor(32) as ex:
        images = list(ex.map(load_rgb, [image_path(i) for i in ids]))
    convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im}, {"type": "text", "text": prompt}]}] for im in images]
    start = time.perf_counter()
    outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": max_pixels})
    sec = time.perf_counter() - start
    parse = parse_v6 if name == "V6" else parse_ocr
    rows = [{"rep_odaiId": i, **parse(o.outputs[0].text), "raw": o.outputs[0].text} for i, o in zip(ids, outs)]
    return pl.DataFrame(rows), sec, sum(len(o.outputs[0].token_ids) for o in outs)


def main() -> None:
    """引数のモデルと指示で対象の画像に問い，結果と速さを保存する（結果があれば飛ばす）．

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct-FP8")
    ap.add_argument("--max_pixels", type=int, default=802816)
    ap.add_argument("--prompt", nargs="+", default=["V6"])
    args = ap.parse_args()
    out_dir = os.path.join(OUTPUT_DIR, "qwen", f"{args.model.split('/')[-1]}__{args.max_pixels}")
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in args.prompt if not os.path.exists(os.path.join(out_dir, f"{p}.csv"))]
    if not todo:
        print("skip: 全ての指示の結果がある")
        return
    ids = pl.read_csv(TARGETS)["rep_odaiId"].to_list()
    llm = verify.build_llm(args.model, args.max_pixels)
    speed_path = os.path.join(out_dir, "speed.json")
    speed = json.load(open(speed_path)) if os.path.exists(speed_path) else {}
    for name in todo:
        df, sec, n_tok = run(llm, name, ids, args.max_pixels)
        df.write_csv(os.path.join(out_dir, f"{name}.csv"))
        speed[name] = {"n": len(ids), "sec": sec, "img_per_sec": len(ids) / sec, "tokens": n_tok, "parse_fail": int((~df["ok"]).sum())}
        print(name, speed[name])
    json.dump(speed, open(speed_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
