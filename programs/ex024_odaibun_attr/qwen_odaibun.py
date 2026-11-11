"""Qwen3-VL に，お題画像に「お題文あり」かを問う．指示（プロンプト）とモデルを変えて比べる．

指示：
- P0：パッケージの現在の確認の質問（投稿者が回答者に向けて主題として書いた文があるか．英語）
- P1：お題文ありの定義（定義書 v5 の案）を日本語で与え，真偽を答えさせる
- P1en：P1 の英語版
- P2：画像中の文を最大6つ書き出させ，文ごとに空欄を含むか・出どころ・種類を答えさせる．判定は規則で導く
  （空欄を含まず，投稿者が書き加え，種類が問い・指示・題目・状況設定・台本や手紙の文のどれかである文があれば真）．
  モデル自身の最終の答え（has_odai_text）も P2self として記録する．
- P3：P1 に，空欄を含む文章のまとまり（段落，説明文，字幕，歌詞など）は全体を空欄を含む文とみなす規則と，判断の手順を加える
- P4：P3 で，お題文に当たる文を書き写させる（odai_text）．書き写した文が空，または空欄の記号を含む場合は偽とする．
  モデル自身の最終の答えも P4self として記録する．

出力：outputs/ex024_odaibun_attr/qwen/{モデル}__{max_pixels}/{指示}.csv（列 fileId, answer, raw）と speed.json

実行例:
    .venv_annotate/bin/python programs/ex024_odaibun_attr/qwen_odaibun.py --model Qwen/Qwen3-VL-8B-Instruct --prompts P0 P1 P1en P2
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, PKG_DIR, image_path, load_truth_v4  # noqa: E402

sys.path.insert(0, PKG_DIR)
from bokete_annotation import verify  # noqa: E402
from bokete_annotation.data import load_rgb  # noqa: E402

P1 = """この画像は大喜利サイト「ボケて」のお題画像です．画像に「お題文」があるかを判定してください．

お題文とは，お題の投稿者が回答者に向けて書き加えた，大喜利の題となる文です．次のようなものです．
- 問い（「〜とは？」「〜なのはなぜ？」「最初に放った一言とは？」）
- 指示（「〜してください」「〜しろ」）
- 題目の句（「こんな〜は嫌だ」「〜あるある」「〜の共通点」「〜な事」）．題目の後に例が列挙され，最後の項目が空欄のものを含む
- 状況の設定（「〜したら…」「〜なのに」）や短い筋書き
- ボケてのテンプレートや無地の背景，画像の上下の帯に投稿者が書いた主題の文（台本，手紙の書き出し，題名など）

次のものはお題文ではありません．
- 空欄（□，○○，？，空の枠，空の吹き出しなど）を含む文そのもの．空欄とは別の文として問い・指示・題目・状況の文がある場合だけ，お題文ありとします
- 元の素材にある文字：テレビの字幕やテロップ，看板，ポスター，本・CD・DVD の表紙，商品，ウェブの記事，SNS や アプリの画面，ゲーム・アニメ・漫画の台詞，歌詞，人物やものの説明文
- 絵の中の吹き出しに書かれた登場人物の台詞
- ひらがな1文字のカルタの記号，ロゴ，透かし，署名

JSON のみを出力してください．"""

P1EN = """This image is a prompt image ("odai") from the Japanese comedy site Bokete. Decide whether the image contains "odai text".

Odai text is text that the poster ADDED for the respondents as the theme of the comedy prompt, such as:
- a question ("〜とは？", "〜なのはなぜ？", "What was the first thing he said?")
- an instruction or request ("〜してください", "〜しろ")
- a topic phrase ("こんな〜は嫌だ", "〜あるある", "〜の共通点", "〜な事"), including a topic followed by a list of examples whose last item is blank
- a situation setting or short scenario ("〜したら…", "〜なのに")
- a theme sentence the poster wrote on a Bokete template, a plain background, or a band added above/below the picture (a script, the opening of a letter, a title)

The following are NOT odai text:
- a sentence that itself contains a blank (□, ○○, ？, an empty box or an empty speech bubble). Answer true only if there is a separate sentence that is a question, instruction, topic or situation
- text from the original material: TV captions and subtitles, signboards, posters, book/CD/DVD covers, products, web articles, SNS or app screens, dialogue in games/anime/manga, song lyrics, descriptions of people or things
- dialogue of characters written in speech bubbles inside the picture
- a single hiragana karuta mark, logos, watermarks, signatures

Answer with JSON only."""

P2 = """この画像は大喜利サイト「ボケて」のお題画像です．画像中のまとまった文字（文，見出し，台詞など）を，重要なものから最大6つ書き出し，それぞれについて次を答えてください．
- text：その文字（30文字以内に省略してよい）
- has_blank：その文の中に空欄（□，○○，？，空の枠など）があるか
- source：poster_added（お題の投稿者が書き加えた文字．ボケてのテンプレートや無地の背景，画像の上下の帯，写真や絵の上に重ねた文字）または original（元の素材にある文字．テレビの字幕，看板，表紙，商品，記事，SNS の画面，漫画やゲームの台詞，歌詞，説明文）
- kind：question（問い），instruction（指示），topic（題目の句や見出し），scenario（状況の設定や筋書き），script_or_letter（台本や手紙の文），speech_bubble（絵の中の吹き出しの台詞），label（名前や項目のラベル），other のどれか
最後に has_odai_text に，お題の投稿者が回答者に向けて書き加えた，空欄を含まない問い・指示・題目・状況の文があるかを答えてください．
JSON のみを出力してください．"""

P3 = P1.replace("""次のものはお題文ではありません．""", """次のものはお題文ではありません．
- 空欄を含む文章のまとまり（段落，説明文，記事，字幕やテロップ，歌詞，引用，台詞）．空欄のない文が含まれていても，同じまとまりの一部ならお題文ではありません""").replace(
    "JSON のみを出力してください．", """判断の手順：まず画像中の文字を，投稿者が書き加えたものと元の素材にあるものに分けます．次に，投稿者が書き加えた文字から，空欄を含む文章のまとまりを除きます．残った文字に問い・指示・題目・状況の文があればお題文ありです．迷う場合はお題文なしとします．
JSON のみを出力してください．""")
P4 = P3.replace("JSON のみを出力してください．", """odai_text には，お題文に当たる文をそのまま書き写してください（40文字以内．ない場合は空の文字列）．
JSON のみを出力してください．""")
BLANK_CHARS = "□■○◯〇？?＿_"
ODAI_KINDS = {"question", "instruction", "topic", "scenario", "script_or_letter"}
BOOL_SCHEMA = {"type": "object", "properties": {"has_odai_text": {"type": "boolean"}}, "required": ["has_odai_text"],
               "additionalProperties": False}
P2_SCHEMA = {"type": "object", "properties": {
    "texts": {"type": "array", "maxItems": 6, "items": {"type": "object", "properties": {
        "text": {"type": "string", "maxLength": 30}, "has_blank": {"type": "boolean"},
        "source": {"type": "string", "enum": ["poster_added", "original"]},
        "kind": {"type": "string", "enum": ["question", "instruction", "topic", "scenario", "script_or_letter", "speech_bubble",
                                           "label", "other"]}},
        "required": ["text", "has_blank", "source", "kind"], "additionalProperties": False}},
    "has_odai_text": {"type": "boolean"}}, "required": ["texts", "has_odai_text"], "additionalProperties": False}
P4_SCHEMA = {"type": "object", "properties": {"odai_text": {"type": "string", "maxLength": 40}, "has_odai_text": {"type": "boolean"}},
             "required": ["odai_text", "has_odai_text"], "additionalProperties": False}
PROMPTS = {
    "P0": (verify.VERIFY_PROMPT, verify.SCHEMA, "is_poster_prompt"),
    "P1": (P1, BOOL_SCHEMA, "has_odai_text"),
    "P1en": (P1EN, BOOL_SCHEMA, "has_odai_text"),
    "P2": (P2, P2_SCHEMA, None),
    "P3": (P3, BOOL_SCHEMA, "has_odai_text"),
    "P4": (P4, P4_SCHEMA, None),
}


def p2_rule(rec: dict) -> bool:
    """P2 の書き出しから規則でお題文ありを導く．

    Args:
        rec (dict): P2 の JSON の出力．

    Returns:
        bool: 空欄を含まず，投稿者が書き加え，種類が問い・指示・題目・状況・台本や手紙の文がある場合に真．
    """
    return any((not t["has_blank"]) and t["source"] == "poster_added" and t["kind"] in ODAI_KINDS for t in rec.get("texts", []))


def parse(name: str, text: str) -> dict:
    """モデルの出力を解釈する．

    Args:
        name (str): 指示の名前．
        text (str): モデルの出力（JSON の文字列）．

    Returns:
        dict: answer（bool．解釈できない場合は None）と，P2 では self（モデル自身の最終の答え）．
    """
    # 推論型のモデル（Thinking）は思考の後に JSON を出すため，最後の JSON の部分を取り出す
    text = text.split("</think>")[-1]
    found = re.findall(r"\{[^{}]*\}", text)
    try:
        rec = json.loads(text) if text.strip().startswith("{") and text.strip().endswith("}") else json.loads(found[-1])
    except (json.JSONDecodeError, IndexError):
        return {"answer": None, "self": None}
    if name in ("P1", "P1en", "P3") and PROMPTS[name][2] not in rec:
        # 推論型のモデルは指定と異なるキーで答えることがあるため，値の真偽の語から読み取る
        vals = " ".join(str(v) for v in rec.values()).lower()
        yes = any(k in vals for k in ["true", "あり", "yes"])
        no = any(k in vals for k in ["false", "なし", "no"])
        return {"answer": yes if yes != no else None, "self": None}
    if name == "P2":
        return {"answer": p2_rule(rec), "self": bool(rec.get("has_odai_text"))}
    if name == "P4":
        txt = rec.get("odai_text", "").strip()
        ok = bool(rec.get("has_odai_text")) and len(txt) > 0 and not any(c in txt for c in BLANK_CHARS)
        return {"answer": ok, "self": bool(rec.get("has_odai_text"))}
    return {"answer": bool(rec.get(PROMPTS[name][2])), "self": None}


def run(llm, name: str, ids: list, max_pixels: int, max_tokens: int, thinking: bool = False) -> tuple:
    """1つの指示で全ての画像に問う．

    Args:
        llm (vllm.LLM): モデル．
        name (str): 指示の名前．
        ids (list[int]): お題IDのリスト．
        max_pixels (int): 入力画像の最大画素数．
        max_tokens (int): 生成の最大トークン数．
        thinking (bool): 推論型のモデルか（真なら出力を JSON に制約しない）．

    Returns:
        tuple[pl.DataFrame, float, int]: 列 fileId, answer, self, raw の表，推論の秒数，生成したトークンの総数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompt, schema, _ = PROMPTS[name]
    if thinking:
        # 推論型のモデルは出力を JSON に制約すると思考も制約されるため，制約せずに生成する
        params = SamplingParams(temperature=0, max_tokens=max_tokens)
    else:
        params = SamplingParams(temperature=0, max_tokens=max_tokens, structured_outputs=StructuredOutputsParams(json=schema))
    with ThreadPoolExecutor(32) as ex:
        images = list(ex.map(load_rgb, [image_path(i) for i in ids]))
    convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im}, {"type": "text", "text": prompt}]}] for im in images]
    start = time.perf_counter()
    outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": max_pixels})
    sec = time.perf_counter() - start
    raws = [o.outputs[0].text for o in outs]
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    rows = [{"fileId": i, **parse(name, r), "raw": r} for i, r in zip(ids, raws)]
    return pl.DataFrame(rows, schema={"fileId": pl.Int64, "answer": pl.Boolean, "self": pl.Boolean, "raw": pl.Utf8}), sec, n_tok


def reparse(out_dir: str) -> None:
    """保存済みの出力（raw）を解釈し直して保存する．

    Args:
        out_dir (str): 出力先（{モデル}__{画素}）．

    Returns:
        None
    """
    for path in sorted(os.listdir(out_dir)):
        if not path.endswith(".csv"):
            continue
        df = pl.read_csv(os.path.join(out_dir, path))
        name = path[:-4]
        rows = [{"fileId": i, **parse(name, r or ""), "raw": r} for i, r in zip(df["fileId"], df["raw"])]
        new = pl.DataFrame(rows, schema={"fileId": pl.Int64, "answer": pl.Boolean, "self": pl.Boolean, "raw": pl.Utf8})
        new.write_csv(os.path.join(out_dir, path))
        sp = json.load(open(os.path.join(out_dir, "speed.json")))
        sp[name]["n_parse_error"] = int(new["answer"].null_count())
        json.dump(sp, open(os.path.join(out_dir, "speed.json"), "w"), indent=1)
        print(name, sp[name])


def main() -> None:
    """指定のモデルで，指定の指示を全ての目視のラベルの画像に問う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--reparse", action="store_true", help="保存済みの出力を解釈し直す（推論はしない）")
    args = parser.parse_args()
    out_dir = os.path.join(OUTPUT_DIR, "qwen", f"{args.model.split('/')[-1]}__{args.max_pixels}")
    if args.reparse:
        reparse(out_dir)
        return
    todo = [p for p in args.prompts if not os.path.exists(os.path.join(out_dir, f"{p}.csv"))]
    if not todo:
        print("skip: all done")
        return
    os.makedirs(out_dir, exist_ok=True)
    ids = load_truth_v4()["fileId"].sort().to_list()
    llm = verify.build_llm(args.model, args.max_pixels)
    speed_path = os.path.join(out_dir, "speed.json")
    speed = json.load(open(speed_path)) if os.path.exists(speed_path) else {}
    for name in tqdm(todo, desc="prompts", leave=False):
        df, sec, n_tok = run(llm, name, ids, args.max_pixels, args.max_tokens, "Thinking" in args.model)
        df.write_csv(os.path.join(out_dir, f"{name}.csv"))
        speed[name] = {"n": len(ids), "sec": sec, "images_per_sec": len(ids) / sec, "output_tokens": n_tok,
                       "n_parse_error": int(df["answer"].null_count())}
        json.dump(speed, open(speed_path, "w"), indent=1)
        print(name, speed[name])


if __name__ == "__main__":
    main()
