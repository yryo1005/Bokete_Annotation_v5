"""Gemini にお題画像のラベル（定義書 v4 の形式・属性・媒体）を付けさせる．

対象は，Claude Code が定義書 v4 で目視のラベルを付けたお題画像（data.py の load_claude_labels，計1,020枚）のうち --set で選んだ集合である．
無料枠は1日の要求数が少ないため，1日の上限に達したら残りを止める（翌日に同じコマンドで続きから再開する）．
指示には定義書 v4 の本文をそのまま与え，JSON（構造化出力）で答えさせる．
1枚ごとに outputs/ex022_gemini_image_labels/{モデル}/raw/{お題ID}.json に保存し，既にある画像は省略する（中断後に再開できる）．
API キーは tokens.json の "gemini" を用いる．

実行例:
    .venv_annotate/bin/python programs/ex022_gemini_image_labels/annotate_gemini.py --model gemini-3.5-flash --set strat --limit 5
    .venv_annotate/bin/python programs/ex022_gemini_image_labels/annotate_gemini.py --model gemini-3.5-flash --set strat
"""
import argparse
import asyncio
import json
import os
import sys
import time

import polars as pl
from google import genai
from google.genai import types
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, PROJECT_DIR, image_path, load_claude_labels  # noqa: E402

DEFINITION = os.path.join(PROJECT_DIR, "programs", "label_guideline", "label_definition_v4.md")
FORMATS = ["karuta", "anaume", "odaibun", "normal"]
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
MAX_RETRY = 5
DAILY_EXHAUSTED = asyncio.Event()  # 1日の上限に達したら以降の要求を止める

INSTRUCTION = """あなたは大喜利サイト「ボケて」のお題画像にラベルを付ける判断者です．
次のラベル定義（第4版）に厳密に従い，与えられた1枚のお題画像の形式・属性・媒体を判定してください．

{definition}

## 回答の形式

JSON で答えてください．
- format：形式（karuta＝カルタ，anaume＝穴埋め，odaibun＝お題文，normal＝通常）から1つ
- has_blank：空欄あり（true/false）
- has_karuta_mark：カルタ記号あり（true/false）
- has_text：文字あり（true/false）
- media：媒体のリスト（photo＝写真，anime＝アニメ，game＝ゲーム，manga＝マンガ，text_graphic＝テキストグラフィック，painting＝絵画，illustration＝イラスト）．原則として1つ，合成画像で異なる媒体の領域がある場合のみ複数
- reason：判断の根拠（日本語で1〜2文）
"""


def response_schema() -> types.Schema:
    """構造化出力のスキーマを作る．

    Returns:
        types.Schema: format, has_blank, has_karuta_mark, has_text, media, reason を持つオブジェクトのスキーマ．
    """
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "format": types.Schema(type=types.Type.STRING, enum=FORMATS),
            "has_blank": types.Schema(type=types.Type.BOOLEAN),
            "has_karuta_mark": types.Schema(type=types.Type.BOOLEAN),
            "has_text": types.Schema(type=types.Type.BOOLEAN),
            "media": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING, enum=MEDIA)),
            "reason": types.Schema(type=types.Type.STRING),
        },
        required=["reason", "format", "has_blank", "has_karuta_mark", "has_text", "media"],
        property_ordering=["reason", "format", "has_blank", "has_karuta_mark", "has_text", "media"],
    )


def make_client() -> genai.Client:
    """tokens.json の API キーで Gemini のクライアントを作る．

    Returns:
        genai.Client: クライアント．
    """
    with open(os.path.join(PROJECT_DIR, "tokens.json"), encoding="utf-8") as f:
        return genai.Client(api_key=json.load(f)["gemini"])


async def annotate_one(client: genai.Client, model: str, config: types.GenerateContentConfig, fid: int, out_path: str,
                       sem: asyncio.Semaphore) -> bool:
    """1枚の画像のラベルを Gemini に付けさせ，JSON で保存する．

    Args:
        client (genai.Client): クライアント．
        model (str): モデル名．
        config (types.GenerateContentConfig): 生成の設定．
        fid (int): お題ID．
        out_path (str): 保存先．
        sem (asyncio.Semaphore): 同時に送る要求の数を制限するセマフォ．

    Returns:
        bool: 保存できたか．
    """
    with open(image_path(fid), "rb") as f:
        image = types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
    async with sem:
        for k in range(MAX_RETRY):
            if DAILY_EXHAUSTED.is_set():
                return False
            try:
                t0 = time.time()
                res = await client.aio.models.generate_content(model=model, contents=[image, "この画像のラベルを判定してください．"],
                                                               config=config)
                out = json.loads(res.text)
                u = res.usage_metadata
                out.update({"fileId": fid, "model": model, "seconds": round(time.time() - t0, 2),
                            "prompt_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
                            "thought_tokens": u.thoughts_token_count})
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False)
                return True
            except Exception as e:  # noqa: BLE001  API の一時的な失敗は待って再試行する
                msg = str(e)
                if "PerDay" in msg:
                    tqdm.write(f"{fid}: 1日の上限に達した")
                    DAILY_EXHAUSTED.set()
                    return False
                tqdm.write(f"{fid}: {type(e).__name__}: {msg[:120]}（{k + 1}回目）")
                await asyncio.sleep(30 * (k + 1))
    return False


async def run(model: str, target: str, limit: int, concurrency: int) -> None:
    """未処理の画像の全てにラベルを付けさせる．

    Args:
        model (str): モデル名．
        target (str): 対象の集合（all / eval / random / strat）．
        limit (int): 処理する枚数の上限（0 で全て）．
        concurrency (int): 同時に送る要求の数．

    Returns:
        None
    """
    raw_dir = os.path.join(OUTPUT_DIR, model, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    lab = load_claude_labels()
    if target != "all":
        lab = lab.filter(pl.col("set") == target)
    ids = sorted(set(lab["fileId"].to_list()))
    todo = [f for f in ids if not os.path.exists(os.path.join(raw_dir, f"{f}.json"))]
    if limit:
        todo = todo[:limit]
    with open(DEFINITION, encoding="utf-8") as f:
        definition = f.read()
    config = types.GenerateContentConfig(system_instruction=INSTRUCTION.format(definition=definition),
                                         response_mime_type="application/json", response_schema=response_schema())
    client = make_client()
    sem = asyncio.Semaphore(concurrency)
    tasks = [annotate_one(client, model, config, f, os.path.join(raw_dir, f"{f}.json"), sem) for f in todo]
    ok = 0
    for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), leave=False):
        ok += await t
    print(f"{model}：対象 {len(ids)} 枚，今回 {len(todo)} 枚，成功 {ok} 枚")


def main() -> None:
    """コマンドライン引数を読み，アノテーションを実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--set", default="strat", choices=["all", "eval", "random", "strat"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(run(args.model, args.set, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
