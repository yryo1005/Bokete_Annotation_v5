"""VLM（Qwen3-VL）の教師で，ボケにラベルを付ける（お題画像，お題の形式，ボケの文を与える）．

問う項目（定義書 boke_label_definition_v1.md）：制約の順守（カルタ・穴埋め・お題文のみ），発話の型，必要な知識，
画像中の文字への依存，性的・不適切．出力は JSON スキーマで制約し，温度0で生成する．

- --sets eval：検証データ420件．outputs/ex013_boke_labels/teacher_{モデル}/eval.csv
- --sets train：学習用のボケ（datasets/ex013_boke_labels/train_pool.parquet）．同じ場所の train.csv

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/teacher.py --model q30a3 --sets eval
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, KNOWS, OUTPUT_DIR, VOICES, image_path, load_eval  # noqa: E402

MODELS = {"q8": "Qwen/Qwen3-VL-8B-Instruct", "q30a3": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
          "q32": "Qwen/Qwen3-VL-32B-Instruct-FP8"}
FORMAT_JA = {"karuta": "カルタ（札のひらがなで始まる一言を答える）", "anaume": "穴埋め（空欄に入る語句を答える）",
             "odaibun": "お題文（画像中の問いや指示に答える）", "normal": "通常（画像に自由に一言を付ける）"}
PROMPT = """これは大喜利サイト「ボケて」のお題画像と，それに投稿されたボケ（回答）です．
お題の形式：{fmt}
ボケ：「{boke}」

次の項目を判定し，JSON のみを出力してください．
- complies：お題の形式の制約に従っているか．カルタは札のひらがなと同じ音で始まるか（漢字・カタカナは読みで判断），
  穴埋めは空欄に入る語句か，お題文は問いや指示に応じているか．形式が通常なら true．お題と無関係な文（宣伝など）は false．
- voice：ボケの発話の型．S＝画像内の人物・動物・物のセリフ（掛け合いを含む），M＝画像内の人物の心の声・独り言，
  T＝画像を見る側のツッコミ・感想・呼びかけ，N＝画像の状況の説明や前後の物語，L＝名付け・見立て・題（「〜の時の俺」「〜界の〜」を含む），
  A＝お題文の問いや穴埋めの空欄への回答．
- knowledge：ボケを理解するのに必要な知識．0＝画像の見た目とボケの文だけで分かる，G＝一般常識・言葉・ことわざの知識，
  I＝画像に写っている人物・キャラクター・作品・商品が何であるかの情報，O＝画像に写っていない作品・番組・歌・CM・地域・人物・時事の知識．
  I と O の両方なら I．
- uses_image_text：ボケの意味が画像中の文字（お題文，穴埋めの文，字幕，看板，漫画の台詞，札の文字そのもの）を読まないと分からないか．
- inappropriate：性的な内容，特定の属性への差別や侮辱，実在の人物への重大な中傷，犯罪や暴力の助長を含むか（軽い下品さは含まない）．"""
# 第2版：検証データの調整用（dev）で，知識をほぼ全て I とする，通常のお題で回答（A）を選ぶ，セリフとツッコミを混同する誤りが多かった．
# そこで (1) 回答は形式から規則で決め，モデルには残りの5種から選ばせる，(2) 知識は「ボケが頼る特定の人物・作品・事象」を書き出させ，
# それが画像に写っているかと，一般常識・言葉遊びに頼るかを別に問い，規則で4段階にする，(3) 不適切はほのめかしを含めて別に問う．
# 例は検証データと重ならないよう新たに作った．
PROMPT_V2 = """これは大喜利サイト「ボケて」のお題画像と，それに投稿されたボケです．
お題の形式：{fmt}
ボケ：「{boke}」

次の項目を判定し，JSON のみを出力してください．
1. complies：お題の形式の制約に従っているか．カルタは札のひらがなと同じ音で始まるか（漢字・カタカナは読みで判断），穴埋めは空欄に入る語句か，
   お題文は問いや指示に応じているか．形式が通常なら true．お題と無関係な文（宣伝など）は false．
2. voice：誰の視点の言葉か．次の順に考えて1つ選ぶ．
   - S：画像の中の人物・動物・物が，口に出して言っている言葉（誰かへの話しかけ，叫び，掛け合い）．例：犬の写真に「散歩まだ？」
   - M：画像の中の人物などが心の中で思っていること・独り言．例：寝ている猫に「あと5分だけ…」
   - T：画像を見ている人が，画像について言うツッコミや感想．画像の中の人物の言葉ではない．例：大きな帽子の人に「帽子でかすぎだろ」
   - N：語り手が，画像の状況や前後の出来事を説明する文．例：転んだ人に「バナナの皮を踏んだ直後」
   - L：画像や写っている物に，名前・題・例えを付ける言葉（名詞で終わることが多い）．例：丸い猫に「もち」，「〜の時の俺」
3. referent：ボケが面白さのために頼っている，特定の人物・キャラクター・作品・番組・商品・地名・出来事の名前（なければ空文字）．
   犬・男性・車のような一般的な物は含めない．
4. referent_in_image：referent が画像に写っている（画像の人物やキャラクターそのもの，画像の作品）なら true，画像の外のものなら false．
5. common_knowledge：ことわざ，慣用句，ダジャレ，一般的な常識の知識を使っているか．
6. uses_image_text：ボケの意味が，画像中の文字（字幕，看板，漫画の台詞，説明文，札の文字そのもの）を読まないと分からないか．
7. inappropriate：性的な内容（ほのめかし，下ネタ，性的な部位や行為への言及を含む），差別や侮辱，実在の人物への重大な中傷，暴力や犯罪の助長を含むか．"""
SCHEMA_V2 = {"type": "object", "properties": {
    "complies": {"type": "boolean"}, "voice": {"type": "string", "enum": ["S", "M", "T", "N", "L"]},
    "referent": {"type": "string", "maxLength": 40}, "referent_in_image": {"type": "boolean"},
    "common_knowledge": {"type": "boolean"}, "uses_image_text": {"type": "boolean"}, "inappropriate": {"type": "boolean"}},
    "required": ["complies", "voice", "referent", "referent_in_image", "common_knowledge", "uses_image_text", "inappropriate"],
    "additionalProperties": False}


def derive_v2(rec: dict, fmt: str) -> dict:
    """第2版の回答を5項目のラベルにする（回答の型と画像中の文字は形式から，知識は規則で決める）．

    Args:
        rec (dict): モデルの回答．
        fmt (str): お題の形式．

    Returns:
        dict: complies, voice, knowledge, uses_image_text, inappropriate．
    """
    ref = (rec.get("referent") or "").strip()
    if ref:
        know = "I" if rec.get("referent_in_image") else "O"
    else:
        know = "G" if rec.get("common_knowledge") else "0"
    qa = fmt in ("anaume", "odaibun")
    return {"complies": bool(rec.get("complies", True)), "voice": "A" if qa else rec.get("voice", "S"), "knowledge": know,
            "uses_image_text": True if qa else bool(rec.get("uses_image_text", False)),
            "inappropriate": bool(rec.get("inappropriate", False)), "referent": ref}


SCHEMA = {"type": "object", "properties": {
    "complies": {"type": "boolean"}, "voice": {"type": "string", "enum": VOICES},
    "knowledge": {"type": "string", "enum": KNOWS}, "uses_image_text": {"type": "boolean"},
    "inappropriate": {"type": "boolean"}},
    "required": ["complies", "voice", "knowledge", "uses_image_text", "inappropriate"], "additionalProperties": False}
CHUNK = 2048


def load_image(fid: int) -> Image.Image:
    """お題画像を RGB で読み込む（極端に細い画像は余白を足す）．

    Args:
        fid (int): お題ID．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(fid)) as im:
        im = im.convert("RGB")
    if min(im.size) < 32:
        canvas = Image.new("RGB", (max(im.width, 32), max(im.height, 32)), "white")
        canvas.paste(im, (0, 0))
        im = canvas
    return im


def build_llm(model: str):
    """vLLM のモデルを読み込む．

    Args:
        model (str): MODELS のキー．

    Returns:
        vllm.LLM: 読み込んだモデル．
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM

    return LLM(model=MODELS[model], max_model_len=4096, gpu_memory_utilization=0.88, seed=0,
               limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs={"max_pixels": 448 * 448})


def label(llm, df: pl.DataFrame, version: str = "v1") -> tuple:
    """ボケの表にラベルを付ける．

    Args:
        llm (vllm.LLM): モデル．
        df (pl.DataFrame): 列 odaiId, text, format を持つ表．
        version (str): 指示の版（v1 / v2）．

    Returns:
        tuple[pl.DataFrame, float]: 列 complies, voice, knowledge, uses_image_text, inappropriate の表と推論の秒数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompt, schema = (PROMPT_V2, SCHEMA_V2) if version == "v2" else (PROMPT, SCHEMA)
    params = SamplingParams(temperature=0, max_tokens=160, structured_outputs=StructuredOutputsParams(json=schema))
    recs, sec = [], 0.0
    rows = df.select("odaiId", "text", "format").rows()
    with ThreadPoolExecutor(32) as ex:
        for s in range(0, len(rows), CHUNK):
            part = rows[s:s + CHUNK]
            uniq = sorted({r[0] for r in part})
            imgs = dict(zip(uniq, ex.map(load_image, uniq)))
            convs = [[{"role": "user", "content": [
                {"type": "image_pil", "image_pil": imgs[fid]},
                {"type": "text", "text": prompt.format(fmt=FORMAT_JA[fmt], boke=(text or "").replace("\n", " ")[:300])}]}]
                for fid, text, fmt in part]
            start = time.perf_counter()
            outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": 448 * 448})
            sec += time.perf_counter() - start
            for (fid, text, fmt), o in zip(part, outs):
                try:
                    rec = json.loads(o.outputs[0].text)
                except json.JSONDecodeError:
                    rec = {}
                if version == "v2":
                    recs.append(derive_v2(rec, fmt))
                else:
                    recs.append({"complies": True, "voice": "S", "knowledge": "G", "uses_image_text": False,
                                 "inappropriate": False, **rec})
    return pl.DataFrame(recs), sec


def main() -> None:
    """指定したモデルで指定した集合にラベルを付ける．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--sets", nargs="+", default=["eval"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    args = parser.parse_args()
    out = os.path.join(OUTPUT_DIR, f"teacher_{args.model}" + ("" if args.version == "v1" else f"_{args.version}"))
    os.makedirs(out, exist_ok=True)
    llm = build_llm(args.model)
    for name in args.sets:
        path = os.path.join(out, f"{name}.csv")
        if os.path.exists(path):
            continue
        if name == "eval":
            df = load_eval().select("no", "odaiId", "text", "format")
        else:
            df = pl.read_parquet(os.path.join(DATASET_DIR, f"{name}_pool.parquet"))
        lab, sec = label(llm, df, args.version)
        pl.concat([df.drop("text"), lab], how="horizontal").write_csv(path)
        json.dump({"n": df.height, "sec": sec, "per_sec": df.height / sec}, open(path.replace(".csv", "_speed.json"), "w"))
        print(args.model, name, f"{df.height / sec:.1f} 件/秒", flush=True)


if __name__ == "__main__":
    main()
