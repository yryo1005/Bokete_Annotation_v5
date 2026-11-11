"""VLM の教師で，発話の型（6区分）と画像外の知識（2値）を判定する指示と入力形式を比べる．

入力は全ての方法で，お題画像（最大 448×448 画素）・お題の形式・ボケの文である．次の要素を加減して比べる．
- dec（分解した問い）：発話の型を1つの選択で問わず，定義書 第3版の手順どおり「名付けか」「話し手は画像の中か」「声に出しているか」
  「状況の説明かツッコミか」に分けて問い，規則で6区分にする．先に「話し手は誰か」を短い文で書かせる．
  画像外の知識は，依存する事物を書き出させてから，特定性・画像の外・必要性の3条件を問い，全てを満たすときに「はい」とする．
- fs（例示）：検証データとは別に作った例（文で書いた画像とボケ，判定）を指示に加える．
- cap（文の入力）：写真のキャプション（BoketeDBv5 の caption）と，お題の媒体・画像中の文字の有無を文で加える．
- one（一括の選択）：第3版の定義の6区分を1つ選ばせ，画像外の知識を1つの真偽で問う（分解しない対照）．
- dec2：dec で話し手を「画像を見ている人」に寄せて答える偏りが見られたため，話し手の既定を画像の中の誰かとし，
  画像の外の言葉であることが明らかな場合のみ外とするよう問い直したもの．

- kt（種類つきの知識）：依存する事物の種類（作品，人物，商品，地名，出来事，ボケての慣習，一般，なし）を選ばせ，固有の種類で，
  画像の外（写っているキャラクターと同じ作品の事物は画像の中とする）で，必要なときに「はい」とする．dec で普通名詞を特定の事物と
  答える誤りが多かったため．
- hr：お題画像を最大 896×896 画素で入力する（既定は 448×448）．

方法：one，one_cap，one_kt_cap，one_kt_cap_hr，dec，dec_fs，dec_cap，dec_fs_cap，dec2．形式が穴埋め・お題文のボケの発話の型は回答とする（規則）．

集合（--sets）：eval（検証データ620件），aug（ex014 の水増し用の9,041件），
extra（ex013 の学習用のボケ3.5万件のうち，aug と重なるものと検証データのお題のものを除いた約2.6万件．列 train_index は元の行番号）

出力：outputs/ex015_boke_labels_v3/teacher_{モデル}/{方法}/{集合}.csv，{集合}_speed.json

実行例:
    .venv_annotate/bin/python programs/ex015_boke_labels_v3/teacher.py --model q32 --sets eval
    .venv_annotate/bin/python programs/ex015_boke_labels_v3/teacher.py --model q32 --sets aug --variants dec_fs_cap
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX13_DATASET, EX13_DIR, EX14_DATASET, OUTPUT_DIR, _load, load_eval, load_odai_labels  # noqa: E402

# ex013 の teacher.py は `from data import ...` で ex013 の data を読むため，読み込む間だけ差し替える
_own = sys.modules["data"]
sys.modules["data"] = _load("ex013_data_for_v3b", os.path.join(EX13_DIR, "data.py"))
T13 = _load("ex013_teacher_for_v3b", os.path.join(EX13_DIR, "teacher.py"))
sys.modules["data"] = _own
VARIANTS = ["one", "dec", "dec_fs", "dec_cap", "dec_fs_cap", "dec2", "one_cap", "one_kt_cap", "one_kt_cap_hr"]
MEDIA_JA = {"photo": "写真", "anime": "アニメ", "game": "ゲーム", "manga": "漫画", "text_graphic": "文字主体の画像", "painting": "絵画",
            "illustration": "イラスト"}

HEAD = """これは大喜利サイト「ボケて」のお題画像と，それに投稿されたボケです．
お題の形式：{fmt}
{extra}ボケ：「{boke}」
"""
VOICE_ONE = """
1. voice：ボケが誰の視点の，どのような言葉か．次の手順で1つ選ぶ．
   (1) ボケが名詞句（体言止め）で，画像全体や写っている物に名前・題・例えを付けている（「〜の時の俺」「〜界の〜」を含む） → L
   (2) 話し手が画像の中の人物・動物・物（擬人化を含む）で，声に出している（呼びかけ，返事，叫び，掛け合い） → S
       心の中の考え，独り言 → M（迷う場合は S）
   (3) 話し手が画像の外の人で，画像の出来事や前後の状況を説明する文 → N
       画像を見て言う感想・指摘・ツッコミ，画像の人物への呼びかけ → T
"""
VOICE_DEC = """
1. 発話の型を，次の問いに分けて答える．
   speaker：ボケの言葉を発している（または思っている）のは誰か．画像中の具体的な人物・動物・物か，「画像を見ている人」「語り手」かを短く書く．
   is_naming：ボケが名詞句（体言止め）で，画像全体や写っている物に名前・題・例えを付けているか（「〜の時の俺」「〜界の〜」を含む）．
   speaker_in_image：話し手が画像の中の人物・動物・物（擬人化を含む）か．画像を見ている人や語り手なら false．
   aloud：（話し手が画像の中の場合）声に出している言葉か（呼びかけ，返事，叫び，掛け合い，相手のいる言葉）．
          心の中の考えや独り言（括弧書き，「…かな」「…しよう」）なら false．迷う場合は true．
   narration：（話し手が画像の外の場合）画像の出来事や前後の状況を説明・設定する文か（「〜したら〜だった」「〜する様子」「〜の結果」）．
          画像を見て言う感想・指摘・ツッコミや，画像の人物への呼びかけなら false．
"""
VOICE_DEC2 = """
1. 発話の型を，次の問いに分けて答える．ボケては画像の中の誰かのセリフや心の声として書かれることが最も多い．
   speaker：このボケを，画像の中の誰か（人物・動物・物．擬人化を含む．話し手が明示されていなくてもよい）の言葉として自然に読めるなら，
            その対象を書く．画像の中の誰の言葉としても読めない場合のみ「画像の外」と書く．
   is_naming：ボケが名詞句（体言止め）で，画像全体や写っている物に名前・題・例えを付けているか（「〜の時の俺」「〜界の〜」を含む）．
   speaker_in_image：speaker が画像の中の対象なら true．画像の外から，画像を三人称で説明する文，前後の状況の設定，画像への感想・ツッコミで
            あることが明らかな場合のみ false．
   aloud：（話し手が画像の中の場合）声に出している言葉か（呼びかけ，返事，叫び，掛け合い，相手のいる言葉）．
          心の中の考えや独り言（括弧書き，「…かな」「…しよう」）なら false．迷う場合は true．
   narration：（話し手が画像の外の場合）画像の出来事や前後の状況を説明・設定する文か（「〜したら〜だった」「〜する様子」「〜の結果」）．
          画像を見て言う感想・指摘・ツッコミや，画像の人物への呼びかけなら false．
"""
KNOW_ONE = """
2. outside：ボケの面白さを理解するのに，画像にも画像中の文字にも示されていない特定の事物（作品，キャラクター，人物，商品，番組，CM，歌，
   地名の印象，出来事，流行語，ネットの定型，有名な台詞，ボケての慣習）の知識が必要か．一般の語彙・慣用句・ことわざ・生活の行事，
   画像に写っている対象の知識，一般的な職業やジャンルの印象は含まない．
"""
KNOW_DEC = """
2. 画像外の知識を，次の手順で答える．
   referent：ボケの面白さが依存している特定の事物（作品，キャラクター，人物，団体，商品，番組，CM，歌，地名の印象，出来事，流行語，
             ネットの定型，有名な台詞，ボケての機能や慣習）の名前．なければ空文字．
   specific：referent が特定の事物か（一般の語彙・慣用句・ことわざ・誰もが知る生活の行事，一般的な職業やジャンルなら false）．
   in_image：referent が画像に写っている，または画像中の文字で名指しされているか（画像の人物やキャラクター自身の設定を使う場合も true）．
   necessary：referent を知らない人には，一般の日本語の語彙と常識があってもボケの面白さが分からないか（名前が背景にすぎないなら false）．
"""
KNOW_TYPED = """
2. 画像外の知識を，次の手順で答える．
   referent：ボケの面白さが依存している事物の名前．なければ空文字．
   referent_type：referent の種類を次から1つ選ぶ．固有名詞でなければ general とする．
     work：特定の作品・キャラクター・番組・映画・ゲーム・歌・CM・有名な台詞やキャッチコピー
     person：実在の特定の人物・団体
     product：特定の商品・ブランド・店
     place：特定の地名と，その土地の印象
     event：特定の出来事・ニュース・流行語・ネットの定型
     bokete：ボケての機能や慣習
     general：普通名詞（一般の物，道具，料理，スポーツ，法律，職業，ジャンル，生活の行事，遊び），ことわざ，慣用句
     none：依存する事物がない
   in_image：referent が画像に写っている，画像中の文字で名指しされている，または画像に写っているキャラクターと同じ作品の登場人物・設定か．
   necessary：referent を知らない人には，一般の日本語の語彙と常識があってもボケの面白さが分からないか（名前が背景にすぎないなら false）．
"""
SPECIFIC_TYPES = ["work", "person", "product", "place", "event", "bokete"]
FEWSHOT = """
例（画像は文で示す）：
- 画像：あくびをする猫．ボケ「会議まだ終わらないの？」→ speaker=猫，is_naming=false，speaker_in_image=true，aloud=true（セリフ）
- 画像：窓の外を見つめる犬．ボケ「今日こそ散歩に行けるかな…」→ speaker=犬，speaker_in_image=true，aloud=false（心の声）
- 画像：巨大なおにぎりを持つ子ども．ボケ「でかすぎだろ」→ speaker=画像を見ている人，speaker_in_image=false，narration=false（ツッコミ）
- 画像：寝ている赤ちゃん．ボケ「起きて！もう昼だよ！」→ speaker=画像を見ている人（赤ちゃんへの呼びかけ），speaker_in_image=false，narration=false
- 画像：泥だらけの男性．ボケ「近道しようとした結果」→ speaker=語り手，speaker_in_image=false，narration=true（状況の説明）
- 画像：丸まった白い猫．ボケ「大福」→ is_naming=true（名付け）
- 画像：腕組みをするおじさん．ボケ「月曜の朝の俺」→ is_naming=true（見立て）
- 画像：二人の男性が向き合う．ボケ「お前，昨日俺のプリン食べただろ」→ speaker=左の男性，speaker_in_image=true，aloud=true
- 画像：猫の絵本の表紙のような猫．ボケ「100万回生きたねこの101回目」→ referent=絵本『100万回生きたねこ』，specific=true，in_image=false，
  necessary=true（画像外の知識：はい）
- 画像：ドラえもんのイラスト．ボケ「どら焼き買ってきて」→ referent=ドラえもん，in_image=true（いいえ）
- 画像：体操をする人々．ボケ「ラジオ体操第三まであるの知らなかった」→ referent=ラジオ体操，specific=false（いいえ）
- 画像：雨の中の男性．ボケ「ディズニーランドで迷子になった帰り」→ referent=ディズニーランド，necessary=false（いいえ）
- 画像：踊る中年男性．ボケ「恋ダンス完コピしました」→ referent=ドラマの恋ダンス，specific=true，in_image=false，necessary=true（はい）
"""
TAIL = "\nJSON のみを出力してください．"
SCHEMA_ONE = {"type": "object", "properties": {"voice": {"type": "string", "enum": ["S", "M", "T", "N", "L"]}, "outside": {"type": "boolean"}},
              "required": ["voice", "outside"], "additionalProperties": False}
SCHEMA_KT = {"type": "object", "properties": {
    "voice": {"type": "string", "enum": ["S", "M", "T", "N", "L"]}, "referent": {"type": "string", "maxLength": 40},
    "referent_type": {"type": "string", "enum": SPECIFIC_TYPES + ["general", "none"]}, "in_image": {"type": "boolean"},
    "necessary": {"type": "boolean"}},
    "required": ["voice", "referent", "referent_type", "in_image", "necessary"], "additionalProperties": False}
SCHEMA_DEC = {"type": "object", "properties": {
    "speaker": {"type": "string", "maxLength": 40}, "is_naming": {"type": "boolean"}, "speaker_in_image": {"type": "boolean"},
    "aloud": {"type": "boolean"}, "narration": {"type": "boolean"},
    "referent": {"type": "string", "maxLength": 40}, "specific": {"type": "boolean"}, "in_image": {"type": "boolean"},
    "necessary": {"type": "boolean"}},
    "required": ["speaker", "is_naming", "speaker_in_image", "aloud", "narration", "referent", "specific", "in_image", "necessary"],
    "additionalProperties": False}


def build_prompt(variant: str, fmt: str, text: str, row: dict) -> str:
    """方法に応じた指示の文を作る．

    Args:
        variant (str): VARIANTS のいずれか．
        fmt (str): お題の形式．
        text (str): ボケの文．
        row (dict): お題のラベル（媒体，has_text，caption）．

    Returns:
        str: 指示の文．
    """
    extra = ""
    if "cap" in variant:
        media = "，".join(MEDIA_JA[m] for m in MEDIA_JA if row.get(m)) or "不明"
        extra = f"お題画像の媒体：{media}．画像中の文字：{'あり' if row.get('has_text') else 'なし'}\n"
        if row.get("caption"):
            extra += f"お題画像の説明：{row['caption']}\n"
    if variant.startswith("one"):
        body = VOICE_ONE + (KNOW_TYPED if "_kt" in variant else KNOW_ONE)
    else:
        body = (VOICE_DEC2 if variant.startswith("dec2") else VOICE_DEC) + KNOW_DEC
    if "fs" in variant:
        body += FEWSHOT
    return HEAD.format(fmt=T13.FORMAT_JA[fmt], extra=extra, boke=(text or "").replace("\n", " ")[:300]) + body + TAIL


def derive(variant: str, rec: dict, fmt: str) -> dict:
    """回答を発話の型（6区分）と画像外の知識にする．

    Args:
        variant (str): 方法．
        rec (dict): モデルの回答．
        fmt (str): お題の形式．

    Returns:
        dict: voice，outside と，分解した問いの回答．
    """
    if "_kt" in variant:
        voice = rec.get("voice", "S")
        outside = rec.get("referent_type") in SPECIFIC_TYPES and not rec.get("in_image", True) and bool(rec.get("necessary"))
    elif variant.startswith("one"):
        voice, outside = rec.get("voice", "S"), bool(rec.get("outside", False))
    else:
        if rec.get("is_naming"):
            voice = "L"
        elif rec.get("speaker_in_image", True):
            voice = "S" if rec.get("aloud", True) else "M"
        else:
            voice = "N" if rec.get("narration") else "T"
        outside = bool((rec.get("referent") or "").strip()) and bool(rec.get("specific")) and not rec.get("in_image", True) \
            and bool(rec.get("necessary"))
    out = {"voice": "A" if fmt in ("anaume", "odaibun") else voice, "outside": outside}
    if "_kt" in variant:
        out.update({k: rec.get(k) for k in ["referent", "referent_type", "in_image", "necessary"]})
    elif not variant.startswith("one"):
        out.update({k: rec.get(k) for k in ["speaker", "referent", "is_naming", "speaker_in_image", "aloud", "narration", "specific",
                                           "in_image", "necessary"]})
    return out


def label(llm, df: pl.DataFrame, variant: str, chunk: int = 2048) -> tuple:
    """ボケの表にラベルを付ける．

    Args:
        llm (vllm.LLM): モデル．
        df (pl.DataFrame): 列 odaiId, text, format, has_text, 媒体, caption を持つ表．
        variant (str): 方法．
        chunk (int): 一度に推論する件数．

    Returns:
        tuple[pl.DataFrame, float]: ラベルの表と推論の秒数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = SCHEMA_KT if "_kt" in variant else SCHEMA_ONE if variant.startswith("one") else SCHEMA_DEC
    px = 896 * 896 if variant.endswith("_hr") else 448 * 448
    params = SamplingParams(temperature=0, max_tokens=256, structured_outputs=StructuredOutputsParams(json=schema))
    recs, sec, rows = [], 0.0, df.to_dicts()
    with ThreadPoolExecutor(32) as ex:
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            uniq = sorted({r["odaiId"] for r in part})
            imgs = dict(zip(uniq, ex.map(T13.load_image, uniq)))
            convs = [[{"role": "user", "content": [
                {"type": "image_pil", "image_pil": imgs[r["odaiId"]]},
                {"type": "text", "text": build_prompt(variant, r["format"], r["text"], r)}]}] for r in part]
            start = time.perf_counter()
            outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": px})
            sec += time.perf_counter() - start
            for r, o in zip(part, outs):
                try:
                    rec = json.loads(o.outputs[0].text)
                except json.JSONDecodeError:
                    rec = {}
                recs.append(derive(variant, rec, r["format"]))
    return pl.DataFrame(recs, infer_schema_length=None), sec


def load_extra() -> pl.DataFrame:
    """ex013 の学習用のボケから，aug と重なるものと検証データのお題のものを除いた表を返す．

    Returns:
        pl.DataFrame: 列 train_index, odaiId, text, format を持つ表．
    """
    t = pl.read_parquet(os.path.join(EX13_DATASET, "train_pool.parquet")).with_row_index("train_index")
    aug = pl.read_csv(os.path.join(EX14_DATASET, "aug_pool.csv")).select("odaiId", "text")
    t = t.join(aug, on=["odaiId", "text"], how="anti")
    return t.filter(~pl.col("odaiId").is_in(load_eval()["odaiId"].unique().to_list())).select(
        pl.col("train_index").cast(pl.Int64), "odaiId", "text", "format")


def load_set(name: str) -> pl.DataFrame:
    """集合の表を，お題のラベルと結合して読み込む．

    Args:
        name (str): eval / aug．

    Returns:
        pl.DataFrame: 識別の列，odaiId, text, format, has_text, 媒体, caption を持つ表．
    """
    cols = ["odaiId", "format", "has_text", "caption"] + list(MEDIA_JA)
    if name == "eval":
        return load_eval().select(["no", "text"] + cols)
    if name == "extra":
        return load_extra().drop("format").join(load_odai_labels(None).select(cols), on="odaiId", how="left", maintain_order="left")
    pool = pl.read_csv(os.path.join(EX14_DATASET, "aug_pool.csv"))
    return pool.drop("format").join(load_odai_labels(pool["odaiId"].unique().to_list()).select(cols), on="odaiId", how="left")


def main() -> None:
    """指定したモデルと方法で指定した集合にラベルを付ける．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(T13.MODELS))
    parser.add_argument("--sets", nargs="+", default=["eval"])
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()
    llm = T13.build_llm(args.model)
    for name in args.sets:
        df = load_set(name)
        for variant in args.variants:
            out = os.path.join(OUTPUT_DIR, f"teacher_{args.model}", variant)
            path = os.path.join(out, f"{name}.csv")
            if os.path.exists(path):
                continue
            os.makedirs(out, exist_ok=True)
            lab, sec = label(llm, df, variant)
            keep = [c for c in df.columns if c not in ("text", "format", "has_text", "caption") and c not in MEDIA_JA]
            pl.concat([df.select(keep), lab], how="horizontal").write_csv(path)
            json.dump({"n": df.height, "sec": sec, "per_sec": df.height / sec}, open(path.replace(".csv", "_speed.json"), "w"))
            print(args.model, variant, name, f"{df.height / sec:.1f} 件/秒", flush=True)


if __name__ == "__main__":
    main()
