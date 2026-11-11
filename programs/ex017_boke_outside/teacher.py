"""教師（Qwen3-VL-32B）の画像外の知識の判定を，指示の変更で改善する．

入力は ex015 の one_kt_cap と同じ（お題画像，形式，媒体，画像中の文字の有無，写真のキャプション，ボケの文）で，画像外の知識のみを問う．
方法：
- kt：ex015 の種類つきの問い（依存する事物 → 種類 → 画像内か → 必要か）．
- src：先に「元ネタ」（特定の作品・人物・台詞・歌詞・CM・流行語・ネットの定型などの引用・もじり・パロディ・連想）を具体的に書かせ，
  その種類・画像内か・必要かを問い，最後に判定を書かせる．report_023 で多かった誤り（定番の台詞や歌詞の引用に気づかない，
  固有名詞が出てくるだけで「はい」とする）への対策．
- src_fs：src に，検証データとは別に作った例（文で書いた画像とボケ，判定と理由）を加える．
- src_sc：src を温度0.7で5回生成し，「はい」の割合を確率とする（0.5以上を「はい」）．

集合（--sets）：labeled（目視のラベルのある1,551件），train（教師のラベルの学習データ34,634件）

出力：outputs/ex017_boke_outside/teacher/{方法}/{集合}.csv（列 key または行番号，outside，p_outside，source など），speed.json

実行例:
    .venv_annotate/bin/python programs/ex017_boke_outside/teacher.py --sets labeled --variants kt src src_fs src_sc
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, PROJECT_DIR, _load, load_labeled, load_odai_labels, load_train  # noqa: E402

# ex015 の teacher.py は `from data import ...` で ex015 の data を読むため，読み込む間だけ差し替える
_own = sys.modules["data"]
sys.modules["data"] = _load("ex015_data_for_v5", os.path.join(PROJECT_DIR, "programs", "ex015_boke_labels_v3", "data.py"))
T15 = _load("ex015_teacher_for_v5", os.path.join(PROJECT_DIR, "programs", "ex015_boke_labels_v3", "teacher.py"))
sys.modules["data"] = _own

VARIANTS = ["kt", "src", "src_fs", "src_sc"]
TYPES = T15.SPECIFIC_TYPES + ["general", "none"]
KNOW_SRC = """
ボケの面白さを理解するのに，画像にも画像中の文字にも示されていない，特定の事物の知識が必要か（画像外の知識）を，次の順に答える．
   source：このボケが，特定の作品・キャラクター・実在の人物・団体・商品・番組・CM・歌・映画・ゲーム・有名な台詞やキャッチコピー・
           流行語・ネットの定型・ニュースや事件・地名の印象・ボケての機能や慣習を，引用・もじり・パロディ・連想で使っているなら，
           その元ネタを具体的に書く（例：「ドラゴンクエストの『へんじがない。ただのしかばねのようだ』」「サンドウィッチマンのネタ」）．
           台詞や歌詞の一部をそのまま使っている場合や，言い回しだけを真似ている場合も含める．なければ空文字．
   source_type：元ネタの種類を次から1つ選ぶ．固有名詞や特定の引用でなければ general とする．
     work（作品・キャラクター・番組・映画・ゲーム・歌・CM・有名な台詞やキャッチコピー），person（実在の人物・団体），
     product（商品・ブランド・店），place（地名と土地の印象），event（出来事・ニュース・流行語・ネットの定型），bokete（ボケての機能や慣習），
     general（普通名詞，一般の物・行事・遊び・職業・ジャンル，ことわざ・慣用句），none（なし）
   in_image：元ネタが画像に写っている，画像中の文字で名指しされている，または画像に写っているキャラクターと同じ作品のものか．
   necessary：元ネタを知らない人には，一般の日本語の語彙と常識があってもボケの面白さが分からないか．
              固有名詞が出てくるだけで，知らなくても面白さが伝わるなら false．
   outside：以上を踏まえて，画像外の知識が必要か（source_type が固有の種類で，in_image が false で，necessary が true のときに true）．
"""
FEWSHOT = """
例（画像は文で示す）：
- 画像：倒れて動かない男性．ボケ「へんじがない。ただのしかばねのようだ」→ source=ドラゴンクエストの定型文，work，in_image=false，
  necessary=true，outside=true
- 画像：困惑した顔の男性．ボケ「ちょっと何言ってるかわからない」→ source=サンドウィッチマン富澤のネタ，person，in_image=false，
  necessary=true，outside=true（知らなくても文は分かるが，面白さは元ネタに依存する）
- 画像：黒板の前で指差す講師．ボケ「いつやるか？今でしょ！」→ source=林修の流行語，event，in_image=false，necessary=true，outside=true
- 画像：ドラえもんのイラスト．ボケ「どら焼き買ってきて」→ source=ドラえもん，work，in_image=true，outside=false
- 画像：泣いている子ども．ボケ「ディズニーランドで迷子になった」→ source=ディズニーランド，place，necessary=false，outside=false
- 画像：雪の中で体操をする人々．ボケ「ラジオ体操第四」→ source=ラジオ体操，general，outside=false
- 画像：怒った顔の猫．ボケ「月曜日の朝の俺」→ source=空，none，outside=false
"""
SCHEMA_SRC = {"type": "object", "properties": {
    "source": {"type": "string", "maxLength": 60}, "source_type": {"type": "string", "enum": TYPES},
    "in_image": {"type": "boolean"}, "necessary": {"type": "boolean"}, "outside": {"type": "boolean"}},
    "required": ["source", "source_type", "in_image", "necessary", "outside"], "additionalProperties": False}
SCHEMA_KT = {"type": "object", "properties": {
    "referent": {"type": "string", "maxLength": 40}, "referent_type": {"type": "string", "enum": TYPES},
    "in_image": {"type": "boolean"}, "necessary": {"type": "boolean"}},
    "required": ["referent", "referent_type", "in_image", "necessary"], "additionalProperties": False}


def build_prompt(variant: str, row: dict) -> str:
    """方法に応じた指示の文を作る．

    Args:
        variant (str): VARIANTS のいずれか．
        row (dict): 列 text, format, has_text, caption, 媒体 を持つ行．

    Returns:
        str: 指示の文．
    """
    media = "，".join(T15.MEDIA_JA[m] for m in T15.MEDIA_JA if row.get(m)) or "不明"
    extra = f"お題画像の媒体：{media}．画像中の文字：{'あり' if row.get('has_text') else 'なし'}\n"
    if row.get("caption"):
        extra += f"お題画像の説明：{row['caption']}\n"
    head = T15.HEAD.format(fmt=T15.T13.FORMAT_JA[row["format"]], extra=extra, boke=(row["text"] or "").replace("\n", " ")[:300])
    if variant == "kt":
        body = T15.KNOW_TYPED.replace("2. 画像外の知識を", "画像外の知識を")
    else:
        body = KNOW_SRC + (FEWSHOT if variant == "src_fs" else "")
    return head + body + T15.TAIL


def derive(variant: str, recs: list) -> dict:
    """回答（複数回の生成を含む）を画像外の知識の判定にする．

    Args:
        variant (str): 方法．
        recs (list[dict]): 生成ごとの回答．

    Returns:
        dict: outside（bool），p_outside（float），先頭の回答の各項目．
    """
    flags = []
    for r in recs:
        if variant == "kt":
            flags.append(r.get("referent_type") in T15.SPECIFIC_TYPES and not r.get("in_image", True) and bool(r.get("necessary")))
        else:
            flags.append(bool(r.get("outside")) and r.get("source_type") in T15.SPECIFIC_TYPES and not r.get("in_image", True))
    p = sum(flags) / max(len(flags), 1)
    first = recs[0] if recs else {}
    out = {"outside": p >= 0.5, "p_outside": p}
    out.update({k: first.get(k) for k in (["referent", "referent_type"] if variant == "kt" else ["source", "source_type"])})
    return out


def label(llm, df: pl.DataFrame, variant: str, chunk: int = 2048) -> tuple:
    """表の各行に画像外の知識の判定を付ける．

    Args:
        llm (vllm.LLM): モデル．
        df (pl.DataFrame): 列 odaiId, text, format, has_text, caption, 媒体 を持つ表．
        variant (str): 方法．
        chunk (int): 一度に推論する件数．

    Returns:
        tuple[pl.DataFrame, float]: 判定の表と推論の秒数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = SCHEMA_KT if variant == "kt" else SCHEMA_SRC
    so = StructuredOutputsParams(json=schema)
    if variant == "src_sc":
        params = SamplingParams(temperature=0.7, top_p=0.95, n=5, seed=0, max_tokens=200, structured_outputs=so)
    else:
        params = SamplingParams(temperature=0, max_tokens=200, structured_outputs=so)
    recs, sec, rows = [], 0.0, df.to_dicts()
    with ThreadPoolExecutor(32) as ex:
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            uniq = sorted({r["odaiId"] for r in part})
            imgs = dict(zip(uniq, ex.map(T15.T13.load_image, uniq)))
            convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": imgs[r["odaiId"]]},
                                                   {"type": "text", "text": build_prompt(variant, r)}]}] for r in part]
            start = time.perf_counter()
            outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": 448 * 448})
            sec += time.perf_counter() - start
            for o in outs:
                got = []
                for c in o.outputs:
                    try:
                        got.append(json.loads(c.text))
                    except json.JSONDecodeError:
                        pass
                recs.append(derive(variant, got))
    return pl.DataFrame(recs, infer_schema_length=None), sec


def load_set(name: str) -> pl.DataFrame:
    """集合の表を読み込む．

    Args:
        name (str): labeled / train．

    Returns:
        pl.DataFrame: 識別の列（key / row）と，指示に必要な列を持つ表．
    """
    if name == "labeled":
        return load_labeled().select("key", "odaiId", "text", "format", "has_text", "caption", *T15.MEDIA_JA)
    return load_train().with_row_index("row").select(pl.col("row").cast(pl.Int64), "odaiId", "text", "format", "has_text", "caption",
                                                     *T15.MEDIA_JA)


def main() -> None:
    """指定した方法で指定した集合に判定を付ける．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=["labeled"])
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()
    llm = T15.T13.build_llm("q32")
    for name in args.sets:
        df = load_set(name)
        for variant in args.variants:
            out = os.path.join(OUTPUT_DIR, "teacher", variant)
            path = os.path.join(out, f"{name}.csv")
            if os.path.exists(path):
                continue
            os.makedirs(out, exist_ok=True)
            lab, sec = label(llm, df, variant)
            pl.concat([df.select(df.columns[0]), lab], how="horizontal").write_csv(path)
            json.dump({"n": df.height, "sec": sec, "per_sec": df.height / sec}, open(path.replace(".csv", "_speed.json"), "w"))
            print(variant, name, f"{df.height / sec:.1f} 件/秒", flush=True)


if __name__ == "__main__":
    main()
