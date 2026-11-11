"""Qwen3-VL-32B に文の形を問う教師（指示 FORM：層を問わない，FORM2：文ごとに場所と形を問う）．

写っている世界の中の文字（看板，商品，画面など）と吹き出しの文字を除いた文字（写真や絵に重ねた文字と，写真や絵の外の文字）を
文ごとに書き起こし，文の形（空欄を含む文，問い，指示，題目の句，その他の文）を選ばせる．
画像ごとの形の有無（T_{形}）は，いずれかの文がその形である場合に真とする．

指示：
- FORM：対象の文字だけを文ごとに書き起こし，形を選ばせる（層は問わない）．
- FORM2：画像の中の読める文字を全て文ごとに書き起こし，文ごとに場所（場面，吹き出し，重ねた文字，余白の文字）と形を選ばせる．
  FORM では場面の文字と吹き出しの文字を対象から除けなかったため（order_047 の 1,620 枚での確認），文ごとに場所を答えさせる．
  T_{形} は重ねた文字か余白の文字の文に限り，F_o_{形}，F_m_{形}（層ごとの形）も記録する．

出力：outputs/ex027_odai_text_improve/qwen/{指示}_{対象}.csv（列 rep_odaiId, ok, T_k, T_q, T_i, T_t, T_x, text, raw．FORM2 は F_o_*，F_m_*，L_o，L_m も）と speed.json
対象：eval（評価の1,620枚），pool（教師ラベルを付けるプール．チャンクごとに保存し，途中から再開できる）

実行例:
    .venv_annotate/bin/python programs/ex027_odai_text_improve/qwen_form.py --prompt FORM2 --target eval
    .venv_annotate/bin/python programs/ex027_odai_text_improve/qwen_form.py --prompt FORM2 --target pool --n_pool 30000
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EX026, FORMS, OUTPUT_DIR, PROJECT_DIR, build_pool, image_path  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0"))
from bokete_annotation import verify  # noqa: E402
from bokete_annotation.data import load_rgb  # noqa: E402

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
MAX_PIXELS = 802816
CHUNK = 2000
FORM_ENUM = {"blank_sentence": "k", "question": "q", "instruction": "i", "topic": "t", "other": "x"}
FORM = """この画像は大喜利サイト「ボケて」のお題画像です．画像の中の読める文字のうち，次の文字を対象にします．
- 対象：写真や絵の上に重ねた文字（テレビの字幕やテロップ，写真への書き込み，半透明の帯の文字）と，写真や絵の外にある文字（上下左右の帯や枠，無地の背景，テンプレートの文字欄の文字）．文字が主体の画像の文字も対象です．
- 対象外：写っている世界の中の文字（看板，ポスター，商品や衣服の表示，本の表紙，黒板，写真に写った画面，スクリーンショットの画面の中の文字，効果音），漫画やイラストの吹き出しと人物の脇に描き込まれた台詞，透かし，ロゴ，署名，著作権表示，日付，ボケてのテンプレートに印刷された定型の見出し（「写真で一言ボケて」）．
誰が書いたか，元の素材の文字か投稿者が加えた文字かは問いません．見た目だけで判断してください．

対象の文字を文ごとに書き起こし（text．40文字以内に省略してよい．最大6文），それぞれの文の形を1つ選んでください（form）．
文の形：
- blank_sentence：空欄（□，○○，＿，語の代わりの？，空の枠など）を含む文．空欄を含む文章のまとまり（段落，説明文，字幕，歌詞，台詞，台本）の中の文は，空欄がなくてもこれにします．
- question：疑問の形の文（〜とは？，なぜ〜？，どっち？），「もし〜たら」「〜したら」の条件の文．
- instruction：依頼や命令の形の文（〜してください，〜を教えて，一言ボケて）．
- topic：大喜利の題となる句（〜あるある，〜な事，〜の共通点，こんな〜は嫌だ，〜する方法，〜ランキング）．
- other：上のどれでもない文や語句（平叙文，台詞，感想，状況の説明，題名，名前，名札だけの見出し）．
対象の文字がない場合は sentences を空の配列にします．JSON のみを出力してください．"""
SCHEMA = {"type": "object", "properties": {"sentences": {"type": "array", "maxItems": 6, "items": {
    "type": "object", "properties": {"text": {"type": "string", "maxLength": 40}, "form": {"type": "string", "enum": list(FORM_ENUM)}},
    "required": ["text", "form"], "additionalProperties": False}}}, "required": ["sentences"], "additionalProperties": False}
WHERE_ENUM = {"scene": "s", "bubble": "b", "overlay": "o", "margin": "m"}
FORM2 = """この画像は大喜利サイト「ボケて」のお題画像です．画像の中の読める文字を全て，文ごとに書き起こしてください（text．40文字以内に省略してよい．最大8文．文字が多い場合は目立つ文を優先します）．
誰が書いたか，元の素材の文字か投稿者が加えた文字かは問いません．見た目だけで判断してください．
それぞれの文について，文字のある場所（where）と文の形（form）を1つずつ選んでください．

場所：
- scene：写っている世界の中の文字．看板，ポスター，商品や衣服の表示，本の表紙，黒板，写真に写った画面，ウェブ・SNS・アプリ・ゲームのスクリーンショットの画面の中の文字，絵に描き込まれた効果音．透かし，ロゴ，署名，著作権表示，日付，ボケてのテンプレートに印刷された定型の見出し（「写真で一言ボケて」）も scene にします．
- bubble：漫画やイラストの吹き出しと語りの枠の中の文字．吹き出しのない，絵の中の人物の脇に描き込まれた台詞も bubble にします．
- overlay：写真や絵の上に重ねた文字．テレビの字幕やテロップ，写真への書き込み，半透明の帯の文字．
- margin：写真や絵の外にある文字．上下左右の帯や枠，無地の背景，テンプレートの文字欄の文字．文字が主体の画像（スクリーンショットを除く）の文字も margin にします．

文の形：
- blank_sentence：空欄（□，○○，＿，語の代わりの？，空の枠など）を含む文．空欄を含む文章のまとまり（段落，説明文，字幕，歌詞，台詞，台本）の中の文は，空欄がなくてもこれにします．
- question：疑問の形の文（〜とは？，なぜ〜？，どっち？），「もし〜たら」「〜したら」の条件の文．
- instruction：依頼や命令の形の文（〜してください，〜を教えて，一言ボケて）．
- topic：大喜利の題となる句（〜あるある，〜な事，〜の共通点，こんな〜は嫌だ，〜する方法，〜ランキング）．
- other：上のどれでもない文や語句（平叙文，台詞，感想，状況の説明，題名，名前，名札だけの見出し）．
読める文字がない場合は sentences を空の配列にします．JSON のみを出力してください．"""
SCHEMA2 = {"type": "object", "properties": {"sentences": {"type": "array", "maxItems": 8, "items": {
    "type": "object", "properties": {"text": {"type": "string", "maxLength": 40}, "where": {"type": "string", "enum": list(WHERE_ENUM)},
                                     "form": {"type": "string", "enum": list(FORM_ENUM)}},
    "required": ["text", "where", "form"], "additionalProperties": False}}}, "required": ["sentences"], "additionalProperties": False}


def parse_form(text: str) -> dict:
    """FORM の出力を形の有無と書き起こしに変える．

    Args:
        text (str): モデルの出力（JSON の文字列）．

    Returns:
        dict: キー ok（解釈できたか），T_{形}（いずれかの文がその形か），text（文を「／」で結んだ書き起こし）．
    """
    rec = {"ok": False, **{f"T_{f}": False for f in FORMS}, "text": ""}
    try:
        sents = json.loads(text)["sentences"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return rec
    sents = [s for s in sents if s["text"].strip()]
    rec.update(ok=True, text="／".join(s["text"] for s in sents))
    for s in sents:
        rec[f"T_{FORM_ENUM[s['form']]}"] = True
    return rec


def parse_form2(text: str) -> dict:
    """FORM2 の出力を，重ねた文字と余白の文字の文に限った形の有無と，層ごとの形・有無に変える．

    Args:
        text (str): モデルの出力（JSON の文字列）．

    Returns:
        dict: キー ok，T_{形}，F_o_{形}，F_m_{形}，L_{s,b,o,m}，text（重ねた文字と余白の文字の文を「／」で結んだもの），text_all（全ての文）．
    """
    rec = {"ok": False, **{f"T_{f}": False for f in FORMS}, **{f"F_{l}_{f}": False for l in "om" for f in FORMS},
           **{f"L_{l}": False for l in "sbom"}, "text": "", "text_all": ""}
    try:
        sents = json.loads(text)["sentences"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return rec
    sents = [s for s in sents if s["text"].strip()]
    added = [s for s in sents if WHERE_ENUM[s["where"]] in "om"]
    rec.update(ok=True, text="／".join(s["text"] for s in added), text_all="／".join(s["text"] for s in sents))
    for s in sents:
        rec[f"L_{WHERE_ENUM[s['where']]}"] = True
    for s in added:
        f, l = FORM_ENUM[s["form"]], WHERE_ENUM[s["where"]]
        rec[f"T_{f}"] = True
        rec[f"F_{l}_{f}"] = True
    return rec


PROMPTS = {"FORM": (FORM, SCHEMA, parse_form), "FORM2": (FORM2, SCHEMA2, parse_form2)}


def run(llm, ids: list, prompt: str) -> tuple:
    """画像のリストに指示を問う．

    Args:
        llm (vllm.LLM): モデル．
        ids (list[int]): 代表の画像のお題IDのリスト．
        prompt (str): 指示の名前（FORM / FORM2）．

    Returns:
        tuple[pl.DataFrame, float, int]: 結果の表，推論の秒数，生成したトークンの総数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    text, schema, parse = PROMPTS[prompt]
    params = SamplingParams(temperature=0, max_tokens=768, structured_outputs=StructuredOutputsParams(json=schema))
    with ThreadPoolExecutor(32) as ex:
        images = list(ex.map(load_rgb, [image_path(i) for i in ids]))
    convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im}, {"type": "text", "text": text}]}] for im in images]
    start = time.perf_counter()
    outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": MAX_PIXELS})
    sec = time.perf_counter() - start
    rows = [{"rep_odaiId": i, **parse(o.outputs[0].text), "raw": o.outputs[0].text} for i, o in zip(ids, outs)]
    return pl.DataFrame(rows), sec, sum(len(o.outputs[0].token_ids) for o in outs)


def target_ids(target: str, n_pool: int) -> list:
    """対象の画像のお題IDを求める．

    Args:
        target (str): eval か pool．
        n_pool (int): プールの画像の数．

    Returns:
        list[int]: 代表の画像のお題IDのリスト．
    """
    ev = pl.read_csv(EX026.TARGETS)["rep_odaiId"].to_list()
    if target == "eval":
        return ev
    return build_pool(n_pool, ev)["rep_odaiId"].to_list()


def main() -> None:
    """対象の画像に指示を問い，チャンクごとに結果を保存する（保存済みのチャンクは飛ばす）．最後に1つの CSV にまとめる．

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=list(PROMPTS), default="FORM2")
    ap.add_argument("--target", choices=["eval", "pool"], default="eval")
    ap.add_argument("--n_pool", type=int, default=30000)
    args = ap.parse_args()
    out_dir = os.path.join(OUTPUT_DIR, "qwen")
    name = f"{args.prompt}_{args.target}"
    part_dir = os.path.join(out_dir, f"parts_{name}")
    os.makedirs(part_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(out_csv):
        print("skip: 結果がある", out_csv)
        return
    ids = target_ids(args.target, args.n_pool)
    chunks = [ids[k:k + CHUNK] for k in range(0, len(ids), CHUNK)]
    todo = [k for k in range(len(chunks)) if not os.path.exists(os.path.join(part_dir, f"part_{k:04d}.csv"))]
    speed_path = os.path.join(out_dir, "speed.json")
    speed = json.load(open(speed_path)) if os.path.exists(speed_path) else {}
    if todo:
        llm = verify.build_llm(MODEL, MAX_PIXELS)
        sec_sum, tok_sum, n_sum = 0.0, 0, 0
        for k in tqdm(todo, desc=name, leave=False):
            df, sec, n_tok = run(llm, chunks[k], args.prompt)
            df.write_csv(os.path.join(part_dir, f"part_{k:04d}.csv"))
            sec_sum, tok_sum, n_sum = sec_sum + sec, tok_sum + n_tok, n_sum + len(chunks[k])
            print(f"chunk {k}: {len(chunks[k]) / sec:.2f} img/s, parse_fail={int((~df['ok']).sum())}", flush=True)
        speed[name] = {"n": n_sum, "sec": sec_sum, "img_per_sec": n_sum / sec_sum, "tokens": tok_sum}
        json.dump(speed, open(speed_path, "w"), ensure_ascii=False, indent=1)
    df = pl.concat([pl.read_csv(p, schema_overrides={"text": pl.Utf8, "text_all": pl.Utf8, "raw": pl.Utf8})
                    for p in sorted(glob.glob(os.path.join(part_dir, "part_*.csv")))])
    assert df.height == len(ids), "チャンクの結果が欠けている"
    df.write_csv(out_csv)
    print(name, df.height, "parse_fail", int((~df["ok"]).sum()), speed.get(name))


if __name__ == "__main__":
    main()
