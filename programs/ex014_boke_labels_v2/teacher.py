"""VLM の教師（第3版の指示）で，ボケにラベルを付ける．

第3版の指示（ex013 の第2版からの変更）：
- 性的・不適切を除く．
- 必要な知識を，上から順に判定する手順として問う：見た目だけで分かるか → 一般常識を加えれば分かるか → 画像中の対象の情報が要るか →
  （いずれでもなければ）画像外の情報．回答は4つの問いの真偽で受け，最初に真となった段階とする．
- 発話の型は第2版と同じ5種（回答は形式から規則で決める）．

集合（--sets）：
- eval：検証データ620件
- swap：穴埋め・お題文の制約の順守の検証．検証データの穴埋め・お題文のボケを，同じ形式の別のお題に付け替えた組
  （datasets/ex014_boke_labels_v2/swap_pairs.csv の odaiId は付け替え先）
- aug：発話の型の水増し用（make_aug_pool.py で作る．無作為の5千件と少ない型の候補5千件）

出力：outputs/ex014_boke_labels_v2/teacher_{モデル}_v3/{集合}.csv，{集合}_speed.json

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/teacher.py --model q32 --sets eval swap aug
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, EX13_DIR, OUTPUT_DIR, _load, load_eval  # noqa: E402

T13 = _load("ex013_teacher_for_v3", os.path.join(EX13_DIR, "teacher.py"))
PROMPT_V3 = """これは大喜利サイト「ボケて」のお題画像と，それに投稿されたボケです．
お題の形式：{fmt}
ボケ：「{boke}」

次の項目を判定し，JSON のみを出力してください．
1. complies：お題の形式の制約に従っているか．カルタは札のひらがなと同じ音で始まるか（漢字・カタカナは読みで判断），穴埋めは空欄に入る語句として
   意味が通るか，お題文は画像中の問いや指示に応じた答えになっているか．形式が通常なら true．お題と無関係な文（宣伝など）は false．
2. voice：誰の視点の言葉か．
   - S：画像の中の人物・動物・物が，口に出して言っている言葉（誰かへの話しかけ，叫び，掛け合い）．例：犬の写真に「散歩まだ？」
   - M：画像の中の人物などが心の中で思っていること・独り言．例：寝ている猫に「あと5分だけ…」
   - T：画像を見ている人が，画像について言うツッコミや感想．画像の中の人物の言葉ではない．例：大きな帽子の人に「帽子でかすぎだろ」
   - N：語り手が，画像の状況や前後の出来事を説明する文．例：転んだ人に「バナナの皮を踏んだ直後」
   - L：画像や写っている物に，名前・題・例えを付ける言葉（名詞で終わることが多い）．例：丸い猫に「もち」，「〜の時の俺」
3. ボケの面白さを理解するのに必要な知識を，次の問いに上から順に答えて判定する．
   k1_visual：画像の見た目（表情・姿勢・状況）とボケの字面だけで面白さが分かるか．
   k2_common：（k1 が偽の場合）ことわざ・慣用句・ダジャレの元の語・日常の事物や行事など，誰もが知る一般常識を加えれば分かるか．
   k3_in_image：（k1, k2 が偽の場合）画像に写っている，または画像中の文字で名指しされた特定の人物・キャラクター・作品・商品の情報が必要か．
   k4_outside：（k1〜k3 が偽の場合）画像に写っていない特定の作品・番組・歌・CM・人物・地名・時事の知識が必要か．
4. uses_image_text：ボケの意味が，画像中の文字（字幕，看板，漫画の台詞，説明文，札の文字そのもの）を読まないと分からないか．"""
SCHEMA_V3 = {"type": "object", "properties": {
    "complies": {"type": "boolean"}, "voice": {"type": "string", "enum": ["S", "M", "T", "N", "L"]},
    "k1_visual": {"type": "boolean"}, "k2_common": {"type": "boolean"}, "k3_in_image": {"type": "boolean"},
    "k4_outside": {"type": "boolean"}, "uses_image_text": {"type": "boolean"}},
    "required": ["complies", "voice", "k1_visual", "k2_common", "k3_in_image", "k4_outside", "uses_image_text"],
    "additionalProperties": False}


def derive_v3(rec: dict, fmt: str) -> dict:
    """第3版の回答を4項目のラベルにする．知識は上から順に最初に真となった段階とし，全て偽なら画像外とする．

    Args:
        rec (dict): モデルの回答．
        fmt (str): お題の形式．

    Returns:
        dict: complies, voice, knowledge, uses_image_text．
    """
    know = "O"
    for k, lab in [("k1_visual", "0"), ("k2_common", "G"), ("k3_in_image", "I")]:
        if rec.get(k):
            know = lab
            break
    qa = fmt in ("anaume", "odaibun")
    return {"complies": bool(rec.get("complies", True)), "voice": "A" if qa else rec.get("voice", "S"), "knowledge": know,
            "uses_image_text": True if qa else bool(rec.get("uses_image_text", False))}


def label(llm, df: pl.DataFrame, chunk: int = 2048) -> tuple:
    """ボケの表に第3版の指示でラベルを付ける．

    Args:
        llm (vllm.LLM): モデル．
        df (pl.DataFrame): 列 odaiId, text, format を持つ表．
        chunk (int): 一度に推論する件数．

    Returns:
        tuple[pl.DataFrame, float]: ラベルの表と推論の秒数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    params = SamplingParams(temperature=0, max_tokens=120, structured_outputs=StructuredOutputsParams(json=SCHEMA_V3))
    recs, sec = [], 0.0
    rows = df.select("odaiId", "text", "format").rows()
    with ThreadPoolExecutor(32) as ex:
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            uniq = sorted({r[0] for r in part})
            imgs = dict(zip(uniq, ex.map(T13.load_image, uniq)))
            convs = [[{"role": "user", "content": [
                {"type": "image_pil", "image_pil": imgs[fid]},
                {"type": "text", "text": PROMPT_V3.format(fmt=T13.FORMAT_JA[fmt], boke=(text or "").replace("\n", " ")[:300])}]}]
                for fid, text, fmt in part]
            start = time.perf_counter()
            outs = llm.chat(convs, sampling_params=params, use_tqdm=False, mm_processor_kwargs={"max_pixels": 448 * 448})
            sec += time.perf_counter() - start
            for (fid, text, fmt), o in zip(part, outs):
                try:
                    rec = json.loads(o.outputs[0].text)
                except json.JSONDecodeError:
                    rec = {}
                recs.append(derive_v3(rec, fmt))
    return pl.DataFrame(recs), sec


def load_set(name: str) -> pl.DataFrame:
    """集合の表を読み込む．

    Args:
        name (str): eval / swap / aug．

    Returns:
        pl.DataFrame: 列 odaiId, text, format と識別の列を持つ表．
    """
    if name == "eval":
        return load_eval().select("no", "odaiId", "text", "format")
    return pl.read_csv(os.path.join(DATASET_DIR, f"{name}_pairs.csv" if name == "swap" else f"{name}_pool.csv"))


def main() -> None:
    """指定したモデルで指定した集合にラベルを付ける．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(T13.MODELS))
    parser.add_argument("--sets", nargs="+", default=["eval"])
    args = parser.parse_args()
    out = os.path.join(OUTPUT_DIR, f"teacher_{args.model}_v3")
    os.makedirs(out, exist_ok=True)
    llm = T13.build_llm(args.model)
    for name in args.sets:
        path = os.path.join(out, f"{name}.csv")
        if os.path.exists(path):
            continue
        df = load_set(name)
        lab, sec = label(llm, df)
        pl.concat([df.drop("text"), lab], how="horizontal").write_csv(path)
        json.dump({"n": df.height, "sec": sec, "per_sec": df.height / sec}, open(path.replace(".csv", "_speed.json"), "w"))
        print(args.model, name, f"{df.height / sec:.1f} 件/秒", flush=True)


if __name__ == "__main__":
    main()
