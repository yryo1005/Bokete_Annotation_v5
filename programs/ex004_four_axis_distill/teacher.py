"""確定版の定義によるプロンプト v3 で，Qwen3-VL-8B に4軸のラベルを付けさせる（教師）．

プロンプトは定義書（programs/label_guideline/label_definition_v1.md）の判定手順の順に質問を並べ，
出力は JSON スキーマで制約する．v4 は v3 の誤り（文中のひらがなをカルタ記号とする，作品の知識を過剰に付ける）を
受けて，属性を先に答えさせて形式を属性から決めさせ，作品名と人物名を書かせる版である．karuta_char には形式がカルタの場合のひらがなを出力させる．
出力の後処理として，形式から必ず真になる属性を補い，媒体が1つも選ばれなかった場合はイラストとする．

- 評価：outputs/ex004_four_axis_distill/teacher_{モデル}/{プロンプト}/0/
- ラベル：datasets/ex004_four_axis_distill/teacher_labels/{モデル}_{プロンプト}.csv

実行例:
    .venv_annotate/bin/python programs/ex004_four_axis_distill/teacher.py --label_pool
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, is_done, save_run, throughput  # noqa: E402
from data import (ATTRIBUTES, DATASET_DIR, FORMAT_IMPLIES, FORMATS, KNOWLEDGE, MEDIA, image_path,  # noqa: E402
                  load_eval_labels, load_pool)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 448 * 448
CHUNK = 2048

PROMPT_V3 = """この画像は大喜利サイト「ボケて」のお題画像です．次の手順で4つの軸を判定し，JSON のみを出力してください．

## 手順1：形式（format，1つ選択）
ユーザーの回答を制限する画像かを，以下の順に判定し，最初に当てはまったものを選ぶ．
1. karuta（カルタ）：ひらがな1文字が入った記号（○，□などの枠）がある．ひらがなは文の一部ではなく独立した枠の中にあるものに限る．空欄も含む場合もカルタとする．karuta_char にそのひらがなを書く．
2. anaume（穴埋め）：語の代わりに置かれた空欄がある（白や色付きの矩形，○○，□，＿＿，語の代わりの「？」，語の代わりの黒塗り，話者に付いた空の吹き出し）．顔や個人情報の目隠し，文末の疑問符は空欄ではない．お題文も含む場合も穴埋めとする．
3. odaibun（お題文）：投稿者が書き加えた問いや指示の文がある．問い（〜とは？），指示（〜してください），題目の句（〜あるある，〜な事，〜の共通点，〜にありがちなこと），状況の設定（〜なのに…），名前などの列挙を含む．ボケてのテンプレートの枠内の文字，無地の背景に置いた文字，画像の上下に付け足した文字はお題文の候補である．テレビの字幕，看板，商品の表示，漫画の台詞など画像に元からある文字はお題文ではない．
4. normal（通常）：上記のいずれでもない．
投稿者の意図をくみ取って判定する．

## 手順2：属性（当てはまるものを true）
- has_blank：空欄あり（手順1の2の空欄）
- has_prompt_text：お題文あり（手順1の3のお題文）
- has_karuta_mark：カルタ記号あり
- has_other_text：お題文，カルタ記号以外の，ボケに関わりうる文字がある（字幕，看板，台詞など．透かし，著作権表示，ロゴ，判読できない小さな文字は除く）
- composite：合成画像（複数の画像の貼り合わせ，写真と絵の重ね合わせ）

## 手順3：媒体（原則1つだけ true．合成画像で異なる媒体の領域が画像の約2割以上ずつある場合のみ複数）
- photo：現実の写真（実写の映像，フィギュアや人形，着ぐるみ，像を撮影したものを含む）
- anime：アニメのスクリーンショット
- game：ゲーム画面のキャプチャ
- cg：ゲームやアニメの画面ではない3DCG
- manga：漫画作品から切り抜いたコマ
- text_graphic：文字や記号が主体で，写真や絵がほとんどない画像（ウェブページやアプリの画面を含む）
- painting：古典的絵画（浮世絵などの版画を含み，写真に撮ったものも絵画とする）
- illustration：上記に含まれない絵（キャラクターの描き下ろし，模写，漫画風の一枚絵，カルタの札の絵，手描きの絵を含む）
生成AIによる画像は，同じ見た目の画像を人が作った場合の媒体とする．

## 手順4：事前知識（当てはまるものを true．なければすべて false）
- known_person：名前を特定できる実在の著名人が主な被写体である（テロップなどから分かる場合を含む）
- known_work：作品名を特定できるキャラクターや場面が主な被写体である（アニメ，漫画，ゲーム，映画など．企業のマスコットを含む）
- work_unknown：作品に由来するキャラクターや場面が主な被写体であるが，作品名を特定できない"""

PROMPT_V4 = """この画像は大喜利サイト「ボケて」のお題画像です．以下の項目を上から順に判定し，JSON のみを出力してください．

## 1. 画像の中の記号と文字（当てはまるものを true）
- has_karuta_mark：カルタ記号がある．カルタ記号とは，円や四角の枠の中にひらがな1文字だけを書いた，文から独立した札の記号である（多くは画像の隅にある）．空欄の前後に書かれたひらがな（「□い。」「お□」など）や，文章の一部のひらがなはカルタ記号ではない．
- karuta_char：カルタ記号のひらがな1文字（なければ空文字）．
- has_blank：語の代わりに置かれた空欄がある（白や色付きの矩形，○○，□，＿＿，語の代わりの「？」，語の代わりの黒塗り，話者に付いた空の吹き出し）．顔や個人情報の目隠し，文末の疑問符は空欄ではない．
- has_prompt_text：投稿者が書き加えた問いや指示の文（お題文）がある．問い（〜とは？），指示（〜してください），題目の句（〜あるある，〜な事，〜の共通点，〜にありがちなこと），状況の設定（〜なのに…），名前などの列挙を含む．ボケてのテンプレートの枠内の文字，無地の背景に置いた文字，画像の上下に付け足した文字が該当する．テレビの字幕，看板，商品の表示，漫画の台詞など画像に元からある文字は該当しない．
- has_other_text：お題文とカルタ記号以外の，ボケに関わりうる文字がある（字幕，看板，台詞など．透かし，ロゴ，判読できない小さな文字は除く）．
- composite：合成画像（複数の画像の貼り合わせ，写真と絵の重ね合わせ）．

## 2. 形式（format，1つ選択）
上の判定から，次の順で最初に当てはまるものを選ぶ．has_karuta_mark なら karuta，has_blank なら anaume，has_prompt_text なら odaibun，いずれでもなければ normal．

## 3. 媒体（原則1つだけ true．合成画像で異なる媒体の領域が画像の約2割以上ずつある場合のみ複数）
- photo：現実の写真（実写の映像，フィギュアや人形，着ぐるみ，像を撮影したものを含む）
- anime：アニメのスクリーンショット
- game：ゲーム画面のキャプチャ
- cg：ゲームやアニメの画面ではない3DCG
- manga：漫画作品から切り抜いたコマ
- text_graphic：文字や記号が主体で，写真や絵がほとんどない画像（ウェブページやアプリの画面を含む）
- painting：古典的絵画（浮世絵などの版画を含み，写真に撮ったものも絵画とする）
- illustration：上記に含まれない絵（キャラクターの描き下ろし，模写，漫画風の一枚絵，カルタの札の絵，手描きの絵を含む）
生成AIによる画像は，同じ見た目の画像を人が作った場合の媒体とする．

## 4. 事前知識
- person_name：主な被写体が実在の著名人で，その名前を特定できる場合はその名前（テロップから分かる場合を含む）．特定できなければ空文字．
- work_name：主な被写体が作品（アニメ，漫画，ゲーム，映画，特撮など）のキャラクターや場面，または企業のマスコットで，その作品名を特定できる場合は作品名．特定できなければ空文字．
- known_person：person_name が空でなければ true．
- known_work：work_name が空でなければ true．
- work_unknown：作品に由来するキャラクターや場面（アニメの画面や漫画のコマを含む）が主な被写体であるが，work_name を特定できない場合に true．
一般の人物，動物，風景，無名のイラストはいずれも false．"""

PROMPT_V5 = PROMPT_V4.replace(
    "空欄の前後に書かれたひらがな（「□い。」「お□」など）や，文章の一部のひらがなはカルタ記号ではない．",
    "空欄の前後に書かれたひらがな（「□い。」「お□」など），文章の一部のひらがな，中身が空の枠（空欄）はカルタ記号ではない．"
    "カルタ記号の枠の中には必ずひらがなが1文字書かれている．迷ったら false．",
).replace(
    "テレビの字幕，看板，商品の表示，漫画の台詞など画像に元からある文字は該当しない．",
    "テレビの字幕やテロップ，アニメの字幕，看板，商品の表示，漫画やゲームの台詞，本や DVD の題名など画像に元からある文字は該当しない．"
    "回答者に向けて大喜利の課題を示す文でなければ false．迷ったら false．",
)

PROMPTS = {"v3": PROMPT_V3, "v4": PROMPT_V4, "v5": PROMPT_V5}
# 形式を VLM の回答ではなく属性から定義の優先順位で導く版
DERIVE_FORMAT = {"v5"}
# プロンプトの版ごとの出力の項目の順．構造化出力は項目をこの順で生成するため，判定の順序を誘導できる
FIELD_ORDER = {
    "v3": ["format", "karuta_char"] + ATTRIBUTES + MEDIA + KNOWLEDGE,
    "v5": ["has_karuta_mark", "karuta_char", "has_blank", "has_prompt_text", "has_other_text", "composite", "format"]
          + MEDIA + ["person_name", "work_name"] + KNOWLEDGE,
    "v4": ["has_karuta_mark", "karuta_char", "has_blank", "has_prompt_text", "has_other_text", "composite", "format"]
          + MEDIA + ["person_name", "work_name"] + KNOWLEDGE,
}


def json_schema(version: str) -> dict:
    """出力の JSON スキーマを作る．項目は FIELD_ORDER の順に並べる．

    Args:
        version (str): プロンプトの版．

    Returns:
        dict: format（列挙），各項目（真偽値），karuta_char（最大1文字），person_name・work_name（最大40文字）を
        必須とする JSON スキーマ．
    """
    types = {"format": {"type": "string", "enum": FORMATS}, "karuta_char": {"type": "string", "maxLength": 1},
             "person_name": {"type": "string", "maxLength": 40}, "work_name": {"type": "string", "maxLength": 40}}
    props = {k: types.get(k, {"type": "boolean"}) for k in FIELD_ORDER[version]}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def derive_format(rec: dict) -> str:
    """属性から定義の優先順位（カルタ記号 → 空欄 → お題文）で形式を導く．

    Args:
        rec (dict): 属性の真偽値を持つ辞書．

    Returns:
        str: 形式．
    """
    for attr, fmt in [("has_karuta_mark", "karuta"), ("has_blank", "anaume"), ("has_prompt_text", "odaibun")]:
        if rec.get(attr):
            return fmt
    return "normal"


def postprocess(rec: dict, derive: bool = False) -> dict:
    """VLM の出力を整える．形式から必ず真になる属性を補い，媒体が空の場合はイラストとする．

    カルタ記号ありでも karuta_char がひらがなでない場合は，カルタ記号なしとする．

    Args:
        rec (dict): VLM の出力（JSON を解釈した辞書．解釈できない場合は空の辞書）．
        derive (bool): 真の場合，形式を VLM の回答ではなく属性から導く．

    Returns:
        dict: キー format，karuta_char，person_name，work_name と各項目（bool）を持つ辞書．
    """
    out = {"format": rec.get("format", "normal"), "karuta_char": rec.get("karuta_char", "") or "",
           "person_name": rec.get("person_name", "") or "", "work_name": rec.get("work_name", "") or ""}
    for k in ATTRIBUTES + MEDIA + KNOWLEDGE:
        out[k] = bool(rec.get(k, False))
    if derive:
        if out["has_karuta_mark"] and not ("ぁ" <= out["karuta_char"] <= "ゖ"):
            out["has_karuta_mark"] = False
        out["format"] = derive_format(out)
    if out["format"] in FORMAT_IMPLIES:
        out[FORMAT_IMPLIES[out["format"]]] = True
    if not any(out[m] for m in MEDIA):
        out["illustration"] = True
    return out


def build_llm():
    """vLLM のモデルを読み込む．

    Returns:
        vllm.LLM: 読み込んだモデル．
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM

    return LLM(model=MODEL_NAME, max_model_len=4096, gpu_memory_utilization=0.88, seed=0,
               limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs={"max_pixels": MAX_PIXELS})


def _load(fid: int) -> Image.Image:
    """画像を RGB で読み込む．

    Args:
        fid (int): お題ID．

    Returns:
        PIL.Image.Image: RGB 画像．
    """
    with Image.open(image_path(fid)) as im:
        return im.convert("RGB")


def predict(llm, ids: list, version: str, desc: str) -> tuple:
    """画像のリストに4軸のラベルを付ける．

    Args:
        llm (vllm.LLM): モデル．
        ids (list[int]): お題IDのリスト（長さ N）．
        version (str): プロンプトの版．
        desc (str): 進捗バーの説明．

    Returns:
        tuple[pl.DataFrame, float]: 長さ N の予測の表（列 fileId, format, karuta_char と各項目）と，推論の秒数．
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompt = PROMPTS[version]
    params = SamplingParams(temperature=0, max_tokens=400,
                            structured_outputs=StructuredOutputsParams(json=json_schema(version)))
    recs, sec = [], 0.0
    for s in tqdm(range(0, len(ids), CHUNK), desc=desc, leave=False):
        with ThreadPoolExecutor(32) as ex:
            images = list(ex.map(_load, ids[s:s + CHUNK]))
        convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                                {"type": "text", "text": prompt}]}] for im in images]
        start = time.perf_counter()
        outs = llm.chat(convs, sampling_params=params, use_tqdm=False)
        sec += time.perf_counter() - start
        for o in outs:
            try:
                recs.append(postprocess(json.loads(o.outputs[0].text), version in DERIVE_FORMAT))
            except json.JSONDecodeError:
                recs.append(postprocess({}, version in DERIVE_FORMAT))
    cols = ["fileId", "format", "karuta_char", "person_name", "work_name"] + ATTRIBUTES + MEDIA + KNOWLEDGE
    df = pl.DataFrame(recs).with_columns(pl.Series("fileId", ids)).select(cols)
    return df, sec


def main() -> None:
    """評価用データで教師を評価し，必要に応じて教師ラベル用の画像にラベルを付ける．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v3")
    parser.add_argument("--label_pool", action="store_true")
    parser.add_argument("--rules_only", action="store_true", help="保存した予測に後処理の規則を適用して評価する")
    args = parser.parse_args()
    if args.rules_only:
        evaluate_rules(args.prompt)
        return
    name = MODEL_NAME.split("/")[-1]
    run_dir = os.path.join(OUTPUT_DIR, f"teacher_{name}", args.prompt, "0")
    label_path = os.path.join(DATASET_DIR, "teacher_labels", f"{name}_{args.prompt}.csv")
    if is_done(run_dir) and (not args.label_pool or os.path.exists(label_path)):
        return
    labels = load_eval_labels()
    llm = build_llm()
    predict(llm, labels["fileId"].to_list()[:16], args.prompt, "warmup")
    pred, sec = predict(llm, labels["fileId"].to_list(), args.prompt, "eval")
    meta = {"method": "teacher_vlm", "model": MODEL_NAME, "prompt_version": args.prompt, "prompt": PROMPTS[args.prompt],
            "max_pixels": MAX_PIXELS, "throughput_eval": throughput(len(pred), sec)}
    if args.label_pool:
        ids = load_pool()["fileId"].to_list()
        pool_pred, pool_sec = predict(llm, ids, args.prompt, "pool")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        pool_pred.write_csv(label_path)
        meta["throughput_pool"] = throughput(len(ids), pool_sec)
    save_run(run_dir, labels, pred.drop("fileId"), meta)



def evaluate_rules(version: str) -> None:
    """保存した教師の予測に後処理の規則（rules.py）を適用した結果を評価して保存する．

    Args:
        version (str): プロンプトの版．

    Returns:
        None
    """
    from rules import apply_rules

    name = MODEL_NAME.split("/")[-1]
    base = os.path.join(OUTPUT_DIR, f"teacher_{name}", version, "0")
    labels = load_eval_labels()
    p = pl.read_csv(os.path.join(base, "predictions.csv"))
    pred = p.select([c for c in p.columns if c.startswith("pred_")]).rename(lambda c: c[5:])
    for tag, wu, bk in [("rule_wu", True, False), ("rule_wu_boke", True, True)]:
        out = apply_rules(pred, labels["fileId"].to_list(), use_work_unknown=wu, use_boke=bk)
        meta = {"method": "teacher_vlm_rules", "model": MODEL_NAME, "prompt_version": version,
                "rules": {"work_unknown": wu, "boke_karuta_veto": bk}}
        save_run(os.path.join(OUTPUT_DIR, f"teacher_{name}", f"{version}_{tag}", "0"), labels, out, meta)


if __name__ == "__main__":
    main()
