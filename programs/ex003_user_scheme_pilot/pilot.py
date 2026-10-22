"""ユーザー案のラベル定義（order_004）を Qwen3-VL-8B のプロンプトにし，評価用データ640枚に適用するパイロット．

定義の文言はユーザー案をそのまま用い，出力は JSON スキーマによる構造化出力で
形式（1つ），属性・媒体・事前知識（各項目の真偽値），カルタのひらがな（1文字）に制約する．
さらに BoketeDBv4 のボケの本文を結合し，形式の判断を実際の大喜利で裏付けられるかを調べる．

出力（outputs/ex003_user_scheme_pilot/vlm_Qwen3-VL-8B-Instruct/user_v1_px200704/0/）:
- predictions.csv：画像ごとの出力と，現行の目視ラベル，ボケの統計
- meta.json：条件と処理速度
- boke_stats.json：形式ごとのボケの統計（カルタのひらがなから始まるボケの割合など）

実行例:
    .venv_annotate/bin/python programs/ex003_user_scheme_pilot/pilot.py
"""
import json
import os
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from PIL import Image

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex002_odai_type_distill"))
from data import image_path, load_eval_labels  # noqa: E402

DB_DIR = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 448 * 448
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex003_user_scheme_pilot", "vlm_Qwen3-VL-8B-Instruct",
                       f"user_v1_px{MAX_PIXELS}", "0")

FORMATS = ["karuta", "anaume", "odaibun", "normal"]
ATTRIBUTES = ["has_blank", "has_prompt_text", "has_karuta_mark", "has_other_text"]
MEDIA = ["photo", "anime", "game", "cg", "manga", "text_graphic", "painting", "illustration"]
KNOWLEDGE = ["known_person", "known_work"]

PROMPT = """この画像は大喜利サイト「ボケて」のお題画像です．以下の定義に従って分類してください．

【形式（1つ選択）】ユーザーの回答を制限する画像
- karuta（カルタ）：ひらがな1文字が入った記号（○，□など）がある画像．ひらがなから始まる大喜利を投稿する．
- anaume（穴埋め）：語の代わりに置かれた空欄（矩形，○，吹き出しなど）がある画像．空欄に入る大喜利を投稿する．
- odaibun（お題文）：問いや指示の文（「～とは？」，「～してください」など）がある画像．問いや指示に対応した大喜利を投稿する．
- normal（通常）：上記の形式以外の画像．
画像の投稿者の意図をくみ取って決定してください．

【属性（複数選択，当てはまるものを true）】
- has_blank：空欄あり
- has_prompt_text：お題文あり
- has_karuta_mark：カルタ記号あり
- has_other_text：お題文，カルタ記号ではない文字あり

【媒体（複数選択，当てはまるものを true）】画像のスタイル
- photo（写真）：現実の写真
- anime（アニメ）：アニメのスクリーンショット
- game（ゲーム）：ゲーム画面のキャプチャ
- cg（CG）：コンピュータグラフィック
- manga（マンガ）：漫画作品から切り抜いたコマ
- text_graphic（テキストグラフィック）：文字が記入された画像
- painting（絵画）：古典的絵画
- illustration（イラスト）：上記に含まれない絵

【事前知識（複数選択，当てはまるものを true）】
- known_person（実在の著名人）：芸能人などの有名人を含む画像
- known_work（特定の作品）：ドラゴンボールやドラえもんなどの作品に由来するものを含む画像

karuta_char には，カルタ記号の中のひらがな1文字を書いてください（カルタ記号がなければ空文字）．
JSON のみを出力してください．"""


def json_schema() -> dict:
    """出力の JSON スキーマを作る．

    Returns:
        dict: 形式（列挙），各項目（真偽値），karuta_char（最大1文字）を必須とする JSON スキーマ．
    """
    props = {"format": {"type": "string", "enum": FORMATS}}
    for k in ATTRIBUTES + MEDIA + KNOWLEDGE:
        props[k] = {"type": "boolean"}
    props["karuta_char"] = {"type": "string", "maxLength": 1}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def run_vlm(ids: list) -> tuple:
    """評価用画像に VLM を適用する．

    Args:
        ids (list[int]): お題IDのリスト（長さ N）．

    Returns:
        tuple[list[dict], float]: 長さ N の出力（解釈できない場合は空の辞書）と推論の秒数．
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=MODEL_NAME, max_model_len=4096, gpu_memory_utilization=0.88, seed=0,
              limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs={"max_pixels": MAX_PIXELS})
    params = SamplingParams(temperature=0, max_tokens=400, structured_outputs=StructuredOutputsParams(json=json_schema()))

    def load(fid: int) -> Image.Image:
        """画像を RGB で読み込む．

        Args:
            fid (int): お題ID．

        Returns:
            PIL.Image.Image: RGB 画像．
        """
        with Image.open(image_path(fid)) as im:
            return im.convert("RGB")

    with ThreadPoolExecutor(32) as ex:
        images = list(ex.map(load, ids))
    convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im}, {"type": "text", "text": PROMPT}]}]
             for im in images]
    llm.chat(convs[:16], sampling_params=params, use_tqdm=False)  # ウォームアップ
    start = time.perf_counter()
    outs = llm.chat(convs, sampling_params=params, use_tqdm=False)
    sec = time.perf_counter() - start
    results = []
    for o in outs:
        try:
            results.append(json.loads(o.outputs[0].text))
        except json.JSONDecodeError:
            results.append({})
    return results, sec


def to_hiragana(ch: str) -> str:
    """カタカナ1文字をひらがなに変換し，NFKC で正規化する．

    Args:
        ch (str): 1文字．

    Returns:
        str: ひらがなに変換した1文字（カタカナ以外はそのまま）．
    """
    ch = unicodedata.normalize("NFKC", ch)
    return chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch


def first_char(text: str) -> str:
    """ボケの本文の最初の意味のある文字を，ひらがなに揃えて返す．

    先頭の空白，括弧，引用符，記号は読み飛ばす．

    Args:
        text (str): ボケの本文．

    Returns:
        str: 最初の文字（なければ空文字）．
    """
    for ch in text or "":
        if ch.isspace() or unicodedata.category(ch)[0] in "PSZ":
            continue
        return to_hiragana(ch)
    return ""


def load_bokes(ids: list) -> pl.DataFrame:
    """評価用画像に投稿されたボケを読み込む．

    Args:
        ids (list[int]): お題IDのリスト．

    Returns:
        pl.DataFrame: 列 odaiId, text, rateSum を持つ表．
    """
    shards = sorted({fid // 10000 + 1 for fid in ids})
    paths = [os.path.join(DB_DIR, "boke", f"boke_{n}.parquet") for n in shards]
    return (pl.scan_parquet(paths).filter(pl.col("odaiId").is_in(ids))
            .select("odaiId", "text", "rateSum").collect())


def boke_features(df: pl.DataFrame, bokes: pl.DataFrame) -> pl.DataFrame:
    """お題ごとのボケの統計を求め，予測の表に結合する．

    Args:
        df (pl.DataFrame): 列 fileId と karuta_char を持つ表．
        bokes (pl.DataFrame): load_bokes の戻り値．

    Returns:
        pl.DataFrame: df に n_bokes（ボケの数），median_len（本文の文字数の中央値），
        karuta_start_rate（karuta_char から始まるボケの割合），top_bokes（評価の高い3件）を加えた表．
    """
    chars = dict(zip(df["fileId"].to_list(), df["karuta_char"].to_list()))
    rows = []
    for fid, g in bokes.group_by("odaiId"):
        fid = fid[0]
        texts = g["text"].fill_null("").to_list()
        ch = to_hiragana(chars.get(fid) or "") if chars.get(fid) else ""
        rate = float(np.mean([first_char(t) == ch for t in texts])) if ch and texts else None
        top = g.sort("rateSum", descending=True, nulls_last=True).head(3)["text"].to_list()
        rows.append({"fileId": fid, "n_bokes": len(texts), "median_len": float(np.median([len(t) for t in texts])),
                     "karuta_start_rate": rate, "top_bokes": " / ".join(t.replace("\n", " ")[:40] for t in top)})
    return df.join(pl.DataFrame(rows), on="fileId", how="left")


def summarize(df: pl.DataFrame) -> dict:
    """形式ごとのボケの統計と，各軸のラベルの分布をまとめる．

    Args:
        df (pl.DataFrame): boke_features の戻り値．

    Returns:
        dict: 形式ごとの件数・ボケの文字数の中央値・カルタのひらがなから始まるボケの割合，各項目が true の割合，
        媒体の選択数の分布，現行の目視ラベルとの対応．
    """
    by_fmt = (df.group_by("format").agg(pl.len().alias("n"), pl.col("median_len").median().alias("median_len"),
                                        pl.col("karuta_start_rate").mean().alias("karuta_start_rate_mean"))
              .sort("n", descending=True))
    by_hand = (df.group_by("hand_format").agg(pl.len().alias("n"), pl.col("median_len").median().alias("median_len"),
                                              pl.col("karuta_start_rate").mean().alias("karuta_start_rate_mean")))
    n_media = df.select(pl.sum_horizontal([pl.col(m).cast(int) for m in MEDIA]).alias("k"))["k"]
    cross = df.group_by("hand_format", "format").len().sort("len", descending=True)
    return {"by_vlm_format": by_fmt.to_dicts(), "by_hand_format": by_hand.to_dicts(),
            "true_rate": {k: float(df[k].mean()) for k in ATTRIBUTES + MEDIA + KNOWLEDGE},
            "n_media_selected": {int(k): int(v) for k, v in zip(*np.unique(n_media.to_numpy(), return_counts=True))},
            "hand_vs_vlm_format": cross.to_dicts()}


def main() -> None:
    """パイロットを実行して結果を保存する．

    Returns:
        None
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = load_eval_labels()
    ids = labels["fileId"].to_list()
    pred_path = os.path.join(OUT_DIR, "raw_outputs.json")
    if os.path.exists(pred_path):
        results, sec = json.load(open(pred_path)), json.load(open(os.path.join(OUT_DIR, "meta.json")))["infer_sec"]
    else:
        results, sec = run_vlm(ids)
        json.dump(results, open(pred_path, "w"), ensure_ascii=False)
    keys = ["format"] + ATTRIBUTES + MEDIA + KNOWLEDGE + ["karuta_char"]
    cols = {k: [r.get(k) for r in results] for k in keys}
    df = pl.DataFrame({"fileId": ids, "pool": labels["pool"], "hand_format": labels["format"],
                       "hand_domain": labels["domain"], **cols})
    df = boke_features(df, load_bokes(ids))
    df.write_csv(os.path.join(OUT_DIR, "predictions.csv"))
    summary = summarize(df)
    json.dump(summary, open(os.path.join(OUT_DIR, "boke_stats.json"), "w"), indent=2, ensure_ascii=False)
    meta = {"model": MODEL_NAME, "max_pixels": MAX_PIXELS, "prompt": PROMPT, "n_images": len(ids), "infer_sec": sec,
            "images_per_sec": len(ids) / sec, "n_unparsed": sum(1 for r in results if not r)}
    json.dump(meta, open(os.path.join(OUT_DIR, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "hand_vs_vlm_format"}, ensure_ascii=False, indent=1))
    print(f"{meta['images_per_sec']:.1f} 枚/秒，解釈できない出力 {meta['n_unparsed']} 件")


if __name__ == "__main__":
    main()
