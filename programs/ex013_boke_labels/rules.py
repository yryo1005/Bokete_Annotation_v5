"""規則による判定（VLM を使わない最も速い方法）．

- 制約の順守：カルタは，ボケの書き出しのかなが，お題の札のかな（そのお題のボケの書き出しの最頻値．漢字は pykakasi で読みに変換）と一致するか．
  漢字で始まるボケは読みが複数ありうるため，順守とみなす．
  穴埋め・お題文は，日本語を含み，宣伝や URL だけでないか．通常は対象外．
- 発話の型：形式（穴埋め・お題文は回答），括弧・文末・語句の表層の手がかり．
- 画像中の文字への依存：形式が穴埋め・お題文なら依存する．
- 性的・不適切：語の辞書．
- 必要な知識：規則では判定できないため，最も多い区分（一般常識）とする．

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/rules.py
"""
import json
import os
import re
import sys
from collections import Counter

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR, load_bokes, load_eval  # noqa: E402

SKIP = set("「『（(【［[ 　\"'“”…・‥.、。,!！?？〜~ー-")
JA = re.compile(r"[ぁ-んァ-ン一-龥]")
_KAKASI = None
# 性的・不適切の語の辞書（部分一致）．辞書だけでは文脈が分からないため，再現率と適合率の目安として用いる
# 検証データを見る前から想定できる一般的な語に限る（検証データに現れた語をそのまま加えると，評価が甘くなるため）
LEXICON = ["セックス", "エッチ", "えっち", "勃起", "射精", "精子", "ちんこ", "ちんぽ", "まんこ", "おっぱい", "乳首", "童貞", "処女",
           "オナニー", "シコシコ", "中出し", "AV女優", "風俗", "ソープ", "レイプ", "痴漢", "変態", "ラブホ", "包茎", "亀頭", "肉便器",
           "ガイジ", "池沼", "死ね", "殺す"]


def kakasi():
    """pykakasi の変換器を返す（初回のみ作る）．

    Returns:
        pykakasi.kakasi: 変換器．
    """
    global _KAKASI
    if _KAKASI is None:
        import pykakasi

        _KAKASI = pykakasi.kakasi()
    return _KAKASI


def head_kana(text: str) -> str:
    """ボケの書き出しの音（ひらがな1文字）を求める．括弧や記号は飛ばし，漢字やカタカナは読みに変換する．

    Args:
        text (str): ボケの文．

    Returns:
        str: ひらがな1文字（求められなければ空文字）．
    """
    s = "".join(text or "")
    i = 0
    while i < len(s) and s[i] in SKIP:
        i += 1
    s = s[i:i + 8]
    if not s:
        return ""
    if "ぁ" <= s[0] <= "ゖ":
        return s[0]
    if "ァ" <= s[0] <= "ヶ":
        return chr(ord(s[0]) - 0x60)
    if JA.match(s[0]):
        conv = kakasi().convert(s)
        h = conv[0]["hira"] if conv else ""
        return h[0] if h and "ぁ" <= h[0] <= "ゖ" else ""
    return ""


def karuta_chars(odai_ids: list) -> dict:
    """お題ごとの札のかな（ボケの書き出しの最頻値．小書きのかなは含めない）を推定する．

    Args:
        odai_ids (list[int]): お題IDのリスト．

    Returns:
        dict[int, str]: お題ID → ひらがな1文字．
    """
    b = load_bokes(odai_ids, ("odaiId", "text"))
    out = {}
    for (fid,), g in b.group_by("odaiId"):
        c = Counter(k for k in (head_kana(t) for t in g["text"].fill_null("").to_list()) if k and k not in "ぁぃぅぇぉっゃゅょゎ")
        if c:
            out[int(fid)] = c.most_common(1)[0][0]
    return out


def rule_voice(text: str, fmt: str) -> str:
    """発話の型を表層の手がかりで判定する．

    Args:
        text (str): ボケの文．
        fmt (str): お題の形式．

    Returns:
        str: 発話の型（S, M, T, N, L, A）．
    """
    if fmt in ("anaume", "odaibun"):
        return "A"
    t = (text or "").strip()
    if re.search(r"(時の俺|ときの俺|界の|の図|選手権|バージョン|の巻|伝説|の一生)$", t):
        return "L"
    if re.search(r"(様子|ところ|瞬間|結果|光景)。?$", t) or re.search(r"たら.*た。?$", t):
        return "N"
    if re.search(r"(すぎる|すぎ|過ぎる|かよ|じゃん|だろ)[。！!？?]*$", t) and "「" not in t:
        return "T"
    if t.startswith(("（", "(")) or re.search(r"(かな|かなぁ|なぁ|だろう|しよう)[…。.]*$", t):
        return "M"
    if "「" in t or re.search(r"[！!？?ぞぜわよねな]$", t) or re.search(r"(お前|おまえ|あなた|君|キミ)", t):
        return "S"
    if len(t) <= 12 and not re.search(r"[たるいだすよねか。！!？?]$", t):
        return "L"
    return "S"


def rule_labels(df: pl.DataFrame, kchars: dict) -> pl.DataFrame:
    """規則で全ての項目を判定する．

    Args:
        df (pl.DataFrame): 列 odaiId, text, format を持つ表．
        kchars (dict[int, str]): お題ID → 札のかな（karuta_chars の戻り値）．

    Returns:
        pl.DataFrame: 列 complies, voice, knowledge, uses_image_text, inappropriate の表．
    """
    recs = []
    for fid, text, fmt in df.select("odaiId", "text", "format").rows():
        text = text or ""
        if fmt == "karuta":
            # 漢字で始まるボケは読みが複数あり（「辛い」はからい／つらい），読みの変換を誤るため，かなで始まるボケだけを判定する
            first = next((ch for ch in text if ch not in SKIP), "")
            k = head_kana(text)
            if first and not ("ぁ" <= first <= "ゖ" or "ァ" <= first <= "ヶ") and JA.match(first):
                comp = True
            else:
                comp = (k == kchars.get(fid, k)) if k else False
        else:
            comp = bool(JA.search(text)) and not re.search(r"(https?://|www\.|\.com)", text)
        recs.append({"complies": comp, "voice": rule_voice(text, fmt), "knowledge": "G",
                     "uses_image_text": fmt in ("anaume", "odaibun"),
                     "inappropriate": any(w in text for w in LEXICON)})
    return pl.DataFrame(recs)


def main() -> None:
    """検証データを規則で判定して保存する．

    Returns:
        None
    """
    import time

    ev = load_eval()
    kch = karuta_chars(ev.filter(pl.col("format") == "karuta")["odaiId"].unique().to_list())
    start = time.perf_counter()
    lab = rule_labels(ev, kch)
    sec = time.perf_counter() - start
    out = os.path.join(OUTPUT_DIR, "rules")
    os.makedirs(out, exist_ok=True)
    pl.concat([ev.select("no", "odaiId", "format"), lab], how="horizontal").write_csv(os.path.join(out, "eval.csv"))
    json.dump({"n": ev.height, "sec": sec, "per_sec": ev.height / sec}, open(os.path.join(out, "eval_speed.json"), "w"))
    print("rules", f"{ev.height / sec:.0f} 件/秒")


if __name__ == "__main__":
    main()
