"""本番のキャプション（captions.parquet）に品質の目印を付け，お題ごと（内容が同じ画像のお題を含む）に展開する．

caption_ok が偽になるのは，次のいずれかに当てはまる文である（全体の約0.003%）．
- 最大トークン数での打ち切り，80字超，改行を含む，日本語を含まない（英語の文など）
- 指示文の言い直し（「アシスタントは」「はい、承知」で始まる）
- 同じ語の4回以上の繰り返し（ただし数字と，笑い声などの「っ」を含む繰り返しは正常とみなす）

出力：outputs/ex011_caption_final/photo_captions/photo_captions_by_odai.parquet
（列 odaiId, rep_odaiId, caption, caption_ok．媒体が写真のみのお題の全て）

実行例:
    .venv_annotate/bin/python programs/ex011_caption_final/postprocess_captions.py
"""
import json
import os
import re
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caption_all import ANN_PATH, MAX_CHARS, MEDIA, PROJECT_DIR  # noqa: E402

OUT = os.path.join(PROJECT_DIR, "outputs", "ex011_caption_final", "photo_captions")
REPEAT = re.compile(r"(.{2,8})\1{3,}")
META = re.compile(r"^(アシスタントは|はい、承知|はい、以下)")


def is_ok(caption: str, finish: str) -> bool:
    """キャプションが正常かを判定する．

    Args:
        caption (str): 句読点をそろえたキャプション．
        finish (str): 生成の終わり方（stop / length）．

    Returns:
        bool: 正常なら真．
    """
    if finish == "length" or len(caption) > MAX_CHARS or "\n" in caption or not caption:
        return False
    if not re.search(r"[ぁ-んァ-ン一-龥]", caption) or META.search(caption):
        return False
    m = REPEAT.search(caption)
    return not (m and not re.search(r"[0-9っッ]", m.group(1)))


def main() -> None:
    """品質の目印を付けて，お題ごとに展開して保存する．

    Returns:
        None
    """
    c = pl.read_parquet(os.path.join(OUT, "captions.parquet"))
    c = c.with_columns(pl.Series("caption_ok", [is_ok(s, f) for s, f in zip(c["caption"], c["finish"])]))
    ann = pl.scan_parquet(ANN_PATH).filter(pl.col("photo") & ~pl.any_horizontal([pl.col(m) for m in MEDIA if m != "photo"])) \
        .select("odaiId", "rep_odaiId").collect()
    by = ann.join(c.select(pl.col("odaiId").alias("rep_odaiId"), "caption", "caption_ok"), on="rep_odaiId", how="left").sort("odaiId")
    assert by["caption"].null_count() == 0, "キャプションのないお題がある"
    by.write_parquet(os.path.join(OUT, "photo_captions_by_odai.parquet"))
    bad = c.filter(~pl.col("caption_ok"))
    stats = {"n_images": c.height, "n_odai": by.height, "n_not_ok": bad.height, "rate_not_ok": bad.height / c.height,
             "examples_not_ok": bad.head(10)["caption"].to_list()}
    json.dump(stats, open(os.path.join(OUT, "postprocess.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
