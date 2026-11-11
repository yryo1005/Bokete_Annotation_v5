"""最終版のアノテーション（パッケージ v5.3.0．ボケは v5.2.0）を，新しいプロジェクトで使うための独立したディレクトリに書き出す．

書き出し先（--out．既定 /mnt/d/BoketeDBv5_annotation_final）の構成：
- data/odai/：annotation.parquet，images.parquet，captions.parquet，summary.json，predict/ と embeddings/（SigLIP2 の埋め込みと行の対応），odai_text/
- data/boke/：boke_annotation/part_*.parquet，boke_summary.json
- data/subsets/：写真のみ，写真・文字なし，写真・通常のお題の一覧（ボケの件数付き）
- docs/：ラベルの定義書，方法と精度，データの形式，最終版のまとめ（report_051），部分集合の精度（report_050），prompts/（指示の本文）
- eval/：目視のラベル（お題画像 v4 の1,020枚，文の属性 v6 の1,620枚・636枚・600枚，ボケの検証データ620件，キャプションの参照文）と，
  精度の集計の表（metrics/）
- annotator/：アノテーションのパッケージのコードとモデルの重み（outputs，.git，キャッシュを除く）

複写は大きさが同じファイルを飛ばすため，中断しても同じコマンドで再開できる．README.md は，このディレクトリの README_final.md を複写する．

実行例:
    .venv_annotate/bin/python programs/ex031_export_final/export.py
    .venv_annotate/bin/python programs/ex031_export_final/export.py --skip_large   # 埋め込みとボケのラベル以外だけを書き出す
"""
import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys

import polars as pl
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
ANN = os.path.join(PKG, "outputs", "BoketeDBv5_annotation")
REPORTS = os.path.join(PROJECT_DIR, ".reports")
# 複写するもの（複写元，書き出し先の相対パス，大きいか）
COPIES = [
    (os.path.join(THIS_DIR, "README_final.md"), "README.md", False),
    (os.path.join(ANN, "annotation.parquet"), "data/odai/annotation.parquet", False),
    (os.path.join(ANN, "images.parquet"), "data/odai/images.parquet", False),
    (os.path.join(ANN, "captions.parquet"), "data/odai/captions.parquet", False),
    (os.path.join(ANN, "summary.json"), "data/odai/summary.json", False),
    (os.path.join(ANN, "predict"), "data/odai/predict", False),
    (os.path.join(ANN, "odai_text"), "data/odai/odai_text", False),
    (os.path.join(ANN, "embeddings"), "data/odai/embeddings", True),
    (os.path.join(ANN, "boke_annotation"), "data/boke/boke_annotation", True),
    (os.path.join(ANN, "boke_summary.json"), "data/boke/boke_summary.json", False),
    (os.path.join(PKG, "docs", "label_definition.md"), "docs/label_definition_v5.md", False),
    (os.path.join(PKG, "docs", "label_definition_v4.md"), "docs/label_definition_v4.md", False),
    (os.path.join(PROJECT_DIR, "programs", "label_guideline", "label_definition_v6.md"), "docs/label_definition_v6_draft.md", False),
    (os.path.join(PKG, "docs", "boke_label_definition.md"), "docs/boke_label_definition_v3.md", False),
    (os.path.join(PKG, "docs", "method.md"), "docs/method.md", False),
    (os.path.join(PKG, "docs", "data_format.md"), "docs/data_format.md", False),
    (os.path.join(REPORTS, "report_051.md"), "docs/final_summary.md", False),
    (os.path.join(REPORTS, "report_051.pdf"), "docs/final_summary.pdf", False),
    (os.path.join(REPORTS, "report_050.md"), "docs/photo_subset.md", False),
    (os.path.join(REPORTS, "figures_050"), "docs/figures_050", False),
    (os.path.join(REPORTS, "report_050.pdf"), "docs/photo_subset.pdf", False),
    (os.path.join(PROJECT_DIR, "programs", "ex010_photo_caption", "reference_captions.txt"), "eval/caption_reference_120a.txt", False),
    (os.path.join(PROJECT_DIR, "programs", "ex011_caption_final", "reference_captions_eval2.txt"), "eval/caption_reference_120b.txt", False),
]
METRICS = {
    "outputs/ex030_final_summary/image_metrics.csv": "image_format_attr_media.csv",
    "outputs/ex030_final_summary/rare_metrics.csv": "image_rare_categories.csv",
    "outputs/ex030_final_summary/odai_text_metrics.csv": "image_odai_text.csv",
    "outputs/ex030_final_summary/boke_metrics.csv": "boke_voice_outside.csv",
    "outputs/ex029_photo_subset/metrics.csv": "photo_subset.csv",
    "outputs/ex029_photo_subset/counts.csv": "photo_subset_counts.csv",
    "outputs/ex028_odai_holdout/eval_df/df_rates.csv": "odai_text_df_rates.csv",
}
PKG_IGNORE = shutil.ignore_patterns("outputs", ".git", "__pycache__", ".pytest_cache", "*.pyc")


def _load(name: str, path: str):
    """ファイルからモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．
        path (str): ファイルのパス．

    Returns:
        module: 読み込んだモジュール．
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def copy_file(src: str, dst: str) -> bool:
    """ファイルを複写する（同じ大きさのファイルがあれば飛ばす）．

    Args:
        src (str): 複写元．
        dst (str): 複写先．

    Returns:
        bool: 複写したなら True．
    """
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: str, dst: str) -> int:
    """ディレクトリの中のファイルを全て複写する．

    Args:
        src (str): 複写元のディレクトリ．
        dst (str): 複写先のディレクトリ．

    Returns:
        int: 複写したファイルの数．
    """
    files = [p for p in glob.glob(os.path.join(src, "**", "*"), recursive=True) if os.path.isfile(p)]
    n = 0
    for p in tqdm(files, desc=os.path.basename(src), leave=False):
        n += copy_file(p, os.path.join(dst, os.path.relpath(p, src)))
    return n


def export_subsets(out: str) -> dict:
    """写真のみ，写真・文字なし，写真・通常のお題の一覧を書き出す（ex029 の条件と同じ）．

    Args:
        out (str): 書き出し先の data/subsets．

    Returns:
        dict[str, int]: 部分集合ごとのお題の数．
    """
    d29 = _load("ex029_data_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex029_photo_subset", "data.py"))
    a = d29.load_pred().join(d29.count_boke(), on="odaiId", how="left").with_columns(pl.col("n_boke", "n_outside").fill_null(0))
    os.makedirs(out, exist_ok=True)
    sizes = {}
    for name, col in [("photo_only", "photo_only"), ("photo_notext", "A_photo_notext"), ("photo_normal", "B_photo_normal")]:
        s = a.filter(pl.col(col)).select("odaiId", "rep_odaiId", "n_boke", "n_outside").sort("odaiId")
        s.write_parquet(os.path.join(out, f"{name}.parquet"))
        sizes[name] = s.height
    return sizes


def export_eval_labels(out: str) -> dict:
    """精度の評価に用いた目視のラベルを CSV に書き出す．

    Args:
        out (str): 書き出し先の eval．

    Returns:
        dict[str, int]: ファイルごとの行数．
    """
    os.makedirs(out, exist_ok=True)
    rows = {}
    e22 = _load("ex022_data_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex022_gemini_image_labels", "data.py"))
    e23 = _load("ex023_data_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex023_image_label_pr", "data.py"))
    raw = e22.load_claude_labels().select("fileId", "format")
    t = e23.load_truth().join(raw, on="fileId", how="left").rename({"fileId": "odaiId"})
    t = t.select("odaiId", "set", "stratum", "w", "format", *[c for c in t.columns if c.startswith("format_") is False and c not in
                                                              ("odaiId", "set", "stratum", "w", "format")])
    t.write_csv(os.path.join(out, "image_labels_v4_1020.csv"))
    rows["image_labels_v4_1020.csv"] = t.height
    e26 = _load("ex026_data_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex026_text_attr_v6", "data.py"))
    a26 = e26.load_all().drop("qwen_v5", strict=False)
    a26.write_csv(os.path.join(out, "text_attr_v6_1620.csv"))
    rows["text_attr_v6_1620.csv"] = a26.height
    e28 = _load("ex028_data_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex028_odai_holdout", "data.py"))
    keep = lambda d: d.select([c for c in d.columns if not c.startswith("p_") and c not in ("has_odai_text",)])  # noqa: E731
    h = keep(e28.load_all())
    h.write_csv(os.path.join(out, "text_attr_v6_holdout_636.csv"))
    rows["text_attr_v6_holdout_636.csv"] = h.height
    df = keep(e28.load_df())
    df.write_csv(os.path.join(out, "text_attr_v6_strata_DF_600.csv"))
    rows["text_attr_v6_strata_DF_600.csv"] = df.height
    sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm"))
    saved = sys.modules.get("data")
    d16 = _load("ex016_data_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm", "data.py"))
    if saved is not None:
        sys.modules["data"] = saved
    b = d16.load_eval().select("no", "odaiId", "text", "split", "stratum", "format", "voice", "voice3", "outside", "comp", "uses_text")
    b.write_csv(os.path.join(out, "boke_labels_620.csv"))
    rows["boke_labels_620.csv"] = b.height
    return rows


def export_prompts(out: str) -> list:
    """教師と分類器の指示の本文を書き出す．

    Args:
        out (str): 書き出し先の docs/prompts．

    Returns:
        list[str]: 書き出したファイルの名前．
    """
    os.makedirs(out, exist_ok=True)
    names = []

    def write(name: str, text: str) -> None:
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")
        names.append(name)

    ex5 = os.path.join(PROJECT_DIR, "programs", "ex005_rare_weighted_distill")
    saved = sys.modules.get("data")
    sys.modules["data"] = _load("ex005_data_for_ex031", os.path.join(ex5, "data.py"))
    try:
        p5 = _load("ex005_prompts_for_ex031", os.path.join(ex5, "prompts.py"))
    finally:
        sys.modules["data"] = saved
    for k in ["F_v2", "J_v6", "M_v6", "P_v8_en", "X_v10_en", "O_v11_en"]:
        write(f"image_teacher_{k}.txt", p5.COMPONENTS[k]["prompt"])
    saved = sys.modules.get("data")
    sys.modules["data"] = _load("ex024_data_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex024_odaibun_attr", "data.py"))
    try:
        q24 = _load("ex024_qwen_for_ex031", os.path.join(PROJECT_DIR, "programs", "ex024_odaibun_attr", "qwen_odaibun.py"))
    finally:
        sys.modules["data"] = saved
    write("odai_text_teacher_P3.txt", q24.P3)
    cfg = json.load(open(os.path.join(PKG, "config.json"), encoding="utf-8"))
    write("caption.txt", cfg["caption"]["prompt"])
    ex15 = os.path.join(PROJECT_DIR, "programs", "ex015_boke_labels_v3")
    sys.path.insert(0, ex15)
    saved = sys.modules.get("data")
    sys.modules["data"] = _load("ex015_data_for_ex031", os.path.join(ex15, "data.py"))
    try:
        t15 = _load("ex015_teacher_for_ex031", os.path.join(ex15, "teacher.py"))
    finally:
        sys.path.remove(ex15)
        sys.modules["data"] = saved
    row = {"caption": "{写真のキャプション}", "has_text": False}
    text = t15.build_prompt("one_kt_cap", "normal", "{ボケの文}", row).replace("お題画像の媒体：不明", "お題画像の媒体：{媒体}").replace(
        "画像中の文字：なし", "画像中の文字：{あり／なし}").replace("お題の形式：通常（画像に自由に一言を付ける）", "お題の形式：{形式とその説明}")
    write("boke_teacher_one_kt_cap.txt", text + "\n\n# 出力の JSON スキーマ\n" + json.dumps(t15.SCHEMA_KT, ensure_ascii=False, indent=1))
    write("boke_classifier_input.txt", "大喜利サイト「ボケて」のお題画像（形式：{形式}）に投稿されたボケ：「{ボケの文．改行を空白にし，200文字まで}」\n"
          "# 入力：お題画像（長辺448画素以下）とこの文．Qwen3-VL-4B＋LoRA の最終層の最後のトークンの隠れ状態から，発話の型（6区分）と画像外の知識（2値）を出力する．")
    return names


def main() -> None:
    """書き出しを行う．

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/d/BoketeDBv5_annotation_final")
    ap.add_argument("--skip_large", action="store_true")
    args = ap.parse_args()
    log = {"copied": {}}
    for src, rel, large in tqdm(COPIES, desc="copy", leave=False):
        if large and args.skip_large:
            continue
        dst = os.path.join(args.out, rel)
        log["copied"][rel] = copy_tree(src, dst) if os.path.isdir(src) else int(copy_file(src, dst))
    shutil.copytree(PKG, os.path.join(args.out, "annotator"), ignore=PKG_IGNORE, dirs_exist_ok=True)
    for src, name in METRICS.items():
        copy_file(os.path.join(PROJECT_DIR, src), os.path.join(args.out, "eval", "metrics", name))
    log["subsets"] = export_subsets(os.path.join(args.out, "data", "subsets"))
    log["eval_labels"] = export_eval_labels(os.path.join(args.out, "eval"))
    log["prompts"] = export_prompts(os.path.join(args.out, "docs", "prompts"))
    json.dump(log, open(os.path.join(args.out, "export_log.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
