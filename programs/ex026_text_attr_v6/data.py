"""ex026（文の属性 v6 の付与と，生徒で学べるかの確認）で用いるデータを扱うモジュール．

- 対象：既存の目視のラベル付きの画像1,620枚（ex024 の1,020枚，ex025 の新しい評価用データ420枚，order_043 の追加の180枚）．
- 目視のラベル：定義書 v6 の草案の文の属性（文字の層と文の形）．書式は label_definition_v6.md の付録（v6_labels.txt）．
- 重み：ex025 の600枚は事後層別の重み（normal_miss.build_set）で全データを推定する．ex024 の1,020枚は学習（交差検証）にだけ用いる．
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex026_text_attr_v6")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex026_text_attr_v6")
TARGETS = os.path.join(DATASET_DIR, "targets.csv")
LABELS = os.path.join(THIS_DIR, "v6_labels.txt")
LAYERS = ["s", "b", "o", "m"]
FORMS = ["k", "q", "i", "t", "x"]
LAYER_NAMES = {"s": "場面の文字", "b": "吹き出し", "o": "重ねた文字", "m": "余白の文字"}
FORM_NAMES = {"k": "空欄を含む文", "q": "問い", "i": "指示", "t": "題目の句", "x": "その他の文"}
SEED = 0


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


EX025_DIR = os.path.join(PROJECT_DIR, "programs", "ex025_odaibun_eval_v5")
EX024 = _load("ex024_data_for_ex026", os.path.join(PROJECT_DIR, "programs", "ex024_odaibun_attr", "data.py"))
image_path = EX024.image_path
load_embeddings = EX024.load_embeddings


def _load_normal_miss():
    """ex025 の normal_miss モジュールを読み込む（ex025 の data モジュールに依存するため sys.modules を一時的に差し替える）．

    Returns:
        module: normal_miss モジュール．
    """
    if "ex025_normal_miss" in sys.modules:
        return sys.modules["ex025_normal_miss"]
    saved = sys.modules.get("data")
    sys.modules["data"] = _load("ex025_data_for_ex026", os.path.join(EX025_DIR, "data.py"))
    sys.path.insert(0, EX025_DIR)
    try:
        mod = _load("ex025_normal_miss", os.path.join(EX025_DIR, "normal_miss.py"))
    finally:
        sys.path.remove(EX025_DIR)
        if saved is None:
            sys.modules.pop("data", None)
        else:
            sys.modules["data"] = saved
    return mod


def build_targets() -> pl.DataFrame:
    """対象の1,620枚の表を作る（v5 の目視のラベルと，ex025 の画像の重みを付ける）．

    Returns:
        pl.DataFrame: 列 rep_odaiId, source（ex024 / ex025）, label_v5（お題文あり．bool）, w（ex025 の重み．ex024 は null）,
        stratum2, format_argmax, p, qwen_v5（ex025 の Qwen P3 の答え）を持つ表．
    """
    nm = _load_normal_miss()
    e25 = nm.build_set().select("rep_odaiId", pl.lit("ex025").alias("source"), pl.col("label").alias("label_v5"), "w", "stratum2",
                                "format_argmax", "p", pl.col("qwen").alias("qwen_v5"))
    t24 = EX024.load_truth().select(pl.col("fileId").alias("rep_odaiId"), pl.lit("ex024").alias("source"),
                                    pl.col("odaibun_attr").alias("label_v5"))
    t24 = t24.filter(~pl.col("rep_odaiId").is_in(e25["rep_odaiId"].implode()))
    return pl.concat([e25, t24], how="diagonal_relaxed")


def parse_label(code: str) -> dict:
    """目視のラベルの記号（例 "s,o:x,m:qk"）を属性の真偽に変える．

    Args:
        code (str): ラベルの記号．文字がない場合は "-"．

    Returns:
        dict[str, bool]: キー L_{層}（層があるか）と F_{層}_{形}（重ねた文字・余白の文字の形）の真偽．
    """
    rec = {f"L_{l}": False for l in LAYERS}
    rec.update({f"F_{l}_{f}": False for l in ["o", "m"] for f in FORMS})
    if code == "-":
        return rec
    for part in code.split(","):
        layer, _, forms = part.partition(":")
        assert layer in LAYERS, f"層の記号が不正：{code}"
        rec[f"L_{layer}"] = True
        for f in forms:
            assert layer in ("o", "m") and f in FORMS, f"形の記号が不正：{code}"
            rec[f"F_{layer}_{f}"] = True
    return rec


def load_labels(path: str = LABELS) -> pl.DataFrame:
    """目視で付けた v6 の属性を読み込む．

    Args:
        path (str): ラベルのファイル．書式は「番号 記号 [メモ]」（番号は targets.csv の no）．

    Returns:
        pl.DataFrame: 列 no, code, memo と parse_label の列を持つ表．
    """
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            parts = line.rstrip("\n").split(maxsplit=2)
            rows.append({"no": int(parts[0]), "code": parts[1], "memo": parts[2] if len(parts) > 2 else "", **parse_label(parts[1])})
    return pl.DataFrame(rows)


def add_derived(d: pl.DataFrame, prefix: str = "") -> pl.DataFrame:
    """文の属性から，層と形をまとめた属性とお題文あり（規則 R0 / R1 / R1b / R2，既定は R1b）を導く．

    Args:
        d (pl.DataFrame): F_{層}_{形} と L_{層} の列（prefix 付き）を持つ表．
        prefix (str): 列名の接頭辞（予測の列なら "pred_" など）．

    Returns:
        pl.DataFrame: 列 {prefix}A_{形}（重ねた文字か余白の文字にその形がある），{prefix}odai_R0, R1, R1b, R2 を加えた表．
    """
    c = lambda name: pl.col(prefix + name)  # noqa: E731
    any_form = {f: c(f"F_o_{f}") | c(f"F_m_{f}") for f in FORMS}
    task = any_form["q"] | any_form["i"] | any_form["t"]
    return d.with_columns(*[v.alias(f"{prefix}A_{f}") for f, v in any_form.items()], task.alias(f"{prefix}odai_R0"),
                          (task | c("F_m_x")).alias(f"{prefix}odai_R1"),
                          (task | (c("F_m_x") & ~c("F_m_k"))).alias(f"{prefix}odai_R1b"), (task | c("F_m_x") | c("F_o_x")).alias(f"{prefix}odai_R2"))


def load_all() -> pl.DataFrame:
    """対象の表に目視の v6 の属性と導いたお題文ありを付けて読み込む．

    Returns:
        pl.DataFrame: targets.csv の列と load_labels・add_derived の列を持つ表．
    """
    t = pl.read_csv(TARGETS).join(load_labels(), on="no", how="left")
    assert t["code"].null_count() == 0, "目視のラベルが欠けている"
    return add_derived(t)
