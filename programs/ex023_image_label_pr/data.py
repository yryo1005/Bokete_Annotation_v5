"""ex023（お題画像のラベルの判断者ごとの適合率・再現率）で用いるデータを扱うモジュール．

- 正解：Claude Code の目視のラベル（定義書 v4）．評価用データ640枚，実データの無作為抽出160枚，層別抽出220枚（ex022 の data.py で読み込む）．
- Qwen：Qwen3-VL-8B 単独のラベル（生徒の教師ラベル T_v11．ex008 の出力）．
- 生徒：パッケージの annotation.parquet の生徒の確率から，確認の質問を反映する前のラベルを再構成したもの．
- 全プロセス：パッケージの annotation.parquet のラベル（生徒＋確認の質問）．
- 全プロセス（属性から形式）：確認の質問が偽のとき，生徒の属性から形式を導く（カルタ記号あり→カルタ，空欄あり→穴埋め，それ以外→通常）場合の模擬．
"""
import importlib.util
import os
import sys

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG_DIR = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex023_image_label_pr")
ANNOTATION = os.path.join(PKG_DIR, "outputs", "BoketeDBv5_annotation", "annotation.parquet")
QWEN_DIR = os.path.join(PROJECT_DIR, "outputs", "ex008_label_v4", "teacher_T_v11", "rules", "0")
FORMATS = ["karuta", "anaume", "odaibun", "normal"]
ATTRIBUTES = ["has_blank", "has_karuta_mark", "has_text"]
MEDIA = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
LABEL_COLS = ["format"] + ATTRIBUTES + MEDIA
# 評価するカテゴリ（形式はクラスごとの2値として扱う）
CATEGORIES = [f"format_{f}" for f in FORMATS] + ATTRIBUTES + MEDIA
JA = {"format_karuta": "カルタ", "format_anaume": "穴埋め", "format_odaibun": "お題文", "format_normal": "通常",
      "has_blank": "空欄あり", "has_karuta_mark": "カルタ記号あり", "has_text": "文字あり", "photo": "写真", "anime": "アニメ",
      "game": "ゲーム", "manga": "マンガ", "text_graphic": "テキストグラフィック", "painting": "絵画", "illustration": "イラスト"}
AXIS = {**{c: "形式" for c in CATEGORIES[:4]}, **{c: "属性" for c in ATTRIBUTES}, **{c: "媒体" for c in MEDIA}}


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


EX022 = _load("ex022_data_for_ex023", os.path.join(PROJECT_DIR, "programs", "ex022_gemini_image_labels", "data.py"))
image_path = EX022.image_path


def to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """形式をクラスごとの2値の列に展開し，カテゴリの列（CATEGORIES）を bool にそろえる．

    Args:
        df (pl.DataFrame): 列 fileId と LABEL_COLS を持つ表．

    Returns:
        pl.DataFrame: 列 fileId と CATEGORIES（bool）を持つ表．
    """
    cols = [(pl.col("format") == f).alias(f"format_{f}") for f in FORMATS]
    cols += [pl.col(c).cast(pl.Utf8).str.to_lowercase().is_in(["true", "1"]).alias(c) for c in ATTRIBUTES + MEDIA]
    return df.select([pl.col("fileId").cast(pl.Int64)] + cols)


def load_truth() -> pl.DataFrame:
    """正解（Claude Code の目視のラベル）を読み込み，実データの重みを付ける．

    無作為抽出の重みは「母集団の枚数 / 160」，層別抽出の重みは抽出確率の逆数とする（ex009 の推定と同じ）．
    評価用データの重みは1とする．

    Returns:
        pl.DataFrame: 列 fileId, set（eval / random / strat）, w と CATEGORIES を持つ表（1,020行）．
    """
    c = EX022.load_claude_labels()
    n_pop = float(c.filter(pl.col("set") == "strat")["weight"].sum())
    n_rand = c.filter(pl.col("set") == "random").height
    w = pl.when(pl.col("set") == "random").then(n_pop / n_rand).when(pl.col("set") == "strat").then(pl.col("weight")).otherwise(1.0)
    c = c.with_columns(w.alias("w"), pl.col("stratum").fill_null(""))
    return pl.concat([c.select("set", "w", "stratum"), to_binary(c.select(["fileId"] + LABEL_COLS))], how="horizontal")


def load_qwen() -> pl.DataFrame:
    """Qwen3-VL-8B 単独のラベル（教師 T_v11）を読み込む．

    Returns:
        pl.DataFrame: 列 fileId と CATEGORIES を持つ表．
    """
    ev = pl.read_csv(os.path.join(QWEN_DIR, "predictions.csv"))
    ev = ev.select(["fileId"] + [pl.col(f"pred_{c}").alias(c) for c in LABEL_COLS])
    real = pl.read_csv(os.path.join(QWEN_DIR, "check_predictions.csv")).select(["fileId"] + LABEL_COLS)
    return pl.concat([to_binary(ev), to_binary(real)]).unique("fileId", keep="first")


def _labels_from_probs(a: pl.DataFrame, fmt: np.ndarray) -> pl.DataFrame:
    """形式の番号と，annotation.parquet の属性・媒体の確率から，パッケージの後処理でラベルを作る．

    Args:
        a (pl.DataFrame): annotation.parquet の行（列 odaiId と p_* を持つ）．
        fmt (np.ndarray): 形状 (N,) の形式の番号．

    Returns:
        pl.DataFrame: 列 fileId と LABEL_COLS を持つ表．
    """
    p_attr = a.select([f"p_{c}" for c in ATTRIBUTES]).to_numpy()
    p_media = a.select([f"p_{c}" for c in MEDIA]).to_numpy()
    lab = PP.to_labels(fmt, p_attr, p_media)
    return pl.concat([a.select(pl.col("odaiId").alias("fileId")), lab], how="horizontal")


# パッケージの後処理のモジュール（load_pipeline で読み込む）
PP = None


def _postprocess():
    """パッケージの後処理のモジュール（bokete_annotation.postprocess）を読み込む．

    Returns:
        module: bokete_annotation.postprocess．
    """
    if PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)
    from bokete_annotation import postprocess

    return postprocess


def load_pipeline(ids: list) -> dict:
    """生徒，全プロセス，全プロセス（属性から形式）のラベルを作る．

    Args:
        ids (list[int]): お題IDのリスト．

    Returns:
        dict[str, pl.DataFrame]: 判断者名 → 列 fileId と CATEGORIES を持つ表．
        "_info" には列 fileId, verify_asked, verify_answer を持つ表を入れる．
    """
    global PP
    PP = _postprocess()
    a = pl.scan_parquet(ANNOTATION).filter(pl.col("odaiId").is_in(ids)).collect().sort("odaiId")
    p_fmt = a.select([f"p_format_{f}" for f in FORMATS]).to_numpy()
    p_attr = a.select([f"p_{c}" for c in ATTRIBUTES]).to_numpy()
    ans = a["verify_answer"].fill_null(True).to_numpy()
    student = _labels_from_probs(a, p_fmt.argmax(1))
    full = _labels_from_probs(a, PP.decide_format(p_fmt, ans))
    # 確認の質問が偽のとき，生徒の属性（確率0.5以上）から形式を導く
    alt = np.where(p_attr[:, 1] >= 0.5, 0, np.where(p_attr[:, 0] >= 0.5, 1, 3))
    rejected = PP.verify_targets(p_fmt) & ~ans
    full_attr = _labels_from_probs(a, np.where(rejected, alt, p_fmt.argmax(1)))
    official = a.select(pl.col("odaiId").alias("fileId"), *LABEL_COLS)
    assert full.select(LABEL_COLS).equals(official.select(LABEL_COLS)), "再構成した全プロセスのラベルがパッケージの出力と一致しない"
    info = a.select(pl.col("odaiId").alias("fileId"), pl.col("verify_asked").fill_null(False), "verify_answer")
    return {"student": to_binary(student), "full": to_binary(full), "full_attr": to_binary(full_attr), "_info": info}
