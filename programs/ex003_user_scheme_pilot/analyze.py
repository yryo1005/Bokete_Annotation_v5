"""ユーザー案のパイロット（pilot.py）の結果を分析し，report_004 の図と表を作る．

1. ボケの書き出しによるカルタの判定：かなで始まるボケのうち最も多い書き出しの文字の割合（集中度）を求め，
   目視のカルタのラベルをどの程度判別できるかを調べる．
2. ボケの長さによる穴埋めの判定：ボケの文字数の中央値の分布を形式ごとに比べる．
3. 論点ごとの実例の図：テキストグラフィックの過剰な選択，媒体の複数選択，事前知識，カルタの見落とし，形式の食い違い．

出力は outputs/ex003_user_scheme_pilot/figures/ と analysis.json である．

実行例:
    .venv_annotate/bin/python programs/ex003_user_scheme_pilot/analyze.py
"""
import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot import DB_DIR, MEDIA, OUT_DIR, first_char, load_bokes  # noqa: E402

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "programs", "ex002_odai_type_distill"))
from analysis import plot_image_grid  # noqa: E402

FIG_DIR = os.path.join(PROJECT_DIR, "outputs", "ex003_user_scheme_pilot", "figures")
MIN_KANA_BOKES = 5
SHORT = {"photo": "写真", "anime": "アニメ", "game": "ゲーム", "cg": "CG", "manga": "マンガ", "text_graphic": "テキスト",
         "painting": "絵画", "illustration": "イラスト"}
FMT_JA = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常", "text": "お題文"}


def is_kana(ch: str) -> bool:
    """ひらがな（カタカナはひらがなに変換済み）かを判定する．

    Args:
        ch (str): 1文字．

    Returns:
        bool: ひらがなであれば真．
    """
    return "ぁ" <= ch <= "ゖ"


def kana_concentration(bokes: pl.DataFrame) -> pl.DataFrame:
    """お題ごとに，かなで始まるボケの書き出しの集中度を求める．

    Args:
        bokes (pl.DataFrame): 列 odaiId, text を持つボケの表．

    Returns:
        pl.DataFrame: 列 fileId, n_kana（かなで始まるボケの数），top_kana（最も多い書き出しの文字），
        concentration（その割合）を持つ表．
    """
    rows = []
    for (fid,), g in bokes.group_by("odaiId"):
        chars = [c for c in (first_char(t) for t in g["text"].fill_null("").to_list()) if is_kana(c)]
        if not chars:
            rows.append({"fileId": fid, "n_kana": 0, "top_kana": "", "concentration": None})
            continue
        ch, cnt = Counter(chars).most_common(1)[0]
        rows.append({"fileId": fid, "n_kana": len(chars), "top_kana": ch, "concentration": cnt / len(chars)})
    return pl.DataFrame(rows)


def setup_font() -> None:
    """図に日本語のフォントを設定する．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()


def plot_boke_signals(df: pl.DataFrame, path: str) -> dict:
    """目視の形式ごとに，ボケの書き出しの集中度とボケの文字数の中央値の分布を描く．

    Args:
        df (pl.DataFrame): 列 hand_format, concentration, n_kana, median_len を持つ表．
        path (str): 保存先のパス．

    Returns:
        dict: カルタの判別の AUC と，しきい値ごとの適合率・再現率．
    """
    setup_font()
    fmts = ["normal", "text", "anaume", "karuta"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    d = df.filter(pl.col("n_kana") >= MIN_KANA_BOKES)
    axes[0].boxplot([d.filter(pl.col("hand_format") == f)["concentration"].to_list() for f in fmts],
                    tick_labels=[FMT_JA[f] for f in fmts])
    axes[0].set_ylabel("かなで始まるボケのうち最多の書き出しの文字の割合")
    axes[0].set_xlabel(f"目視の形式（かなで始まるボケが{MIN_KANA_BOKES}件以上の画像）")
    axes[1].boxplot([df.filter(pl.col("hand_format") == f)["median_len"].drop_nulls().to_list() for f in fmts],
                    tick_labels=[FMT_JA[f] for f in fmts])
    axes[1].set_ylabel("ボケの文字数の中央値")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("目視の形式")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    y = (d["hand_format"] == "karuta").to_numpy()
    s = d["concentration"].to_numpy()
    out = {"n_images": int(len(d)), "n_karuta": int(y.sum()), "auc": float(roc_auc_score(y, s)), "thresholds": []}
    for th in [0.3, 0.4, 0.5, 0.6]:
        p = s >= th
        out["thresholds"].append({"threshold": th, "precision": float((p & y).sum() / max(p.sum(), 1)),
                                  "recall": float((p & y).sum() / y.sum()), "n_pred": int(p.sum())})
    return out


def media_str(r: dict) -> str:
    """媒体の真偽値から表示用の文字列を作る．

    Args:
        r (dict): 予測の行．

    Returns:
        str: 例 "写真+テキスト"．
    """
    return "+".join(SHORT[m] for m in MEDIA if r[m]) or "なし"


def grid(df: pl.DataFrame, ids: list, suptitle: str, name: str, extra: callable) -> None:
    """指定したお題IDの画像を，現行の目視ラベルとパイロットの出力を添えて図示する．

    Args:
        df (pl.DataFrame): 予測の表．
        ids (list[int]): お題IDのリスト．
        suptitle (str): 図の題．
        name (str): 保存するファイル名（拡張子なし）．
        extra (callable): 行を受け取り，題に追加する文字列を返す関数．

    Returns:
        None
    """
    titles = []
    for fid in ids:
        r = df.filter(pl.col("fileId") == fid).row(0, named=True)
        titles.append(f"id={fid} 目視:{FMT_JA[r['hand_format']]}/{r['hand_domain']}\n"
                      f"案:{FMT_JA[r['format']]} 媒体:{media_str(r)}\n{extra(r)}")
    plot_image_grid(ids, titles, suptitle, os.path.join(FIG_DIR, f"{name}.png"), ncols=4)


def pick(df: pl.DataFrame, expr: pl.Expr, n: int, seed: int = 0) -> list:
    """条件に合う画像から無作為に選ぶ．

    Args:
        df (pl.DataFrame): 予測の表．
        expr (pl.Expr): 条件．
        n (int): 選ぶ数．
        seed (int): 乱数のシード．

    Returns:
        list[int]: お題IDのリスト．
    """
    d = df.filter(expr)
    return d.sample(n=min(n, len(d)), seed=seed)["fileId"].to_list()


def bokes_line(r: dict, width: int = 18) -> str:
    """評価の高いボケを短く切って1行にする．

    Args:
        r (dict): 予測の行．
        width (int): 1件あたりの最大文字数．

    Returns:
        str: 表示用の文字列．
    """
    return "ボケ: " + " / ".join(b.strip()[:width] for b in (r["top_bokes"] or "").split(" / ")[:2])


def main() -> None:
    """分析を実行して図と analysis.json を保存する．

    Returns:
        None
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pl.read_csv(os.path.join(OUT_DIR, "predictions.csv"))
    conc = kana_concentration(load_bokes(df["fileId"].to_list()))
    df = df.join(conc, on="fileId", how="left")
    df.write_csv(os.path.join(OUT_DIR, "predictions_with_boke_signals.csv"))
    result = {"karuta_by_boke": plot_boke_signals(df, os.path.join(FIG_DIR, "fig_boke_signals.png"))}

    # テキストグラフィックの過剰な選択
    grid(df, pick(df, pl.col("text_graphic") & (pl.col("photo") | pl.col("manga") | pl.col("anime")), 8),
         "「テキストグラフィック」が他の媒体と同時に選ばれた画像", "fig_text_graphic", lambda r: "")
    # 媒体の複数選択（テキスト以外の組み合わせ）
    others = [m for m in MEDIA if m != "text_graphic"]
    multi = pl.sum_horizontal([pl.col(m).cast(int) for m in others]) >= 2
    grid(df, pick(df, multi, 8, seed=1), "テキスト以外の媒体が複数選ばれた画像", "fig_multi_media", lambda r: "")
    # 事前知識
    grid(df, pick(df, pl.col("known_person") & (pl.col("hand_domain") == "photo"), 8, seed=2),
         "「実在の著名人」が選ばれた写真", "fig_known_person", lambda r: "")
    grid(df, pick(df, (pl.col("hand_domain") == "manga") & (pl.col("format") == "normal"), 8, seed=3),
         "漫画のコマ（通常）と「特定の作品」の判断", "fig_known_work_manga",
         lambda r: f"特定の作品: {'あり' if r['known_work'] else 'なし'}")
    # カルタ：VLM が見落としたがボケが裏付ける画像
    miss = (pl.col("hand_format") == "karuta") & (pl.col("format") != "karuta")
    grid(df, df.filter(miss).sort("concentration", descending=True, nulls_last=True).head(8)["fileId"].to_list(),
         "目視ではカルタ，VLM はカルタ以外と判定した画像とボケの書き出し", "fig_karuta_missed",
         lambda r: f"最多の書き出し「{r['top_kana']}」{(r['concentration'] or 0):.0%}\n{bokes_line(r)}")
    # 形式の食い違い：目視の穴埋めとお題文を VLM が通常とした画像
    grid(df, pick(df, (pl.col("hand_format").is_in(["anaume", "text"])) & (pl.col("format") == "normal"), 8, seed=4),
         "目視では穴埋め・お題文，VLM は通常と判定した画像と実際のボケ", "fig_format_disagree",
         lambda r: bokes_line(r))
    # 偽陽性の確認：通常の画像でボケの書き出しが集中したもの
    fp = df.filter((pl.col("hand_format") != "karuta") & (pl.col("n_kana") >= MIN_KANA_BOKES)
                   & (pl.col("concentration") >= 0.4))
    result["non_karuta_high_concentration"] = fp.select("fileId", "hand_format", "top_kana", "concentration",
                                                        "n_kana").to_dicts()
    result["text_graphic_combos"] = (df.filter(pl.col("text_graphic")).with_columns(
        pl.concat_str([pl.when(pl.col(m)).then(pl.lit(m)) for m in others], separator="+", ignore_nulls=True)
        .alias("with")).group_by("with").len().sort("len", descending=True).to_dicts())
    json.dump(result, open(os.path.join(OUT_DIR, "analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(result["karuta_by_boke"], ensure_ascii=False, indent=1))
    print("カルタ以外で集中度0.4以上:", len(fp))


if __name__ == "__main__":
    main()
