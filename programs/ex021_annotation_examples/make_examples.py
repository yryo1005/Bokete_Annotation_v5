"""Claude Code が目視で付けたラベルの例を，抽出の層ごとに集める．

対象：
- お題画像：ex008 の実データの層別抽出220枚（定義書 v4．ex007 の生徒の予測で11層に分け，各層20枚）．
- 大喜利（ボケ）：ex013／ex014 の検証データ620件．
  - 420件：お題のラベルで7層に分け，各層12お題×5件（評価の上位3件と無作為の2件）．
  - 200件：少ない発話の型の候補（規則か教師がその型と予測したボケ）から，型ごとに40件．
  - 発話の型と画像外の知識は合意ラベル（ex016 の consensus_labels.txt，定義書 第3版）を用いる．
  - 制約の順守，必要な知識（4段階），画像中の文字への依存は ex013／ex014 のラベル（定義書 第1・2版）を用いる．

出力（.reports/figures_028/ と outputs/ex021_annotation_examples/）：
- img_{層}.png：お題画像の層ごとの例（5枚）
- boke_{層}.png：ボケの層ごとの例（4件．画像とボケの文）
- image_stratum_dist.csv，boke_stratum_dist.csv：層ごとのラベルの分布
- tables.md：レポートに載せる表（分布と例のラベル）

実行例:
    .venv_annotate/bin/python programs/ex021_annotation_examples/make_examples.py
"""
import importlib.util
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
FIG_DIR = os.path.join(PROJECT_DIR, ".reports", "figures_028")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex021_annotation_examples")
FONT = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
N_IMG, N_BOKE, SEED = 5, 4, 0

FORMAT_JA = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常"}
ATTR_JA = {"has_blank": "空欄", "has_karuta_mark": "カルタ記号", "has_text": "文字"}
MEDIA_JA = {"photo": "写真", "anime": "アニメ", "game": "ゲーム", "manga": "マンガ", "text_graphic": "テキストグラフィック",
            "painting": "絵画", "illustration": "イラスト"}
IMG_STRATA_JA = {"painting": "絵画", "karuta": "カルタ", "game": "ゲーム", "anime": "アニメ", "odaibun": "お題文",
                 "illustration": "イラスト", "text_graphic": "テキストグラフィック", "manga": "マンガ", "anaume": "穴埋め",
                 "text": "文字あり", "photo": "写真（残り）"}
BOKE_STRATA_JA = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "photo_text": "通常・写真・文字あり",
                  "photo_notext": "通常・写真・文字なし", "anime_manga": "通常・アニメかマンガ", "other": "通常・その他の媒体"}
VOICE_JA = {"S": "セリフ", "M": "心の声", "T": "ツッコミ・呼びかけ", "N": "状況の説明・物語", "L": "名付け・見立て", "A": "回答"}
VOICE3_JA = {"C": "登場人物の言葉", "X": "外からの言葉", "A": "回答"}
KNOW_JA = {"0": "見た目だけ", "G": "一般常識", "I": "画像中の対象", "O": "画像外の対象"}
COMP_JA = {"y": "はい", "n": "いいえ", "-": "対象外"}


def load_module(name: str, path: str):
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


D8 = load_module("ex008_data_for_ex021", os.path.join(PROJECT_DIR, "programs", "ex008_label_v4", "data.py"))
D16 = load_module("ex016_data_for_ex021", os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm", "data.py"))


def setup_font() -> None:
    """図の日本語フォントを設定する．

    Returns:
        None
    """
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def image_label_text(row: dict) -> str:
    """お題画像の目視ラベルを短い文字列にする．

    Args:
        row (dict): 列 format と属性3項目・媒体7項目（bool）を持つ1行．

    Returns:
        str: 「形式／属性／媒体」の文字列．
    """
    attrs = "・".join(v for k, v in ATTR_JA.items() if row[k]) or "なし"
    media = "・".join(v for k, v in MEDIA_JA.items() if row[k])
    return f"{FORMAT_JA[row['format']]}／{attrs}\n{media}"


def open_image(path: str) -> Image.Image:
    """画像を開き，表示用に縮小する．

    Args:
        path (str): 画像のパス．

    Returns:
        Image.Image: RGB の画像（長辺400画素以下）．画像がなければ灰色の画像．
    """
    if not os.path.exists(path):
        return Image.new("RGB", (400, 300), (200, 200, 200))
    im = Image.open(path).convert("RGB")
    im.thumbnail((400, 400))
    return im


def plot_image_row(rows: list, path: str) -> None:
    """お題画像の例を1行に並べて保存する．

    Args:
        rows (list[dict]): 列 fileId とラベルを持つ行のリスト（長さ N_IMG 以下）．
        path (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(1, N_IMG, figsize=(3.2 * N_IMG, 3.6))
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, rows):
        ax.imshow(open_image(D16.image_path(r["fileId"])))
        ax.set_title(f"{r['fileId']}\n{image_label_text(r)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def image_part() -> list:
    """お題画像の層ごとの分布と例を作る．

    Returns:
        list[str]: tables.md に書く行のリスト．
    """
    r = D8.load_real_labels("strat")
    cols = ["format"] + list(ATTR_JA) + list(MEDIA_JA)
    dist = r.group_by("stratum").agg(
        pl.len().alias("n"), pl.col("weight").first(),
        *[(pl.col("format") == f).sum().alias(f"format_{f}") for f in FORMAT_JA],
        *[pl.col(c).sum() for c in list(ATTR_JA) + list(MEDIA_JA)])
    order = list(IMG_STRATA_JA)
    dist = dist.with_columns(pl.col("stratum").replace_strict({s: i for i, s in enumerate(order)}).alias("o")).sort("o").drop("o")
    dist.write_csv(os.path.join(OUT_DIR, "image_stratum_dist.csv"))
    lines = ["## お題画像：層ごとのラベルの分布（各層20枚）", "",
             "| 層 | 母集団の枚数 | " + " | ".join(FORMAT_JA.values()) + " | " + " | ".join(ATTR_JA.values()) + " | "
             + " | ".join(MEDIA_JA.values()) + " |",
             "| :--- | ---: | " + " | ".join(["---:"] * (len(FORMAT_JA) + len(ATTR_JA) + len(MEDIA_JA))) + " |"]
    for d in dist.iter_rows(named=True):
        vals = [d[f"format_{f}"] for f in FORMAT_JA] + [d[c] for c in list(ATTR_JA) + list(MEDIA_JA)]
        lines.append(f"| {IMG_STRATA_JA[d['stratum']]} | {round(d['weight'] * d['n']):,} | " + " | ".join(map(str, vals)) + " |")
    for st in order:
        g = r.filter(pl.col("stratum") == st).sample(n=N_IMG, seed=SEED)
        plot_image_row(g.select(["fileId"] + cols).to_dicts(), os.path.join(FIG_DIR, f"img_{st}.png"))
    print("お題画像：", r.height, "枚，", len(order), "層")
    return lines


def pick_diverse(g: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """お題が重ならず，発話の型がなるべく異なるようにボケを n 件選ぶ．

    Args:
        g (pl.DataFrame): 1つの層のボケ（列 odaiId, voice を持つ）．
        n (int): 選ぶ件数．
        seed (int): 乱数の種．

    Returns:
        pl.DataFrame: 選んだ n 件（番号 no の順）．
    """
    g = g.sample(fraction=1.0, shuffle=True, seed=seed)
    picked, odai, voices = [], set(), set()
    for strict in (True, False):
        for row in g.iter_rows(named=True):
            if len(picked) >= n:
                break
            if row["no"] in picked or row["odaiId"] in odai or (strict and row["voice"] in voices):
                continue
            picked.append(row["no"])
            odai.add(row["odaiId"])
            voices.add(row["voice"])
    return g.filter(pl.col("no").is_in(picked)).sort("no")


def boke_label_text(r: dict) -> str:
    """ボケの目視ラベルを図に書く短い文字列にする．

    Args:
        r (dict): 列 voice, outside, know, comp, uses_text を持つ1行．

    Returns:
        str: 複数行の文字列．
    """
    return (f"発話の型：{VOICE_JA[r['voice']]}\n画像外の知識：{'はい' if r['outside'] else 'いいえ'}\n"
            f"必要な知識：{KNOW_JA[r['know']]}\n制約の順守：{COMP_JA[r['comp']]}\n"
            f"文字への依存：{'はい' if r['uses_text'] else 'いいえ'}")


def plot_boke_grid(rows: list, path: str) -> None:
    """ボケの例を，画像と文を並べて2列×2行で保存する．

    Args:
        rows (list[dict]): 列 no, odaiId, rep_odaiId, text を持つ行のリスト（長さ N_BOKE 以下）．
        path (str): 保存先．

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1, 1.1, 1, 1.1]})
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(rows):
        i, j = k // 2, (k % 2) * 2
        fid = r["rep_odaiId"] if r["rep_odaiId"] is not None else r["odaiId"]
        axes[i, j].imshow(open_image(D16.image_path(fid)))
        axes[i, j].set_title(f"No.{r['no']:03d}（お題 {r['odaiId']}）", fontsize=10)
        body = "\n".join(textwrap.wrap((r["text"] or "（空）").replace("\n", " "), 16))
        axes[i, j + 1].text(0.02, 0.95, body, va="top", fontsize=12)
        axes[i, j + 1].text(0.02, 0.05, boke_label_text(r), va="bottom", fontsize=9, color="#444444")
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def boke_example_table(g: pl.DataFrame) -> list:
    """ボケの例のラベルを表の行にする．

    Args:
        g (pl.DataFrame): 選んだボケ．

    Returns:
        list[str]: マークダウンの表の行．
    """
    lines = ["| 番号 | ボケ | 発話の型（3区分） | 画像外の知識 | 必要な知識（第1・2版） | 制約の順守 | 画像中の文字への依存 |",
             "| :---: | :--- | :--- | :---: | :--- | :---: | :---: |"]
    for r in g.iter_rows(named=True):
        text = (r["text"] or "（空）").replace("\n", " ").replace("|", "｜")
        lines.append(f"| {r['no']:03d} | {text} | {VOICE_JA[r['voice']]}（{VOICE3_JA[r['voice3']]}） | "
                     f"{'はい' if r['outside'] else 'いいえ'} | {KNOW_JA[r['know']]} | {COMP_JA[r['comp']]} | "
                     f"{'はい' if r['uses_text'] else 'いいえ'} |")
    return lines


def boke_dist_table(e: pl.DataFrame, key: str, names: dict) -> list:
    """層ごとのボケのラベルの分布の表を作る．

    Args:
        e (pl.DataFrame): ボケの検証データ（列 key を持つ）．
        key (str): 層の列名．
        names (dict): 層の名前 → 表示名．

    Returns:
        tuple[list[str], list[dict]]: マークダウンの表の行と，層ごとの集計（CSV 用）．
    """
    head =("| 層 | 件数 | " + " | ".join(VOICE_JA.values()) + " | 画像外の知識 | " + " | ".join(KNOW_JA.values())
            + " | 順守：いいえ | 文字への依存 |")
    lines = [head, "| :--- | ---: | " + " | ".join(["---:"] * (len(VOICE_JA) + len(KNOW_JA) + 3)) + " |"]
    rows = []
    for st in names:
        g = e.filter(pl.col(key) == st)
        v = [int((g["voice"] == c).sum()) for c in VOICE_JA]
        k = [int((g["know"] == c).sum()) for c in KNOW_JA]
        rec = {"stratum": st, "n": g.height, **dict(zip(VOICE_JA, v)), "outside": int(g["outside"].sum()),
               **{f"know_{c}": x for c, x in zip(KNOW_JA, k)}, "comp_n": int((g["comp"] == "n").sum()),
               "uses_text": int(g["uses_text"].sum())}
        rows.append(rec)
        lines.append(f"| {names[st]} | {g.height} | " + " | ".join(map(str, v)) + f" | {rec['outside']} | "
                     + " | ".join(map(str, k)) + f" | {rec['comp_n']} | {rec['uses_text']} |")
    return lines, rows


def inappropriate_nos() -> list:
    """ex013 の目視のラベルで性的・不適切（定義書 第1版）とされたボケの番号を返す．例の表示から除くために用いる．

    Returns:
        list[int]: ボケの番号のリスト．
    """
    path = os.path.join(D16.EX13_DIR, "eval_labels.txt")
    rows = [ln.split() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    return [int(r[0]) for r in rows if r[5] == "1"]


def boke_part() -> list:
    """ボケの層ごとの分布と例を作る．

    Returns:
        list[str]: tables.md に書く行のリスト．
    """
    e = D16.load_eval()
    cand = pl.read_csv(os.path.join(D16.EX14_DATASET, "eval2_bokes.csv")).select("no", "cand")
    e = e.join(cand, on="no", how="left")
    old, rare = e.filter(pl.col("part") == "old"), e.filter(pl.col("part") == "rare")
    rare_names = {c: f"候補：{VOICE_JA[c]}" for c in ["M", "T", "N", "L", "S"]}
    l1, r1 = boke_dist_table(old, "stratum", BOKE_STRATA_JA)
    l2, r2 = boke_dist_table(rare, "cand", rare_names)
    pl.DataFrame(r1 + r2).write_csv(os.path.join(OUT_DIR, "boke_stratum_dist.csv"))
    lines = ["## ボケ：お題の層ごとのラベルの分布（420件）", ""] + l1 + ["", "## ボケ：少ない型の候補ごとの分布（200件）", ""] + l2
    groups = [("stratum", st, old) for st in BOKE_STRATA_JA] + [("cand", c, rare) for c in rare_names]
    bad = inappropriate_nos()
    for key, st, d in groups:
        g = pick_diverse(d.filter((pl.col(key) == st) & ~pl.col("no").is_in(bad)), N_BOKE, SEED)
        name = BOKE_STRATA_JA.get(st) if key == "stratum" else rare_names[st]
        plot_boke_grid(g.to_dicts(), os.path.join(FIG_DIR, f"boke_{key}_{st}.png"))
        lines += ["", f"## ボケの例：{name}", ""] + boke_example_table(g)
    print("ボケ：", e.height, "件")
    return lines


def main() -> None:
    """お題画像とボケの例を集め，図と表を保存する．

    Returns:
        None
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_font()
    lines = image_part() + [""] + boke_part()
    with open(os.path.join(OUT_DIR, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
