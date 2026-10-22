"""お題画像の種別判定の比較実験で用いるデータを扱うモジュール．

- BoketeDBv4 のメタデータ（`type` 列，お題タグ）を手がかりに，評価用の画像を層化して抽出する．
- 目視でラベルを付与するためのコンタクトシート（画像を格子状に並べた一覧画像）を作成する．
- 弱ラベル（`type` 列とタグ）による学習用の画像集合を抽出する．
- 目視で付与したラベル（`eval_labels.csv`）を読み込む．
"""
import argparse
import os

import polars as pl
from PIL import Image, ImageDraw, ImageFont

DB_DIR = "/mnt/d/Bokete_Scraping_v3.2/outputs/BoketeDBv4"
IMAGE_DIR = "/mnt/d/Bokete_Scraping_v3.2/datasets/images"
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "ex001_odai_image_type")
LABEL_PATH = os.path.join(os.path.dirname(__file__), "eval_labels.csv")

# 評価用の抽出の内訳（プール名: 件数）．random は全お題からの一様抽出である
EVAL_POOL_SIZES = {
    "random": 240,
    "type_anaume": 40,
    "type_text": 40,
    "tag_karuta": 40,
    "tag_manga": 30,
    "tag_anime": 30,
    "type_illust": 20,
}

# お題タグによる弱ラベルの定義
KARUTA_TAGS = ["カルタ", "かるた", "カルタお題", "かるたお題", "さとうのカルタお題", "カルタシリーズ", "かるたで一言"]
MANGA_TAGS = ["漫画", "マンガ", "漫画でボケて", "漫画の一コマ", "少女漫画", "4コマ", "四コマ", "4コマ漫画"]
ANIME_TAGS = ["アニメ", "アニメキャラ", "ドラえもん", "クレヨンしんちゃん", "野原ひろし", "サザエさん", "ちびまる子ちゃん",
              "ポケモン", "エヴァンゲリオン", "ワンピース", "ジブリ"]

# 2つの軸のラベル．形式（ボケの生成過程が変わるか）と画風（事前知識が必要か）
FORMAT_LABELS = ["normal", "text", "anaume", "karuta"]
DOMAIN_LABELS = ["photo", "anime", "manga", "illust"]
# 6クラス（＋その他）への統合後のラベル
FLAT_LABELS = ["photo", "anime", "manga", "text", "anaume", "karuta", "illust"]


def image_path(odai_id: int) -> str:
    """お題IDから画像ファイルのパスを求める．

    Args:
        odai_id (int): お題ID（`odai.fileId`）．

    Returns:
        str: 画像ファイルの絶対パス．
    """
    return os.path.join(IMAGE_DIR, str(odai_id // 10000 * 10000), f"{odai_id}.jpg")


def to_flat_label(format_label: str, domain_label: str) -> str:
    """形式と画風の2軸のラベルを7クラスのラベルに統合する．

    形式が normal 以外であれば形式を優先し，normal であれば画風をラベルとする．

    Args:
        format_label (str): 形式のラベル（FORMAT_LABELS のいずれか）．
        domain_label (str): 画風のラベル（DOMAIN_LABELS のいずれか）．

    Returns:
        str: 統合後のラベル（FLAT_LABELS のいずれか）．
    """
    return domain_label if format_label == "normal" else format_label


def load_odai_meta() -> pl.DataFrame:
    """お題のメタデータ（画像があるもの）と，お題ごとのタグの集合を読み込む．

    Returns:
        pl.DataFrame: 列 fileId (Int64), type (str), year (int), tags (list[str]) を持つ表．行数は画像があるお題の数．
    """
    odai = (
        pl.scan_parquet(os.path.join(DB_DIR, "odai", "*.parquet"))
        .filter(pl.col("imageStatus") == "present")
        .select("fileId", "type", pl.col("date").dt.year().alias("year"))
    )
    readable = pl.scan_parquet(os.path.join(DB_DIR, "image", "*.parquet")).filter(pl.col("readable")).select(
        pl.col("odaiId").alias("fileId"))
    tags = (
        pl.scan_parquet(os.path.join(DB_DIR, "odai_tag", "*.parquet"))
        .select(pl.col("odaiId").alias("fileId"), "name")
        .group_by("fileId").agg(pl.col("name").alias("tags"))
    )
    return odai.join(readable, on="fileId").join(tags, on="fileId", how="left").collect()


def _has_any_tag(tag_list: list) -> pl.Expr:
    """tags 列がタグの候補のいずれかを含むかを表す式を作る．

    Args:
        tag_list (list[str]): タグの候補．

    Returns:
        pl.Expr: 真偽値の式．
    """
    return pl.col("tags").list.eval(pl.element().is_in(tag_list)).list.any().fill_null(False)


def pool_filters() -> dict:
    """抽出のプール名と，そのプールに属するお題を選ぶ式の対応を返す．

    Returns:
        dict[str, pl.Expr]: プール名をキー，絞り込みの式を値とする辞書．
    """
    return {
        "random": pl.lit(True),
        "type_anaume": pl.col("type") == "穴埋め",
        "type_text": pl.col("type") == "文字",
        "tag_karuta": _has_any_tag(KARUTA_TAGS),
        "tag_manga": _has_any_tag(MANGA_TAGS),
        "tag_anime": _has_any_tag(ANIME_TAGS),
        "type_illust": pl.col("type") == "イラスト",
        "type_photo": pl.col("type") == "写真",
    }


def sample_eval_set(meta: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """評価用の画像をプールごとに重複なく抽出する．

    Args:
        meta (pl.DataFrame): load_odai_meta の戻り値．
        seed (int): 乱数のシード．

    Returns:
        pl.DataFrame: 列 fileId (Int64), pool (str) を持つ表．行数は EVAL_POOL_SIZES の総和．
    """
    filters = pool_filters()
    used = set()
    rows = []
    for pool, size in EVAL_POOL_SIZES.items():
        cand = meta.filter(filters[pool] & ~pl.col("fileId").is_in(list(used)))
        picked = cand.sample(n=size, seed=seed)["fileId"].to_list()
        used.update(picked)
        rows += [{"fileId": fid, "pool": pool} for fid in picked]
    return pl.DataFrame(rows)


def sample_weak_train_set(meta: pl.DataFrame, exclude: set, per_class: int = 1500, seed: int = 0) -> pl.DataFrame:
    """メタデータの弱ラベルによる学習用の画像を抽出する．

    `type` 列は2022年以降のお題で投稿者が選んだ種別を表すため，その期間のお題のみを用いる．
    タグによるクラス（karuta，manga，anime）は全期間から抽出する．

    Args:
        meta (pl.DataFrame): load_odai_meta の戻り値．
        exclude (set[int]): 除外するお題ID（評価用の画像）．
        per_class (int): 1クラスあたりの件数の上限．
        seed (int): 乱数のシード．

    Returns:
        pl.DataFrame: 列 fileId (Int64), weak_label (str) を持つ表．
    """
    filters = pool_filters()
    recent = pl.col("year") >= 2022
    karuta = filters["tag_karuta"]
    class_filters = {
        "karuta": karuta,
        "anaume": filters["type_anaume"] & recent & ~karuta,
        "text": filters["type_text"] & recent & ~karuta,
        "photo": filters["type_photo"] & recent,
        "illust": filters["type_illust"] & recent & ~filters["tag_manga"] & ~filters["tag_anime"],
        "manga": filters["tag_manga"] & ~karuta & (pl.col("type") != "穴埋め"),
        "anime": filters["tag_anime"] & ~karuta & (pl.col("type") != "穴埋め"),
    }
    base = meta.filter(~pl.col("fileId").is_in(list(exclude)))
    frames = []
    used = set()
    for label, expr in class_filters.items():
        cand = base.filter(expr & ~pl.col("fileId").is_in(list(used)))
        cand = cand.sample(n=min(per_class, len(cand)), seed=seed)
        used.update(cand["fileId"].to_list())
        frames.append(cand.select("fileId", pl.lit(label).alias("weak_label")))
    return pl.concat(frames)


def make_contact_sheets(ids: list, out_dir: str, cols: int = 5, rows: int = 4, tile: int = 300) -> list:
    """目視でラベルを付与するためのコンタクトシートを作成する．

    各タイルの左上に，シート内の番号とお題IDを描画する．

    Args:
        ids (list[int]): お題IDのリスト．
        out_dir (str): 出力先のディレクトリ．
        cols (int): 1シートの列数．
        rows (int): 1シートの行数．
        tile (int): 1タイルの一辺の画素数．

    Returns:
        list[str]: 作成したシートのパスのリスト．
    """
    os.makedirs(out_dir, exist_ok=True)
    font = _load_font(22)
    per_sheet = cols * rows
    paths = []
    for s in range(0, len(ids), per_sheet):
        sheet = Image.new("RGB", (cols * tile, rows * tile), "white")
        draw = ImageDraw.Draw(sheet)
        for k, fid in enumerate(ids[s:s + per_sheet]):
            x, y = (k % cols) * tile, (k // cols) * tile
            sheet.paste(_fit_tile(image_path(fid), tile - 6), (x + 3, y + 3))
            draw.rectangle([x + 3, y + 3, x + 150, y + 30], fill="black")
            draw.text((x + 6, y + 4), f"{k:02d}:{fid}", fill="yellow", font=font)
        path = os.path.join(out_dir, f"sheet_{s // per_sheet:03d}.jpg")
        sheet.save(path, quality=90)
        paths.append(path)
    return paths


def _fit_tile(path: str, size: int) -> Image.Image:
    """画像を縦横比を保って正方形のタイルに収める．

    Args:
        path (str): 画像のパス．
        size (int): タイルの一辺の画素数．

    Returns:
        Image.Image: size×size の RGB 画像．
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (200, 200, 200))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def _load_font(size: int) -> ImageFont.ImageFont:
    """描画に用いるフォントを読み込む．見つからない場合は既定のフォントを用いる．

    Args:
        size (int): 文字の大きさ．

    Returns:
        ImageFont.ImageFont: フォント．
    """
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_eval_labels() -> pl.DataFrame:
    """目視で付与した評価用のラベルを読み込み，7クラスのラベルを付け加える．

    Returns:
        pl.DataFrame: 列 fileId, pool, format, domain, flat を持つ表．
    """
    df = pl.read_csv(LABEL_PATH, comment_prefix="#")
    return df.with_columns(
        pl.struct("format", "domain").map_elements(
            lambda r: to_flat_label(r["format"], r["domain"]), return_dtype=pl.String).alias("flat"))


def main() -> None:
    """評価用と弱ラベルの学習用の画像を抽出し，コンタクトシートを作成する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(DATASET_DIR, exist_ok=True)

    meta = load_odai_meta()
    print("お題（画像あり）:", len(meta))
    eval_df = sample_eval_set(meta, seed=args.seed)
    eval_df.write_csv(os.path.join(DATASET_DIR, "eval_candidates.csv"))
    make_contact_sheets(eval_df["fileId"].to_list(), os.path.join(DATASET_DIR, "contact_sheets"))

    weak = sample_weak_train_set(meta, set(eval_df["fileId"].to_list()), seed=args.seed)
    weak.write_csv(os.path.join(DATASET_DIR, "weak_train.csv"))
    print(weak["weak_label"].value_counts())


if __name__ == "__main__":
    main()
