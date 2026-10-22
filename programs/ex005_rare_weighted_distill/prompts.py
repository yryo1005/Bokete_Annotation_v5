"""ex005 で比較する教師（Qwen3-VL-8B）のプロンプトの部品．

各部品は，プロンプトの文，出力の JSON スキーマの項目（生成の順），入力の最大画素数，出力を3軸の値に変換する関数を持つ．
教師はこれらの部品の組み合わせ（形式をどの部品から，属性・媒体をどの部品から取るか）として構成する（combine.py）．

- F_v2：ex002 のプロンプト v2（形式と画風の2軸．形式のみを用いる）
- F_v6：定義書 v2 の形式のみを問う短いプロンプト（テロップの扱いを明記）
- F_v6_hr：F_v6 を高い解像度（最大 672×672 画素）で入力したもの
- F_v6_cot：F_v6 の前に，画像中の文字とその出所（投稿者が重畳したか）を書き出させてから形式を答えさせるもの
- J_v6：3軸をまとめて問うプロンプト（ex004 の v5 から事前知識を除き，テロップの扱いを加えたもの）．形式は属性から導く
- A_v6：属性のみを問うプロンプト
- M_v6：媒体のみを問うプロンプト
- J_v7，M_v7：J_v6，M_v6 の調整用データ（dev）での誤り（合成画像の過剰検出，テキストグラフィックとイラストの重複，
  絵画の見落とし）を受けて，定義を具体化した第2版
"""
import ast
import os

from data import ATTRIBUTES, FORMATS, MEDIA, PROJECT_DIR


def _ex001_prompt_v2() -> str:
    """ex001 の model.py のソースからプロンプト v2 の文字列を取り出す（モジュール名の衝突を避けるため import しない）．

    Returns:
        str: プロンプト v2．
    """
    src = open(os.path.join(PROJECT_DIR, "programs", "ex001_odai_image_type", "model.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_VLM_PROMPT_V2":
            return node.value.value
    raise ValueError("プロンプト v2 が見つからない")


HEAD = "この画像は大喜利サイト「ボケて」のお題画像です．"
BASE_PIXELS, HIGH_PIXELS = 448 * 448, 672 * 672

FORMAT_RULES = """形式（format）を，以下の順に判定し，最初に当てはまったものを1つ選んでください．
1. karuta（カルタ）：円や四角の枠の中にひらがな1文字だけを書いた，文から独立した札の記号がある（多くは画像の隅）．空欄の前後のひらがな（「□い。」など），文章の一部のひらがな，中身が空の枠はカルタ記号ではない．空欄も含む場合もカルタとする．
2. anaume（穴埋め）：語の代わりに置かれた空欄がある（白や色付きの矩形，○○，□，＿＿，語の代わりの「？」，語の代わりの黒塗り，話者に付いた空の吹き出し）．テロップや字幕の中の空欄も含む．顔や個人情報の目隠し，文末の疑問符は空欄ではない．
3. odaibun（お題文）：投稿者が画像に重畳した，回答者への問いや指示の文がある．問い（〜とは？），指示（〜してください），題目の句（〜あるある，〜な事，〜の共通点，〜にありがちなこと），状況の設定（〜なのに…），名前などの列挙を含む．ボケてのテンプレートの枠内の文字，無地の背景に置いた文字，画像の上下に付け足した文字が該当する．テレビのテロップや字幕，看板，商品の表示，漫画やゲームの台詞，本や DVD の題名など，投稿者が重畳したのではない文字は，問いの形をしていてもお題文ではない．
4. normal（通常）：上記のいずれでもない．"""

ATTR_RULES = """属性（当てはまるものを true）：
- has_karuta_mark：カルタ記号がある（円や四角の枠の中にひらがな1文字だけを書いた，文から独立した記号．中身が空の枠や文章の一部の文字は該当しない．迷ったら false）．karuta_char にそのひらがなを書く（なければ空文字）．
- has_blank：語の代わりに置かれた空欄がある（白や色付きの矩形，○○，□，＿＿，語の代わりの「？」，語の代わりの黒塗り，話者に付いた空の吹き出し．テロップや字幕の中の空欄も含む）．顔や個人情報の目隠し，文末の疑問符は空欄ではない．
- has_prompt_text：投稿者が画像に重畳した，回答者への問いや指示の文がある（問い，指示，題目の句，状況の設定，名前などの列挙）．空欄を含む文そのものは該当せず，空欄とは別に問いや指示の文がある場合のみ true．テレビのテロップや字幕，看板，台詞，題名など投稿者が重畳したのではない文字は該当しない．迷ったら false．
- has_other_text：お題文とカルタ記号以外の，ボケに関わりうる文字がある（テロップ，字幕，看板，台詞など．透かし，ロゴ，判読できない小さな文字は除く）．
- composite：合成画像（複数の画像の貼り合わせ，写真と絵の重ね合わせ）．"""

MEDIA_RULES = """媒体（原則1つだけ true．合成画像で異なる媒体の領域が画像の約2割以上ずつある場合のみ複数）：
- photo：現実の写真（実写の映像，フィギュアや人形，着ぐるみ，像を撮影したものを含む）
- anime：アニメのスクリーンショット
- game：ゲーム画面のキャプチャ（ゲームの会話画面，戦闘画面，ドット絵のゲーム画面を含む）
- cg：ゲームやアニメの画面ではない3DCG
- manga：漫画作品から切り抜いたコマ
- text_graphic：文字や記号が主体で，写真や絵がほとんどない画像（ウェブページやアプリの画面を含む）
- painting：古典的絵画（浮世絵などの版画を含み，写真に撮ったものも絵画とする）
- illustration：上記に含まれない絵（キャラクターの描き下ろし，模写，漫画風の一枚絵，カルタの札の絵，手描きの絵を含む）
生成AIによる画像は，同じ見た目の画像を人が作った場合の媒体とする．"""


def _bool_fields(names: list) -> dict:
    """真偽値の項目のスキーマを作る．

    Args:
        names (list[str]): 項目名のリスト．

    Returns:
        dict: 項目名をキー，{"type": "boolean"} を値とする辞書．
    """
    return {n: {"type": "boolean"} for n in names}


FMT_ENUM = {"type": "string", "enum": FORMATS}
ATTR_FIELDS = {"has_karuta_mark": {"type": "boolean"}, "karuta_char": {"type": "string", "maxLength": 1},
               **_bool_fields(["has_blank", "has_prompt_text", "has_other_text", "composite"])}

COMPONENTS = {
    "F_v2": {"prompt": _ex001_prompt_v2(), "max_pixels": BASE_PIXELS,
             "fields": {"format": {"type": "string", "enum": ["normal", "text", "anaume", "karuta"]},
                        "domain": {"type": "string", "enum": ["photo", "anime", "manga", "illust"]}}},
    "F_v6": {"prompt": HEAD + FORMAT_RULES + "\nJSON のみを出力してください．", "max_pixels": BASE_PIXELS,
             "fields": {"format": FMT_ENUM}},
    "F_v6_hr": {"prompt": HEAD + FORMAT_RULES + "\nJSON のみを出力してください．", "max_pixels": HIGH_PIXELS,
                "fields": {"format": FMT_ENUM}},
    "F_v6_cot": {"prompt": HEAD + "まず texts に画像中の主な文字（最大5つ，各20文字以内）を書き出し，それぞれ source に"
                 "投稿者が重畳した文字なら overlaid，テロップ・字幕・看板・台詞など元からある文字なら original を書いてください．"
                 "次に has_blank と has_karuta_mark を判定し，最後に形式を答えてください．\n" + FORMAT_RULES
                 + "\nJSON のみを出力してください．", "max_pixels": BASE_PIXELS,
                 "fields": {"texts": {"type": "array", "maxItems": 5, "items": {
                     "type": "object", "properties": {"text": {"type": "string", "maxLength": 20},
                                                      "source": {"type": "string", "enum": ["overlaid", "original"]}},
                     "required": ["text", "source"], "additionalProperties": False}},
                     "has_blank": {"type": "boolean"}, "has_karuta_mark": {"type": "boolean"}, "format": FMT_ENUM}},
    "J_v6": {"prompt": HEAD + "以下の項目を上から順に判定し，JSON のみを出力してください．\n\n## 1. " + ATTR_RULES
             + "\n\n## 2. 形式\n" + FORMAT_RULES + "\n\n## 3. " + MEDIA_RULES, "max_pixels": BASE_PIXELS,
             "fields": {**ATTR_FIELDS, "format": FMT_ENUM, **_bool_fields(MEDIA)}, "derive_format": True},
    "A_v6": {"prompt": HEAD + ATTR_RULES + "\nJSON のみを出力してください．", "max_pixels": BASE_PIXELS,
             "fields": ATTR_FIELDS},
    "M_v6": {"prompt": HEAD + MEDIA_RULES + "\nJSON のみを出力してください．", "max_pixels": BASE_PIXELS,
             "fields": _bool_fields(MEDIA)},
}


ATTR_RULES_V7 = ATTR_RULES.replace(
    "- has_other_text：お題文とカルタ記号以外の，ボケに関わりうる文字がある（テロップ，字幕，看板，台詞など．透かし，ロゴ，判読できない小さな文字は除く）．",
    "- has_other_text：お題文とカルタ記号以外の，ボケに関わりうる文字がはっきり読める（テロップ，字幕，看板，吹き出しの台詞，"
    "商品名など）．透かし，ロゴ，著作権表示，背景の小さな文字，判読できない文字は除く．",
).replace(
    "- composite：合成画像（複数の画像の貼り合わせ，写真と絵の重ね合わせ）．",
    "- composite：合成画像．2枚以上の別々の写真や絵を並べたり貼り合わせたりした画像や，写真に別の写真や絵を切り貼りした画像に限る．"
    "1枚の画像に文字，空欄，テロップ，枠を書き加えただけのもの，ポスター，パッケージ，ゲーム画面，番組の画面のデザインは"
    "合成画像ではない．迷ったら false．",
)
MEDIA_RULES_V7 = MEDIA_RULES.replace(
    "- text_graphic：文字や記号が主体で，写真や絵がほとんどない画像（ウェブページやアプリの画面を含む）",
    "- text_graphic：文字や記号が主体で，写真や絵がほとんどない画像（ウェブページやアプリの画面を含む）．"
    "写真や絵の上に文字やテロップが書かれているだけの場合は text_graphic ではない．",
).replace(
    "- painting：古典的絵画（浮世絵などの版画を含み，写真に撮ったものも絵画とする）",
    "- painting：古典的絵画．油彩や水彩などで描かれた美術作品（西洋画，日本画，肖像画，宗教画，浮世絵などの版画）．"
    "写真に撮ったものも絵画とする",
).replace(
    "- illustration：上記に含まれない絵（キャラクターの描き下ろし，模写，漫画風の一枚絵，カルタの札の絵，手描きの絵を含む）",
    "- illustration：上記に含まれない絵（キャラクターの描き下ろし，模写，漫画風の一枚絵，カルタの札の絵，手描きの絵，"
    "アイコンやロゴの絵を含む）．ボケてのテンプレートの枠に描かれた小さなマスコットの絵は数えない",
)
COMPONENTS["J_v7"] = {"prompt": HEAD + "以下の項目を上から順に判定し，JSON のみを出力してください．\n\n## 1. " + ATTR_RULES_V7
                      + "\n\n## 2. 形式\n" + FORMAT_RULES + "\n\n## 3. " + MEDIA_RULES_V7, "max_pixels": BASE_PIXELS,
                      "fields": {**ATTR_FIELDS, "format": FMT_ENUM, **_bool_fields(MEDIA)}, "derive_format": True}
COMPONENTS["M_v7"] = {"prompt": HEAD + MEDIA_RULES_V7 + "\nJSON のみを出力してください．", "max_pixels": BASE_PIXELS,
                      "fields": _bool_fields(MEDIA)}


def json_schema(name: str) -> dict:
    """部品の出力の JSON スキーマを作る．

    Args:
        name (str): 部品の名前．

    Returns:
        dict: 全項目を必須とする JSON スキーマ（項目は生成の順に並ぶ）．
    """
    props = COMPONENTS[name]["fields"]
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def parse(name: str, rec: dict) -> dict:
    """部品の出力を3軸の値（部品が持つ項目のみ）に変換する．

    形式は F_v2 の text を odaibun に読み替え，J_v6 では属性から定義の優先順位で導く．
    カルタ記号ありでも karuta_char がひらがなでない場合は，カルタ記号なしとする．

    Args:
        name (str): 部品の名前．
        rec (dict): 出力の JSON を解釈した辞書（解釈できない場合は空の辞書）．

    Returns:
        dict: 部品が持つ項目（format，karuta_char，属性，媒体）の値．
    """
    out = {}
    fields = COMPONENTS[name]["fields"]
    if "karuta_char" in fields:
        out["karuta_char"] = rec.get("karuta_char", "") or ""
    for k in ATTRIBUTES + MEDIA:
        if k in fields:
            out[k] = bool(rec.get(k, False))
    if "has_karuta_mark" in out and "karuta_char" in out and not ("ぁ" <= out["karuta_char"] <= "ゖ"):
        out["has_karuta_mark"] = False
    if "format" in fields:
        out["format"] = {"text": "odaibun"}.get(rec.get("format", "normal"), rec.get("format", "normal"))
    if COMPONENTS[name].get("derive_format"):
        out["format"] = next((f for a, f in [("has_karuta_mark", "karuta"), ("has_blank", "anaume"),
                                             ("has_prompt_text", "odaibun")] if out.get(a)), "normal")
    return out


# ===== order_009 の改善策（v8）．英語版（_en）と日本語版（_ja）を作り，調整用データで比較する =====
_V2 = _ex001_prompt_v2()
F_V8_EN = _V2.replace(
    '- "text": the image consists almost entirely of written text (typically a topic or question on a plain or colored background) and has no main photo, anime or manga picture.',
    '- "text": the poster of the image has overlaid a question or instruction for the respondents (e.g. "What is ...?", "Please ...", '
    '"... that you often see", "the common point of ...", a situation setting, or a list of names). A photo or picture may also be present. '
    'TV captions, subtitles, signs, product labels, speech in manga or games and titles of books are NOT such text.',
).replace(
    '- "normal": an ordinary photo, anime, manga or illustration. Pictures that contain captions, subtitles, signs, titles or speech bubbles with words are still "normal".',
    '- "normal": none of the above. Pictures that only contain TV captions, subtitles, signs, titles or speech bubbles with words are still "normal".',
)
COMPONENTS["F_v8_en"] = {"prompt": F_V8_EN, "max_pixels": BASE_PIXELS, "fields": COMPONENTS["F_v2"]["fields"]}

COMPOSITE_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Answer with JSON only.\n'
                '- n_separate_pictures: how many separate photos or pictures are placed side by side or pasted together, '
                'divided by visible borders, frames or clearly different backgrounds (1 if it is a single picture). Text, blank boxes, '
                'captions or frames added on top of one picture do not count as extra pictures. A poster, package, game screen or TV '
                'screen design counts as one picture.\n'
                '- has_cutout_paste: true if a cut-out of another photo or drawing (e.g. a face or an object) has been pasted onto '
                'the picture.')
COMPOSITE_JA = (HEAD + "JSON のみを出力してください．\n"
                "- n_separate_pictures：境界線，枠，はっきり異なる背景で区切られ，並べたり貼り合わせたりされた別々の写真や絵の数"
                "（1枚の画像なら1）．1枚の画像に書き加えた文字，空欄，テロップ，枠は数えない．ポスター，パッケージ，ゲーム画面，"
                "番組の画面のデザインは1枚と数える．\n"
                "- has_cutout_paste：別の写真や絵の切り抜き（顔や物など）が画像に貼り付けられていれば true．")
_COMP_FIELDS = {"n_separate_pictures": {"type": "integer", "minimum": 1, "maximum": 20},
                "has_cutout_paste": {"type": "boolean"}}
COMPONENTS["C_v8_en"] = {"prompt": COMPOSITE_EN, "max_pixels": BASE_PIXELS, "fields": _COMP_FIELDS}
COMPONENTS["C_v8_ja"] = {"prompt": COMPOSITE_JA, "max_pixels": BASE_PIXELS, "fields": _COMP_FIELDS}

PC_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Answer two independent questions with JSON only.\n'
         '- is_fine_art_painting: is the main picture a work of fine art painted by a painter, such as an oil or watercolor painting, '
         'a classical portrait, a religious painting, a Japanese painting or an ukiyo-e print (including a photo of such a work and an '
         'edited version of a famous painting)? Anime, manga, cartoons, illustrations for books or games, and digital character art are NOT.\n'
         '- is_3dcg_or_ai: is the main picture rendered with 3D computer graphics software or generated by AI (e.g. a 3D rendered '
         'character or scene, a photorealistic AI image)? Screenshots of video games or anime, real photographs and hand drawings are NOT.')
PC_JA = (HEAD + "次の2つの独立した質問に，JSON のみで答えてください．\n"
         "- is_fine_art_painting：主な絵は，画家が描いた美術作品か（油彩画，水彩画，古典的な肖像画，宗教画，日本画，浮世絵などの版画．"
         "それらを写真に撮ったものや，名画を加工したものを含む）．アニメ，漫画，カートゥーン，本やゲームの挿絵，デジタルのキャラクターの絵は該当しない．\n"
         "- is_3dcg_or_ai：主な絵は，3DCG ソフトで作られたか，生成AIで作られたか（3DCG のキャラクターや場面，写実的な生成AIの画像など）．"
         "ゲームやアニメの画面のキャプチャ，実際の写真，手描きの絵は該当しない．")
_PC_FIELDS = {"is_fine_art_painting": {"type": "boolean"}, "is_3dcg_or_ai": {"type": "boolean"}}
COMPONENTS["P_v8_en"] = {"prompt": PC_EN, "max_pixels": BASE_PIXELS, "fields": _PC_FIELDS}
COMPONENTS["P_v8_ja"] = {"prompt": PC_JA, "max_pixels": BASE_PIXELS, "fields": _PC_FIELDS}

TL_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Look at the written text in the image and answer with JSON only.\n'
         '- text_layout: where is the main sentence written? "bokete_template" (inside the Bokete template frame with the bokete logo), '
         '"added_band" (in a band or margin added above or below the picture), "plain_background" (on a plain or simple colored background '
         'with no main picture), "overlaid_on_picture" (typed by the poster directly on top of the picture), "original_text" (text that is '
         'part of the original picture: TV captions, subtitles, signs, product labels, speech balloons in manga or games, book titles), '
         'or "none" (no readable sentence).\n'
         '- addresses_respondents: does that sentence ask a question or give an instruction or topic to the respondents of the comedy '
         'contest (e.g. "What is ...?", "Please ...", "... that you often see", a situation setting, a list of names)?')
TL_JA = (HEAD + "画像中の文字を見て，JSON のみで答えてください．\n"
         "- text_layout：主な文はどこに書かれているか．\"bokete_template\"（ボケてのロゴのあるテンプレートの枠内），"
         "\"added_band\"（画像の上下に付け足した帯や余白），\"plain_background\"（主な絵のない無地や単色の背景），"
         "\"overlaid_on_picture\"（投稿者が絵の上に直接書き加えた），\"original_text\"（テレビのテロップや字幕，看板，商品の表示，"
         "漫画やゲームの台詞，本の題名など，元の画像にある文字），\"none\"（読める文がない）．\n"
         "- addresses_respondents：その文は，大喜利の回答者に向けた問い，指示，題目（〜とは？，〜してください，〜あるある，"
         "状況の設定，名前の列挙など）か．")
_TL_FIELDS = {"text_layout": {"type": "string", "enum": ["bokete_template", "added_band", "plain_background",
                                                          "overlaid_on_picture", "original_text", "none"]},
              "addresses_respondents": {"type": "boolean"}}
COMPONENTS["T_v8_en"] = {"prompt": TL_EN, "max_pixels": BASE_PIXELS, "fields": _TL_FIELDS}
COMPONENTS["T_v8_ja"] = {"prompt": TL_JA, "max_pixels": BASE_PIXELS, "fields": _TL_FIELDS}

M_V8_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Choose its medium (usually exactly one true; '
           'several only if it is a composite where regions of different media each cover about 20% or more). Answer with JSON only.\n'
           '- photo: a real photograph (including live-action video frames and photos of figures, dolls, costumes or statues)\n'
           '- anime: a screenshot of an animated TV series or film\n'
           '- game: a screenshot of a video game (dialogue screens, battle screens, pixel-art game screens)\n'
           '- cg: 3D computer graphics that are not a game or anime screenshot\n'
           '- manga: a panel cut out from a manga work\n'
           '- text_graphic: mainly text or symbols with almost no photo or picture (including web pages and app screens)\n'
           '- painting: a classical painting (including woodblock prints such as ukiyo-e, and photos of paintings)\n'
           '- illustration: any other drawing (newly drawn characters, copies, single cartoon-style pictures, karuta card pictures, hand drawings)\n'
           'For AI-generated images, choose the medium that the same-looking image would have if made by a human.')
COMPONENTS["M_v8_en"] = {"prompt": M_V8_EN, "max_pixels": BASE_PIXELS, "fields": _bool_fields(MEDIA)}


def parse_v8(name: str, rec: dict) -> dict:
    """v8 の補助の質問の部品の出力をそのまま返す（parse の対象外の項目を持つため）．

    Args:
        name (str): 部品の名前．
        rec (dict): 出力の JSON を解釈した辞書．

    Returns:
        dict: 部品の項目の値（不足する項目は既定値）．
    """
    defaults = {"integer": 1, "boolean": False, "string": "none"}
    return {k: rec.get(k, defaults[v["type"]]) for k, v in COMPONENTS[name]["fields"].items()}


_parse_base = parse


def parse(name: str, rec: dict) -> dict:  # noqa: F811
    """部品の出力を変換する．v8 の補助の質問の部品は parse_v8，それ以外は従来の変換を用いる．

    Args:
        name (str): 部品の名前．
        rec (dict): 出力の JSON を解釈した辞書．

    Returns:
        dict: 部品の項目の値．
    """
    if name.startswith(("C_v8", "P_v8", "T_v8")):
        return parse_v8(name, rec)
    return _parse_base(name, rec)

J_V8_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Judge the items below from top to bottom and answer with JSON only.\n\n'
           '## 1. Attributes (true if applicable)\n'
           '- has_karuta_mark: there is a karuta mark, i.e. a circle or square frame containing exactly one hiragana character, independent '
           'of any sentence (usually in a corner). Empty frames and characters that are part of a sentence do not count. If unsure, false. '
           'Write that hiragana in karuta_char (empty string if none).\n'
           '- has_blank: there is a blank placed instead of a word (white or colored boxes, ○○, □, ＿＿, "?" replacing a word, black bars '
           'replacing a word, or an empty speech balloon attached to a speaker), including blanks inside TV captions or subtitles. Masks '
           'hiding faces or personal information and a question mark at the end of a sentence are not blanks.\n'
           '- has_prompt_text: the poster has overlaid a question or instruction for the respondents (a question, an instruction, a topic '
           'phrase, a situation setting, or a list of names). A sentence that merely contains the blank does not count; true only if there '
           'is a separate question or instruction. TV captions, subtitles, signs, speech and titles that were not overlaid by the poster do '
           'not count. If unsure, false.\n'
           '- has_other_text: there is other readable text relevant to a joke besides the prompt text and the karuta mark (TV captions, '
           'subtitles, signs, speech, etc.; excluding watermarks, logos and unreadable small text).\n'
           '- composite: the image is a composite (several pictures pasted together, or a photo combined with a drawing).\n\n'
           '## 2. Format (choose the first that applies, in this order)\n'
           '1. karuta: a karuta mark as above. 2. anaume: a blank as above. 3. odaibun: overlaid prompt text as above. 4. normal: none of these.\n\n'
           '## 3. Medium\n' + M_V8_EN.split("Answer with JSON only.\n", 1)[1])
COMPONENTS["J_v8_en"] = {"prompt": J_V8_EN, "max_pixels": BASE_PIXELS,
                         "fields": {**ATTR_FIELDS, "format": FMT_ENUM, **_bool_fields(MEDIA)}, "derive_format": True}

# F_v8_en は通常の画像の大半を text とした（dev 形式正解率 0.487）．text の範囲を絞った2版を試す
F_V8B_EN = _V2.replace(
    '- "text": the image consists almost entirely of written text (typically a topic or question on a plain or colored background) and has no main photo, anime or manga picture.',
    '- "text": the poster has added a sentence that asks the respondents a question or gives them a topic or instruction (e.g. '
    '"What is ...?", "Please ...", "... that you often see", "the common point of ..."), typically on a plain background, in a Bokete '
    'template frame, or in a band added above or below a picture. A picture may also be present. Most Bokete images do NOT have such a '
    'sentence: TV captions, subtitles, signs, labels, speech balloons and titles do not count, and an ordinary photo with some words is "normal".',
)
F_V8C_EN = _V2.replace(
    'and has no main photo, anime or manga picture.',
    'and has no main photo, anime or manga picture. It is also "text" when a picture is present but the poster has clearly added a '
    'separate question or topic sentence for the respondents (e.g. in a template frame or a band above or below the picture); TV captions, '
    'subtitles, signs and speech balloons do not count.',
)
COMPONENTS["F_v8b_en"] = {"prompt": F_V8B_EN, "max_pixels": BASE_PIXELS, "fields": COMPONENTS["F_v2"]["fields"]}
COMPONENTS["F_v8c_en"] = {"prompt": F_V8C_EN, "max_pixels": BASE_PIXELS, "fields": COMPONENTS["F_v2"]["fields"]}

# ===== order_011：定義書 v3 の媒体（CG なし）．M_v6 から CG の選択肢を除き，マンガの例示と 3DCG・生成AI の扱いを加えた =====
MEDIA_V3 = ["photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
MEDIA_RULES_V9 = """媒体（原則1つだけ true．合成画像で異なる媒体の領域が画像の約2割以上ずつある場合のみ複数）：
- photo：現実の写真（実写の映像，フィギュアや人形，着ぐるみ，像を撮影したものを含む）
- anime：アニメのスクリーンショット
- game：ゲーム画面のキャプチャ（ゲームの会話画面，戦闘画面，ドット絵のゲーム画面，ゲームのキャラクターの3DCG を含む）
- manga：漫画作品から切り抜いたコマ（枠線，スクリーントーン，写植の台詞などをもつ．漫画風の一枚絵や吹き出し付きのカットは illustration）
- text_graphic：文字や記号が主体で，写真や絵がほとんどない画像（ウェブページやアプリの画面を含む）
- painting：古典的絵画（浮世絵などの版画を含み，写真に撮ったものも絵画とする）
- illustration：上記に含まれない絵（キャラクターの描き下ろし，模写，漫画風の一枚絵，カルタの札の絵，手描きの絵，デジタルの絵を含む）
3DCG や生成AIによる画像は，同じ見た目の画像を人が作った場合の媒体とする（写実的なものは photo，ゲームの画面は game，それ以外は illustration）．"""
COMPONENTS["M_v9"] = {"prompt": HEAD + MEDIA_RULES_V9 + "\nJSON のみを出力してください．", "max_pixels": BASE_PIXELS,
                      "fields": _bool_fields(MEDIA_V3)}

# ===== order_012：お題文の過検出の防止．F_v2 の text に「画像に元からある文字」の除外を加えた2版 =====
_F_V9_EXCLUDE = (' Text that was originally part of the photographed scene or the source material is NOT "text", even if it fills '
                 'the image: signboards, posters, book, record or DVD covers, product packages, TV screens and captions, subtitles, '
                 'song lyrics, diagrams and figure labels, and manga pages. Such images are "normal" (or "anaume" if a word is blanked out).')
F_V9A = _V2.replace('and has no main photo, anime or manga picture.',
                    'and has no main photo, anime or manga picture.' + _F_V9_EXCLUDE)
F_V9B = _V2.replace(
    'and has no main photo, anime or manga picture.',
    'and has no main photo, anime or manga picture. The text must have been written by the poster for the respondents: a question, '
    'an instruction, or a topic phrase (e.g. "What is ...?", "Please ...", "... あるある", "the common point of ...").' + _F_V9_EXCLUDE)
COMPONENTS["F_v9a"] = {"prompt": F_V9A, "max_pixels": BASE_PIXELS, "fields": COMPONENTS["F_v2"]["fields"]}
COMPONENTS["F_v9b"] = {"prompt": F_V9B, "max_pixels": BASE_PIXELS, "fields": COMPONENTS["F_v2"]["fields"]}

# ===== order_012：定義書 v4 の文字あり（ロゴ等を含め，読める文字があれば真）を独立に問う部品 =====
X_V10_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Answer with JSON only.\n'
            '- has_any_text: is there any readable text anywhere in the image (Japanese characters, Latin letters or numbers)? '
            'Count every kind of text: overlaid questions, captions, subtitles, speech balloons, sound effects drawn in manga, signs, '
            'posters, product labels, text on clothing or uniforms, logos, watermarks, copyright notices, URLs, and date stamps. '
            'Do NOT count text that is blurred or mosaicked so that it cannot be read, text too small to read, or a karuta mark alone '
            '(a single hiragana in a circle or square).')
COMPONENTS["X_v10_en"] = {"prompt": X_V10_EN, "max_pixels": BASE_PIXELS, "fields": {"has_any_text": {"type": "boolean"}}}

_parse_v9 = parse


def parse(name: str, rec: dict) -> dict:  # noqa: F811
    """部品の出力を変換する．X_v10 は parse_v8 と同じくそのまま返し，それ以外は従来の変換を用いる．

    Args:
        name (str): 部品の名前．
        rec (dict): 出力の JSON を解釈した辞書．

    Returns:
        dict: 部品の項目の値．
    """
    if name.startswith(("X_v10", "O_v10", "O_v11")):
        return parse_v8(name, rec)
    return _parse_v9(name, rec)

# F_v9a/b は text の説明を長くしたことで，かえって text を選びやすくなった（order_012）．
# そこで (c) 除外の規則を normal の側に書く版と，(O) F_v2 が text とした画像に投稿者の問いかを別に確かめる質問を試す
F_V9C = _V2.replace(
    'Pictures that contain captions, subtitles, signs, titles or speech bubbles with words are still "normal".',
    'Pictures that contain captions, subtitles, signs, titles or speech bubbles with words are still "normal". Images whose text was '
    'originally part of a sign, poster, book or DVD cover, product package, TV or news screen, web page, song lyrics or a diagram '
    'are also "normal", even if the text is large.')
COMPONENTS["F_v9c"] = {"prompt": F_V9C, "max_pixels": BASE_PIXELS, "fields": COMPONENTS["F_v2"]["fields"]}
O_V10_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Answer with JSON only.\n'
            '- is_poster_prompt: does the image contain a question, an instruction, or a topic phrase that the person who posted the image '
            'wrote for the respondents to answer (e.g. "What is ...?", "Please ...", "... あるある", "... な事", "the common point of ...", '
            '"What would happen if ...")? Answer false if all the text was originally part of the source material: signs, posters, book or '
            'DVD covers, product packages, TV or news screens and captions, web articles, song lyrics, diagrams, dialogue in games, anime '
            'or manga, or a single quoted line without a question or topic.')
COMPONENTS["O_v10_en"] = {"prompt": O_V10_EN, "max_pixels": BASE_PIXELS, "fields": {"is_poster_prompt": {"type": "boolean"}}}

# ===== order_013：O_v10_en は実データで「こんな〜は嫌だ」などの典型的なお題文を偽とした．典型的な型を明示した版 =====
O_V11_EN = ('This image is a prompt image ("odai") from the Japanese comedy site Bokete. Answer with JSON only.\n'
            '- is_poster_prompt: does the image contain text that the poster wrote as the theme for the respondents to answer? '
            'This includes a question ("〜とは？", "What is ...?"), an instruction or request ("〜してください", "〜しろ"), '
            'a topic phrase ("こんな〜は嫌だ", "〜あるある", "〜な事", "〜した奴優勝", "〜の共通点", "〜にありがちなこと"), '
            'a situation setting or short scenario, a title-like theme written on a plain background or a Bokete template, '
            'and a list of names or items to be used. Answer false only if all the text clearly comes from the original source '
            'material: signboards, posters, book or DVD covers, product packages, TV or news screens and captions, web articles, '
            'song lyrics, charts or diagrams, or dialogue in games, anime or manga.')
COMPONENTS["O_v11_en"] = {"prompt": O_V11_EN, "max_pixels": BASE_PIXELS, "fields": {"is_poster_prompt": {"type": "boolean"}}}
