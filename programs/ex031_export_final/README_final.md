# BoketeDBv5 アノテーション最終版

大喜利投稿サイト「ボケて」のデータベース BoketeDBv4 の全てのお題画像（6,823,270件）とボケ（92,495,946件）に付けたアノテーションの最終版である．ラベルの基準（定義書），判定の方法と指示（プロンプト），精度，評価に用いた目視のラベルをあわせて収める．このディレクトリだけで，新しいプロジェクトのデータとして使える．

| 項目 | 版 |
|---|---|
| お題画像のラベル（形式・属性・媒体・お題文あり）とキャプション | パッケージ v5.3.0（ラベルの定義は第5版） |
| ボケのラベル（発話の型・画像外の知識・制約の順守） | パッケージ v5.2.0（ボケの定義は第3版） |
| 確定日 | 2026-11-10 |

目視のラベル（正解）は全て Claude Code が付けたものであり，人による評価は行っていない．

## 1. ディレクトリの構成

```text
BoketeDBv5_annotation_final/
├── README.md                     # このファイル
├── data/
│   ├── odai/
│   │   ├── annotation.parquet    # お題画像のラベルとキャプション（1行が1お題．6,823,270行）
│   │   ├── images.parquet        # お題画像の一覧（odaiId，path，sha256 など）
│   │   ├── captions.parquet      # キャプション（代表の画像ごと）
│   │   ├── predict/part_*.parquet  # 代表の画像の確率（embeddings と行の順が一致する）
│   │   ├── embeddings/part_*.npy   # 代表の画像の SigLIP2-so400m の埋め込み（float16，L2 正規化，1,152次元）
│   │   ├── odai_text/part_*.parquet # お題文ありの確率
│   │   └── summary.json          # ラベルの割合と処理の記録
│   ├── boke/
│   │   ├── boke_annotation/part_*.parquet  # ボケのラベル（1行が1件のボケ．814ファイル）
│   │   └── boke_summary.json
│   └── subsets/                  # 絞り込みの条件ごとのお題の一覧（列 odaiId, rep_odaiId, n_boke, n_outside）
│       ├── photo_only.parquet    # 媒体が写真のみ（3,474,756お題）
│       ├── photo_notext.parquet  # 写真のみ・文字なし（2,009,955お題）
│       └── photo_normal.parquet  # 写真のみ・形式が通常（3,063,330お題）
├── docs/
│   ├── final_summary.md / .pdf   # 最終版の構成・指示・精度のまとめ（まずこれを読む）
│   ├── label_definition_v5.md    # お題画像のラベルの定義（第5版．このデータの基準）
│   ├── label_definition_v6_draft.md # 第6版の草案（文字の層と文の形，お題文ありの規則 R1b．評価用ラベルの基準）
│   ├── label_definition_v4.md    # 第4版（お題画像1,020枚の目視ラベルの基準）
│   ├── boke_label_definition_v3.md # ボケのラベルの定義（第3版）
│   ├── method.md                 # 方法と精度（パッケージの文書）
│   ├── data_format.md            # 出力の列と途中の出力の形式
│   ├── photo_subset.md / .pdf    # 写真・文字なしと写真・通常の件数と判定精度
│   └── prompts/                  # 教師と分類器の指示の本文
├── eval/
│   ├── image_labels_v4_1020.csv  # お題画像の目視ラベル（形式・属性・媒体．第4版）
│   ├── text_attr_v6_1620.csv     # 文の属性の目視ラベル（第6版の草案）1,620枚
│   ├── text_attr_v6_holdout_636.csv  # 同 独立な評価用データ600枚＋判断のずれの確認36枚
│   ├── text_attr_v6_strata_DF_600.csv # 同 層 D・F の追加の抽出600枚
│   ├── boke_labels_620.csv       # ボケの検証データ620件の合意ラベル
│   ├── caption_reference_120a.txt / 120b.txt # 写真のキャプションの参照文（計240枚）
│   └── metrics/                  # 最終版の精度の集計の表
├── annotator/                    # アノテーションのパッケージ（コード，設定，モデルの重み）
└── export_log.json               # 書き出しの記録
```

`final_summary.md` と `photo_subset.md` の中のファイルのパス（`programs/...`，`outputs/...`）は，元の研究のリポジトリ（Bokete_Annotation_v5）のものである．

## 2. データ

### 2.1 お題画像（data/odai/annotation.parquet）

主な列は次のとおりである．全ての列は `docs/data_format.md` にある．

| 列 | 内容 |
|---|---|
| `odaiId`，`rep_odaiId`，`sha256` | お題ID，内容が同じ画像の代表のお題ID，画像の sha256 |
| `format` | 形式（karuta，anaume，odaibun，normal）．属性から導く |
| `has_blank`，`has_karuta_mark`，`has_text`，`has_odai_text` | 属性：空欄あり，カルタ記号あり，文字あり（読める文字がある），お題文あり |
| `photo`，`anime`，`game`，`manga`，`text_graphic`，`painting`，`illustration` | 媒体（原則1つ．合成画像は複数） |
| `p_*` | 各項目の確率（校正していない） |
| `caption`，`caption_ok` | 写真のみの画像のキャプション（日本語の1文）と，壊れた出力でないか |

内容が同じ画像（sha256 が同じ）には同じラベルが付く．埋め込みは代表の画像ごとにあり，`predict/part_XXXX.parquet` の `rep_odaiId` の行の順と，同じ番号の `embeddings/part_XXXX.npy` の行の順が一致する．

### 2.2 ボケ（data/boke/boke_annotation/part_*.parquet）

| 列 | 内容 |
|---|---|
| `id`，`odaiId` | ボケID，お題ID |
| `format` | ボケの段階を実行した時点の形式（第4版．v5.3.0 の形式とはお題の約1.5%で異なる．最新の形式は annotation.parquet を結合して使う） |
| `predicted` | 発話の型と画像外の知識を予測したか（お題画像がない3,133件は偽） |
| `voice6`，`voice3` | 発話の型．6区分（S セリフ，M 心の声，T ツッコミ・呼びかけ，N 状況の説明・物語，L 名付け・見立て，A 回答）と3区分（character，outside，answer） |
| `outside`，`p_outside` | 画像外の知識が要るか（確率0.5以上）とその確率 |
| `p_voice_*` | 発話の型の確率 |
| `uses_image_text` | 画像中の文字への依存（形式が穴埋め・お題文なら真） |
| `karuta_card`，`karuta_complies`，`compliance_score` | カルタの札のかなと順守，穴埋め・お題文の順守の得点（同じお題の他のボケとの類似度） |
| `empty_text` | ボケの文が空か |

### 2.3 使い方の例

```python
import polars as pl

ann = pl.read_parquet("data/odai/annotation.parquet")
boke = pl.scan_parquet("data/boke/boke_annotation/*.parquet")

# 写真・文字なしのお題のボケ（画像外の知識が要らないもの）
ids = pl.read_parquet("data/subsets/photo_notext.parquet")["odaiId"]
plain = boke.filter(pl.col("odaiId").is_in(ids.implode()) & pl.col("predicted") & ~pl.col("outside")).collect()

# 最新の形式をボケに付ける
boke_fmt = boke.drop("format").join(ann.lazy().select("odaiId", "format"), on="odaiId", how="left")
```

## 3. ラベルの基準と判定の方法

詳しくは `docs/final_summary.md`（構成・指示・精度）と，各定義書を参照すること．要点は次のとおりである．

| ラベル | 判定の方法 | 教師（指示は docs/prompts/） |
|---|---|---|
| 形式・属性・媒体 | SigLIP2-so400m の埋め込みに対する MLP（3 seed）と，微調整した SigLIP2-B の確率の平均．確率0.5以上を真とし，形式は属性から導く | Qwen3-VL-8B への6つの質問（image_teacher_*.txt）を規則で組み合わせた約4.5万枚のラベル |
| お題文あり | 埋め込みと形式・属性の確率を入力とする MLP（3 seed）．形式がお題文なら確率0.05以上，穴埋め・カルタなら0.3以上でお題文あり，通常ならお題文なし | Qwen3-VL-32B の指示 P3（odai_text_teacher_P3.txt） |
| キャプション | 写真のみの画像に Qwen3-VL-30B-A3B（温度0，再現性の設定あり）で生成 | caption.txt |
| 発話の型・画像外の知識 | Qwen3-VL-4B＋LoRA の分類器（お題画像とボケの文を入力．boke_classifier_input.txt） | Qwen3-VL-32B の指示 one_kt_cap（boke_teacher_one_kt_cap.txt）のラベル34,634件で学習し，目視931件で微調整 |
| 制約の順守 | カルタは書き出しの音の規則．穴埋め・お題文は ruri-v3-310m の埋め込みによる類似度 | — |

## 4. 精度

正解は Claude Code の目視のラベルである．実データの推定は，無作為抽出と層別抽出の重み付きの推定である．表の値は `eval/metrics/` の表から引いた．

| ラベル | 適合率 | 再現率 | F1 | 評価用データ |
|---|---:|---:|---:|---|
| 写真 | 0.98 | 0.98 | 0.98 | 実データの推定（380枚） |
| 文字あり | 0.94 | 0.99 | 0.97 | 同上 |
| 形式：通常 | 0.97 | 0.96 | 0.97 | 同上 |
| 形式：穴埋め | 0.93 | 0.97 | 0.95 | 同上 |
| 形式：カルタ | 1.00 | 1.00 | 1.00 | 同上 |
| アニメ | 0.81 | 0.96 | 0.88 | 同上 |
| イラスト | 0.78 | 0.87 | 0.82 | 同上 |
| ゲーム | 0.89 | 0.79 | 0.84 | 少ない区分の評価用データ（800枚） |
| 絵画 | 0.78 | 0.80 | 0.79 | 同上 |
| お題文あり（第6版の R1b） | 0.83 | 0.75 | 0.79 | 実データの推定（1,200枚） |
| お題文あり（第5版） | 0.74 | 0.95 | 0.83 | 層別抽出420枚 |
| 写真・文字なし | 0.96 | 0.98 | 0.97 | 実データの推定（380枚） |
| 写真・通常 | 0.98 | 0.99 | 0.98 | 同上 |
| 発話の型 3区分（マクロ） | 0.88 | 0.86 | 0.86 | ボケの最終評価用312件 |
| 発話の型 6区分（マクロ） | 0.74 | 0.71 | 0.71 | 同上 |
| 画像外の知識 | 0.67 | 0.71 | 0.69 | 同上 |

その他：キャプションは写真240枚で概要として十分（5段階で4以上）が87%．穴埋め・お題文の順守の得点は AUC 0.81，カルタの順守の規則は正解率0.97．ボケの目視ラベルの再現性（同じ判断者の付け直しとの一致）は，6区分 F1 0.80，3区分 0.93，画像外の知識 0.85．

## 5. 評価用の目視ラベル（eval/）

| ファイル | 内容 | 主な列 |
|---|---|---|
| `image_labels_v4_1020.csv` | お題画像1,020枚の形式・属性・媒体（第4版） | `set`（eval：評価用640枚，random：無作為抽出160枚，strat：層別抽出220枚），`stratum`，`w`（実データの推定の重み．eval は1） |
| `text_attr_v6_1620.csv` | 文の属性（第6版の草案）1,620枚 | `code`（層と文の形の記号．定義書 v6 の付録），`L_*`（層），`F_*`（重ねた文字・余白の文字の文の形），`odai_R*`（規則で導いたお題文あり．既定は R1b），`label_v5`（第5版のお題文あり），`source`，`w`，`stratum2`（ex025 の600枚のみ重みあり） |
| `text_attr_v6_holdout_636.csv` | 同 独立な評価用データ600枚（ex025 と同じ9層）と，1,620枚から再び付けた36枚 | `kind`（new / repeat），`stratum2`，`N_h`，`n_h`，`w` |
| `text_attr_v6_strata_DF_600.csv` | 同 層 D・F の追加の抽出各300枚 | 同上 |
| `boke_labels_620.csv` | ボケの検証データ620件の合意ラベル | `split`（dev 308件，test 312件），`voice`（6区分），`voice3`（C 登場人物の言葉，X 外からの言葉，A 回答），`outside`，`comp`（順守），`uses_text` |
| `caption_reference_120a.txt`，`120b.txt` | 写真のキャプションの参照文 | お題ID とキャプション（タブ区切り） |

層の定義（stratum2）は，パッケージ v5.2.0 の形式・確認の質問の回答・お題文ありの確率による．A_odaibun_yes，B_odaibun_rejected，C_blank_p_hi，D_blank_p_lo，F_normal_p_lo と，通常と予測した画像の p の区間 N0〜N3 の9層である（`docs/final_summary.md`，`docs/photo_subset.md`）．

## 6. 注意

- 確率は校正していない．目的に応じてしきい値を変える場合は確率の列を使う．
- `has_text` は読める文字があるかを表す．透かしや小さな看板も文字ありになる．
- `has_odai_text` は第5版の定義で学習している．元の素材の字幕に重ねた問いなど，第6版でお題文ありとする画像の多くを見逃す（層 D・F で約1.5〜2%．docs/final_summary.md）．
- 画像外の知識は，画像に写っている人物（有名人を含む）やキャラクターの知識を「画像外」に含めない．画像と一般知識だけで分かるボケに絞る用途では，写真の有名人に頼るボケが残る．
- 発話の型の6区分は人の判断でも揺れが大きい．集計や条件付けには3区分を使う．
- ボケのファイルの `format` は第4版の形式である．最新の形式は annotation.parquet を結合して使う．
- 回答（A）の再現率は0.83である．入力の形式が通常のボケは回答と判定されない．

## 7. 再生成と出典

- `annotator/` はアノテーションのパッケージである．使い方は `annotator/README.md` を参照すること．BoketeDBv4 の表と画像のファイルが必要である．
- このディレクトリは，研究のリポジトリ（Bokete_Annotation_v5）の `programs/ex031_export_final/export.py` で書き出した．方法の検討の記録は，同リポジトリの `.reports/report_012.md`〜`report_051.md` にある．
