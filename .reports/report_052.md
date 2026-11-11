# report_052：最終版のアノテーションの独立したディレクトリへの書き出し

## 概要

最終版のアノテーションを，新しいプロジェクトで単独で使えるディレクトリ `/mnt/d/BoketeDBv5_annotation_final/` に書き出した．対象はパッケージ v5.3.0（ボケは v5.2.0）である．書き出したものは，データ，ラベルの基準（定義書），判定の方法と指示（プロンプト），精度の表，評価に用いた目視のラベル，アノテーションのパッケージである．このディレクトリは研究のリポジトリの外にあり，ここだけを複写すれば使える．書き出しは `programs/ex031_export_final/export.py` で行い，同じコマンドで再開・再実行できる．

## 方法

`export.py` は次の4つを行う．

- 複写：パッケージの出力（`Bokete_Annotation_programs_v5.0/outputs/BoketeDBv5_annotation/`）から，お題画像のラベル，画像の一覧，キャプション，確率，埋め込み，お題文ありの確率，ボケのラベル，集計の json を複写する．定義書，方法の文書，最終版のまとめ（report_051），部分集合の精度（report_050）と，パッケージのコードとモデルの重みも複写する（outputs，.git，キャッシュを除く）．大きさと更新時刻が同じファイルは飛ばす．
- 部分集合の一覧：写真のみ，写真・文字なし，写真・通常のお題の一覧を書き出す（ex029 と同じ条件）．列は `odaiId`，`rep_odaiId`，ボケの件数 `n_boke`，画像外の知識が要るボケの件数 `n_outside` である．
- 目視のラベル：各実験のデータの読み込みの関数で読み直し，CSV にする．対象は，お題画像1,020枚（第4版），文の属性（第6版の草案）1,620枚・636枚・600枚，ボケの検証データ620件（合意ラベル）である．キャプションの参照文240枚は，元のファイルを複写する．
- 指示の本文：教師と分類器の指示を，各実験のコードの定数から書き出す．画像の教師の6つの部品，お題文ありの P3，キャプション，ボケの教師 one_kt_cap（差し込む値は {} で示す），ボケの分類器の入力の文の10ファイルである．

書き出し先の README.md は `programs/ex031_export_final/README_final.md` を複写したものである．構成，列の説明，使い方の例，判定の方法，精度の要約，評価用ラベルの説明，注意を記した．

## 結果

表1：書き出したもの．

| 場所 | 内容 | 件数・大きさ |
|---|---|---|
| `data/odai/` | annotation.parquet（6,823,270行），images.parquet，captions.parquet，predict/，embeddings/（SigLIP2，1,152次元），odai_text/，summary.json | 約16GB（埋め込み約15GB） |
| `data/boke/` | boke_annotation/（814ファイル，約9,250万件），boke_summary.json | 約2.7GB |
| `data/subsets/` | photo_only（3,474,756お題），photo_notext（2,009,955），photo_normal（3,063,330） | — |
| `docs/` | 定義書（第4版，第5版，第6版の草案，ボケ第3版），method.md，data_format.md，final_summary（report_051），photo_subset（report_050），prompts/（10ファイル） | — |
| `eval/` | 目視のラベル5ファイル（1,020，1,620，636，600，620行），キャプションの参照文2ファイル（各120枚），metrics/（精度の表7ファイル） | — |
| `annotator/` | パッケージのコード，設定，文書，テスト，モデルの重み | 約537MB |

複写の結果は `outputs/ex031_export_final.log` と書き出し先の `export_log.json` に記録した．埋め込み（34ファイル）とボケのラベル（814ファイル）は，複写元と大きさが全て一致した．書き出し先で README.md の使い方の例を実行し，読み込めることを確かめた（写真・文字なしのお題で，画像外の知識が要らないと判定されたボケは14,741,721件）．全体の大きさは約19GBである．

## 使い方

新しいプロジェクトでは，`/mnt/d/BoketeDBv5_annotation_final/` を複写し，`README.md` と `docs/final_summary.md` を最初に読む．面白さの判定に用いる写真・文字なしのお題は，`data/subsets/photo_notext.parquet` の `odaiId` で，ボケのラベル（`data/boke/boke_annotation/`）と結合して得られる．

## ファイル

- プログラム：`programs/ex031_export_final/export.py`，`README_final.md`
- 書き出し先：`/mnt/d/BoketeDBv5_annotation_final/`（リポジトリの外）
- 記録：`outputs/ex031_export_final.log`，`/mnt/d/BoketeDBv5_annotation_final/export_log.json`

## 未解決の論点と次の候補

- 書き出し先は Git で管理していない．別の場所に保存する場合は，ディレクトリごと複写する（約19GB）．
- 元のお題画像のファイルと BoketeDBv4 の表（ボケの文など）は含めていない．ボケの文は BoketeDBv4 の `boke` の表から `id` で結合する．
- 次の作業は，写真・文字なしのボケでの面白さの判定の課題の設計である（評価の分布，同じお題の中の相対的な面白さ，お題単位の分割）．
