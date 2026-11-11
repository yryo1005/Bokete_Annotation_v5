# Order 027

## ユーザーの指示（原文）

この方針で実行して良いです
全件アノテーションしてください

## 指示の解釈

- report_026 の方針（Bokete_Annotation_programs_v5.0 v5.2.0 のボケの段階，判定器は Qwen3-VL-4B 単独，設定は config.json の既定）で，
  BoketeDBv4 の全てのボケ（92,495,946件）にアノテーションを行う．
- 出力先は Bokete_Annotation_programs_v5.0/outputs/BoketeDBv5_annotation/（お題画像の annotation.parquet と同じ場所）．
- 約3.2日の連続実行になるため，セッションから切り離して実行し，進捗（シャードの数，速度）を確かめながら待つ．中断した場合は同じコマンドで再開する．
- 完了後に，件数の確認（全件が出力に含まれるか），ラベルの分布（形式・投稿年ごと），無作為の例の目視の確認，処理時間を集計し，
  report_027 と document.md を更新する．
