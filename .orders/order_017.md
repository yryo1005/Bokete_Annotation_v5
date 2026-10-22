# Order 017

## ユーザーの指示（原文）

この方針で実行してください

## 指示の解釈

- report_016 の本番の設定（Qwen3-VL-30B-A3B-Instruct-FP8，448画素，短い指示，温度0，再現性の設定あり）で，
  写真のお題画像（約337万枚，代表の画像）にキャプションを付ける．
- 実行は研究用のリポジトリの programs/ex011_caption_final/caption_all.py で行い，出力は outputs/ex011_caption_final/photo_captions/ に保存する．
  Bokete_Annotation_programs_v5.0/ は編集しない（読み込みのみ）．
- 終了後，summary.json の異常の件数の確認，無作為の画像の目視の確認，お題ごと（sha256 が同じお題を含む）への展開を行い，report_017 にまとめる．
