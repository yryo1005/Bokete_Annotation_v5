# Order 018

## ユーザーの指示（原文）

キャプションを BoketeDBv5 に加えてください
また，キャプションの付与を前提として@Bokete_Annotation_programs_v5.0/以下のリポジトリを修正してください

## 指示の解釈

- Bokete_Annotation_programs_v5.0/ のパッケージに，写真のキャプションの段階（caption）を加える．方式は report_016・017 の本番と同じ
  （Qwen3-VL-30B-A3B-Instruct-FP8，448画素，短い指示，温度0，再現性の設定，句読点の正規化，品質の目印 caption_ok）とする．
- 処理の流れは list → predict → verify → finalize → caption とし，caption は媒体が写真のみの代表の画像に付けて，
  annotation.parquet に列 caption，caption_ok（写真以外は null）として加える．中断から再開できるようにする．
- 研究用のリポジトリで付けた全写真のキャプション（order_017）は，同じ区切り（5万枚）・同じ順の部分としてパッケージの出力に移し，
  再計算せずに annotation.parquet に加える．パッケージの実装で少量を付け直し，order_017 の結果と一致することを確かめる．
- README.md，docs（方法と精度，データの形式），config.json，テスト，バージョンを更新する．
