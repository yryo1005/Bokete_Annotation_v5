# Order 029

## ユーザーの指示（原文）

私のGeminiのAPIキーを用いてGeminiにも画像をアノテーションさせ、Claude Codeのアノテーションと結果を比較してください

## 指示の解釈

- tokens.json の Gemini の API キーを用い，Gemini（google-genai）にお題画像のラベルを付けさせる．
- 対象は，Claude Code が定義書 v4 で目視のラベルを付けたお題画像の全て（計1,020枚）とする．
  - 評価用データ640枚（ex004／ex007，文字ありは ex008 で v4 に見直し済み）
  - 実データの無作為抽出160枚（ex008 real_random_v4.txt）
  - 実データの層別抽出220枚（ex008 real_strat_v4.txt．report_028 の11層）
- Gemini への指示は定義書 v4（programs/label_guideline/label_definition_v4.md）をそのまま与え，形式・属性・媒体を JSON で答えさせる．
- 比較：軸ごとの一致率・κ係数，クラスごとの F1（Claude Code を基準とした場合），全項目の一致率，層ごとの一致，不一致の例の目視．
  既存の自動の方法（パッケージの最終版のラベル）とも並べ，Claude Code・Gemini・自動の3者の一致を比べる．
- 結果は report_029 と PDF にまとめる．
