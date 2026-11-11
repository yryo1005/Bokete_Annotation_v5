# Order 031

## ユーザーの指示（原文）

（Google Colab で streamlit を `cloudflared tunnel --url http://localhost:8501` のクイックトンネルで公開し，ログの trycloudflare.com の URL を取り出すコードを示して）
こんな感じでログイン不要でCloudlfareを使いたい

## 指示の解釈

- BoketeImageAnnotator を，Cloudflare のアカウントへのログインやデプロイなしで LINE と接続できるようにする．
- 手元の PC で `wrangler dev --local`（ローカルの D1 と静的アセット）を起動し，`cloudflared tunnel --url` のクイックトンネル（trycloudflare.com，アカウント不要）で HTTPS で公開する．
- クイックトンネルの URL は起動のたびに変わるため，起動のスクリプトで URL をログから取り出し，
  画像の URL の基点（PUBLIC_BASE_URL）に渡し，LINE の Messaging API（Webhook のエンドポイントの設定）で Webhook URL を自動で更新し，接続の試験まで行う．
- cloudflared がなければ，リポジトリ内（Git の管理外）に実行ファイルを取得する．ローカルの D1 が空なら表と問題を投入する（既存の回答は消さない）．
- 制約（PC を起動し続ける必要，クイックトンネルは試験用で稼働の保証がないこと）を README とレポートに記す．
