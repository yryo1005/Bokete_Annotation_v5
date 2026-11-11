# BoketeImageAnnotator のログインなしの運用（cloudflared のクイックトンネル）

## 1. 概要

BoketeImageAnnotator（report_030）を，Cloudflare のアカウントへのログインとデプロイなしで LINE に接続して動かせるようにした．手元の PC で Worker を `wrangler dev --local` で動かし，cloudflared のクイックトンネル（`https://*.trycloudflare.com`．アカウント不要）で HTTPS で公開する．起動は `npm run image:tunnel` の1つのコマンドで行い，トンネルの URL の取得，画像の URL の基点の設定，LINE の Webhook URL の更新と接続の試験までを自動で行う．実際の LINE の公式アカウントとの接続は，チャネルの秘密の値がないため未確認である．

| 項目 | 内容 |
| :--- | :--- |
| 追加 | `scripts/serve_image_quick_tunnel.mjs`，npm のスクリプト `image:tunnel` |
| 修正 | 模擬試験の D1 の保存先の分離，子プロセス（workerd）の停止，集計の `--persist-to` |
| 確認 | トンネル経由のトップ・画像・署名の検証・Webhook の処理，模擬試験（12項目），vitest 10件，pytest 5件 |

## 2. 起動のスクリプト

`scripts/serve_image_quick_tunnel.mjs` は次の順に処理する．

1. cloudflared が PATH になければ，GitHub の公開のリリースから Linux 用の実行ファイルを `.tools/`（Git の管理外）に取得する．
2. ローカルの D1（`BoketeImageAnnotator/.wrangler/state`）の問題の数を調べ，問題がなければ表と問題を投入する．問題があれば投入せず，既存の回答を引き継ぐ．
3. 指定のポート（既定は8787）が使用中なら，空いているポートを探す．
4. `cloudflared tunnel --url http://localhost:{ポート}` を起動し，ログから trycloudflare.com の URL を取り出す．
5. その URL を `PUBLIC_BASE_URL` として `wrangler dev --local` を起動する．Flex Message の画像の URL はこの基点から作られる．
6. ローカルとトンネルの URL の両方で，応答の本文が本 Worker のもの（`BoketeImageAnnotator`）であることを確かめる．
7. `.dev.vars` のチャネルアクセストークンで，LINE の Messaging API の Webhook のエンドポイントの設定（`PUT /v2/bot/channel/webhook/endpoint`）と試験（`POST /v2/bot/channel/webhook/test`）を行う．

子プロセスはプロセスグループを分けて起動し，Ctrl+C の際にプロセスグループごと止める．

## 3. 動作の確認

トンネル経由（外部の URL）で次を確かめた．LINE の秘密の値の代わりに試験用の値を用い，Webhook URL の更新は省略した（`--no-webhook`）．

| 確認 | 結果 |
| :--- | :--- |
| トップ（`/`） | 200，本文 `BoketeImageAnnotator` |
| 画像（`/img/61711.jpg`） | 200，`image/jpeg`，37 KB |
| 不正な署名の Webhook | 401 |
| 正しい署名の友だち追加の Webhook | 200．ローカルの D1 に評価者が登録された |
| LINE への応答 | 本物の API に到達し，試験用のトークンのため 401 で拒否された（想定通り） |
| 停止 | Ctrl+C（SIGINT）で cloudflared，wrangler，workerd の全てが終了した |

確認の過程で2つの不具合が見つかり，修正した．1つ目は，模擬試験（`simulate_image.mjs`）が `npx` のプロセスのみを止めていたため，wrangler dev の子の workerd がポート8787を使い続けていたことである．起動のスクリプトの最初の試行ではこのポートの衝突で wrangler dev が終了し，さらにローカルの応答の確認が残っていた別のサーバーの応答で成功と判定していた．プロセスグループごと止める処理，空いているポートの探索，応答の本文の確認を加えた．2つ目は，模擬試験と運用が同じローカルの D1 を用いていたため，模擬の回答が運用のデータに混ざることである．模擬試験の D1 を `.wrangler/sim-state` に分けて毎回作り直すようにし，運用のローカルの D1 は空に戻した．修正の後，模擬試験の12項目，vitest 10件，pytest 5件が全て合格し，模擬試験の終了後に wrangler と workerd のプロセスが残らないことを確かめた．

## 4. 運用の上の注意

- PC と `npm run image:tunnel` を動かしている間のみ Bot が応答する．止めている間の操作には応答しない．
- クイックトンネルの URL は起動のたびに変わる．Webhook URL はスクリプトが更新するが，それ以前に送った吹き出しの画像は表示されなくなる．
- Cloudflare はクイックトンネルを試験と開発の用途とし，稼働を保証していない．
- LINE Developers Console の「Webhook の利用」はあらかじめオンにする必要がある．

## 5. 未解決の論点と次の作業の候補

- 実際の LINE の公式アカウントでの接続（Webhook URL の自動更新と試験，Flex Message の表示）は未確認である．`.dev.vars` にチャネルの秘密の値を書いて `npm run image:tunnel` を実行し，スマートフォンで友だち追加から数問の回答までを確かめる必要がある．
- 多数の評価者から長期間回答を集める場合は，Cloudflare へのデプロイ（README の手順4）に移る．その際，ローカルの D1 の回答は `wrangler d1 export --local` で書き出して取り込めるが，取り込みの手順は未検証である．
- PC の停止中に評価者が操作した場合の案内（停止中である旨の表示）はできない．運用の時間を評価者に事前に知らせる必要がある．
