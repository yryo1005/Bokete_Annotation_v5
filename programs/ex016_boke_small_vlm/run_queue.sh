#!/bin/bash
# ex016 の GPU の処理を順に実行する（ゼロショットの残り → 推論モデルの教師 → 小さな VLM の微調整）．
# 実行例: bash programs/ex016_boke_small_vlm/run_queue.sh <先行するゼロショットのログ>
cd "$(dirname "$0")/../.."
export VLLM_LOGGING_LEVEL=WARNING
OUT=outputs/ex016_boke_small_vlm
# 先行するゼロショットの実行（最後のモデル mcpm4 の終了）を待つ
until grep -q "mcpm4 exit" "$1" 2>/dev/null; do sleep 30; done
for m in lfm16 gemma4 heron2 q32t q30t; do
  .venv_annotate/bin/python programs/ex016_boke_small_vlm/zeroshot.py --models $m > $OUT/zeroshot/$m.log 2>&1
  echo "zeroshot $m exit $?"
done
mkdir -p $OUT/ft
.venv_annotate/bin/python programs/ex016_boke_small_vlm/train.py --models q4 iv4 gemma4 q2 iv2 lfm3 iv1 lfm16 smol2 > $OUT/ft/train_none.log 2>&1
echo "train none exit $?"
