#!/bin/bash
# ex016 の微調整の実行（order_023）．
# 1. 教師のラベルで2エポック学習済みの q4・iv4 を，目視のラベルで微調整する．
# 2. 小さなモデルを教師のラベルで1エポック学習し，目視のラベルで微調整する．
# 3. 対照：目視のラベルのみで学習する（q4, q2）．
cd "$(dirname "$0")/../.."
OUT=outputs/ex016_boke_small_vlm/ft
PY=.venv_annotate/bin/python
for m in q4 iv4; do
  $PY -c "
import sys, os; sys.path.insert(0, 'programs/ex016_boke_small_vlm'); import train as T
d1 = os.path.join(T.OUTPUT_DIR, 'ft', '$m', 'teacher_lr0.0001_ep2_bs16_r16', '0')
d2 = d1.replace('_r16' + os.sep, '_r16__stage2' + os.sep)
os.path.exists(os.path.join(d2, 'final_model.pt')) or T.train_stage2(d2, '$m', d1, 0)
" >> $OUT/stage2_q4iv4.log 2>&1
  echo "stage2 $m exit $?"
done
$PY programs/ex016_boke_small_vlm/train.py --models q2 iv2 lfm16 iv1 --stage2 > $OUT/train_small.log 2>&1
echo "train small exit $?"
$PY programs/ex016_boke_small_vlm/train.py --models q4 q2 --human_only > $OUT/human_only.log 2>&1
echo "human_only exit $?"
