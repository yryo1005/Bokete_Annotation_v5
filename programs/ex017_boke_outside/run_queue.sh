#!/bin/bash
# ex017 の GPU の実行の順序（order_024）．
cd "$(dirname "$0")/../.."
OUT=outputs/ex017_boke_outside
PY=.venv_annotate/bin/python
mkdir -p $OUT/ft
$PY programs/ex017_boke_outside/pseudo.py --model q4 --weights outputs/ex016_boke_small_vlm/ft/q4/teacher_lr0.0001_ep2_bs16_r16__stage2/0/best_model.pt > $OUT/pseudo_q4.log 2>&1
echo "pseudo exit $?"
$PY programs/ex017_boke_outside/train.py --models iv2 --teacher ex015 --w_outside 3 > $OUT/ft/iv2_ex015_wo3.log 2>&1
echo "iv2 ex015 exit $?"
$PY programs/ex017_boke_outside/train.py --models iv2 --teacher pseudo_q4 --w_outside 3 > $OUT/ft/iv2_pseudo_wo3.log 2>&1
echo "iv2 pseudo exit $?"
$PY programs/ex017_boke_outside/train.py --models iv8 --teacher ex015 --w_outside 3 > $OUT/ft/iv8_ex015_wo3.log 2>&1
echo "iv8 ex015 exit $?"
$PY programs/ex017_boke_outside/train.py --models q8 --teacher ex015 --w_outside 3 > $OUT/ft/q8_ex015_wo3.log 2>&1
echo "q8 ex015 exit $?"
