#!/bin/bash
# order_016 の本番前の検証を順に実行する（研究用のリポジトリのみ．v5.0 は読み込みのみ）．
# 1. 候補の方法の採点（240枚）
# 2. 本番用スクリプトで無作為の2万枚（途中で中断して再開できるかも確かめる）
# 3. 決定性の確認（同じ2,000枚をもう一度処理し，1回目と比べる）
set -u
PY=.venv_annotate/bin/python
D=programs/ex011_caption_final
O=outputs/ex011_caption_final
$PY $D/judge_eval2.py > $O/judge_eval2.log 2>&1
# 2万枚：最初のシャードが保存された直後に中断し，再開する
$PY $D/caption_all.py --sample 20000 --out_dir $O/test_20k --stages run > $O/test_20k_run1.log 2>&1 &
PID=$!
until ls $O/test_20k/parts/part_0000.parquet >/dev/null 2>&1 || ! kill -0 $PID 2>/dev/null; do sleep 5; done
sleep 2; pkill -f "caption_all.py --sample 20000"; sleep 20
$PY $D/caption_all.py --sample 20000 --out_dir $O/test_20k > $O/test_20k_run2.log 2>&1
# 決定性：2万枚の対象の先頭2,000枚を別の出力先でもう一度処理する
$PY - <<'PYEOF'
import polars as pl, os
os.makedirs("outputs/ex011_caption_final/determinism", exist_ok=True)
pl.read_parquet("outputs/ex011_caption_final/test_20k/targets.parquet").head(2000).write_parquet("outputs/ex011_caption_final/determinism/targets.parquet")
PYEOF
$PY $D/caption_all.py --out_dir $O/determinism > $O/determinism.log 2>&1
echo done > $O/checks_done
