#!/bin/bash
# order_016 の追加の検証：(1) 同じ入力（同じバッチの構成）での再現性，(2) 複数の部分にまたがる中断と再開
set -u
PY=.venv_annotate/bin/python
D=programs/ex011_caption_final
O=outputs/ex011_caption_final
# (1) determinism と同じ2,000枚を，同じ設定でもう一度処理する
mkdir -p $O/determinism2 && cp $O/determinism/targets.parquet $O/determinism2/
$PY $D/caption_all.py --out_dir $O/determinism2 > $O/determinism2.log 2>&1
# (2) 6,000枚を2,000枚ずつの部分で処理し，最初の部分の保存後に中断して再開する
mkdir -p $O/resume && $PY - <<'PYEOF'
import polars as pl
pl.read_parquet("outputs/ex011_caption_final/test_20k/targets.parquet").slice(2000, 6000).write_parquet("outputs/ex011_caption_final/resume/targets.parquet")
PYEOF
$PY $D/caption_all.py --out_dir $O/resume --shard 2000 --stages run > $O/resume_run1.log 2>&1 &
PID=$!
until ls $O/resume/parts/part_0000.parquet >/dev/null 2>&1 || ! kill -0 $PID 2>/dev/null; do sleep 2; done
ls $O/resume/parts > $O/resume_parts_at_kill.txt
pkill -f "out_dir outputs/ex011_caption_final/resume"; sleep 30
$PY $D/caption_all.py --out_dir $O/resume --shard 2000 > $O/resume_run2.log 2>&1
$PY $D/caption_all.py --out_dir $O/test_20k --stages finalize > $O/test_20k_finalize.log 2>&1
echo done > $O/checks2_done
