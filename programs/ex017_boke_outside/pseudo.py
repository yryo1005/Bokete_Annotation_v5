"""目視のラベルで微調整した分類器（ex016 の二段階の微調整）で，教師のラベルの学習データ（34,634件）の画像外の知識を付け直す（自己学習の疑似ラベル）．

32B 教師の画像外の知識は目視のラベルとの一致が F1 0.59 にとどまるのに対し，二段階の微調整をした Qwen3-VL-4B は検証データで 0.66〜0.69 である．
この分類器の確率を新しい教師のラベルとし，train.py の --teacher pseudo_{モデル} で学習に用いる．

出力：outputs/ex017_boke_outside/teacher/pseudo_{モデル}/train.csv（列 row, outside, p_outside）

実行例:
    .venv_annotate/bin/python programs/ex017_boke_outside/pseudo.py --model q4 --weights outputs/ex016_boke_small_vlm/ft/q4/.../best_model.pt
"""
import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
from data import OUTPUT_DIR, load_train  # noqa: E402


def main() -> None:
    """疑似ラベルを付けて保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    out = os.path.join(OUTPUT_DIR, "teacher", f"pseudo_{args.model}")
    os.makedirs(out, exist_ok=True)
    df = load_train().select("odaiId", "text", "format", "voice", pl.col("outside").cast(pl.Float64))
    model = T.M16.load_model(args.model, args.weights, seed=0, gradient_checkpointing=False).cuda()
    _, po, _ = T.T16.predict(model, T.make_loader(df, args.model, 64, False, 0))
    pl.DataFrame({"row": list(range(df.height)), "outside": po > 0.5, "p_outside": po}).write_csv(os.path.join(out, "train.csv"))
    print("pseudo", args.model, "rate", float((po > 0.5).mean()), flush=True)


if __name__ == "__main__":
    main()
