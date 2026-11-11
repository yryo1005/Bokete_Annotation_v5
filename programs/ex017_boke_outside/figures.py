"""ex017 の図を作る（evaluate.py の summary.csv と fast/bench_*.json から）．

- fig_outside.png：主な方法の画像外の知識の F1（調整用・最終評価用．最終評価用は95%信頼区間）
- fig_speed.png：通常の推論とプレフィックスのキャッシュによる推論の件数/秒（前処理込み）

出力：outputs/ex017_boke_outside/figures/

実行例:
    .venv_annotate/bin/python programs/ex017_boke_outside/figures.py
"""
import json
import os
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUTPUT_DIR  # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
plt.rcParams["font.family"] = ["IPAexGothic", "sans-serif"]
# 表示する方法 → 表示名
SHOW = {"teacher/kt": "32B 教師（kt）", "teacher/src_fs": "32B 教師（元ネタ＋例示）", "ex016/iv2_stage2": "前回 InternVL3.5-2B",
        "ex016/q4_stage2": "前回 Qwen3-VL-4B", "iv2/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2": "InternVL3.5-2B 重み3",
        "iv2/pseudo_q4_wo3_lr0.0001_ep1_bs16_r16__stage2": "InternVL3.5-2B 疑似ラベル",
        "iv8/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2": "InternVL3.5-8B", "q8/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2": "Qwen3-VL-8B",
        "ens:ex016/q4_stage2+q8/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2": "Qwen3-VL-4B＋8B",
        "ens:ex016/q4_stage2+q8/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2+iv8/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2": "Qwen3-VL-4B＋8B＋InternVL3.5-8B"}
HUMAN = 0.849


def main() -> None:
    """図を作る．

    Returns:
        None
    """
    df = pl.read_csv(os.path.join(FIG, "summary.csv"))
    d = {(r["method"], r["split"]): r for r in df.iter_rows(named=True)}
    keys = [k for k in SHOW if (k, "dev") in d]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(keys))
    dev = [d[(k, "dev")]["f1"] for k in keys]
    test = [d[(k, "test")]["f1"] for k in keys]
    err = np.array([[d[(k, "test")]["f1"] - d[(k, "test")]["f1_lo"], d[(k, "test")]["f1_hi"] - d[(k, "test")]["f1"]] for k in keys]).T
    ax.barh(y - 0.2, dev, 0.38, color="#2a78d6", label="調整用（308件）")
    ax.barh(y + 0.2, test, 0.38, color="#eb6834", label="最終評価用（312件，95%信頼区間）", xerr=err, ecolor="#555555", capsize=2)
    for yi, a, b in zip(y, dev, test):
        ax.text(a + 0.005, yi - 0.2, f"{a:.2f}", va="center", fontsize=7, color="#333333")
        ax.text(b + 0.06, yi + 0.2, f"{b:.2f}", va="center", fontsize=7, color="#333333")
    ax.axvline(HUMAN, color="#7a7a7a", ls="--", lw=1, label="目視のラベルの再現性（0.85）")
    ax.set_yticks(y, [SHOW[k] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("画像外の知識の F1（合意ラベル）")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_outside.png"), dpi=150)
    plt.close(fig)
    rows = []
    for tag, name in [("iv2", "InternVL3.5-2B"), ("q4", "Qwen3-VL-4B"), ("q8", "Qwen3-VL-8B")]:
        b = json.load(open(os.path.join(OUTPUT_DIR, "fast", f"bench_{tag}.json")))
        fast = max(v for k, v in b.items() if k.startswith("fast_") and k.endswith("_per_sec"))
        rows.append((name, b["standard_bs32_per_sec"], fast))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r[1] for r in rows], 0.38, color="#7a7a7a", label="通常（バッチ32）")
    ax.barh(y + 0.2, [r[2] for r in rows], 0.38, color="#1baf7a", label="プレフィックスのキャッシュ")
    for yi, r in zip(y, rows):
        ax.text(r[1] + 2, yi - 0.2, f"{r[1]:.0f}", va="center", fontsize=8)
        ax.text(r[2] + 2, yi + 0.2, f"{r[2]:.0f}（{r[2] / r[1]:.1f}倍，全件 {92_500_000 / r[2] / 86400:.1f}日）", va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 300)
    ax.set_xlabel("件/秒（画像の読み込み・前処理を含む，実データ8,664件）")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_speed.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
