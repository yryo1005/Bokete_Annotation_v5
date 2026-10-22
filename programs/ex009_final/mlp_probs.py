"""埋め込みの MLP（ex008 の最良の条件）を同じ設定で学習し直し，全ての評価用の画像の確率を保存する．

ex008 の students.py は評価用データ640枚の確率を保存していないため，後処理と組み合わせの比較に用いる確率を得る目的で，
ex008 で選ばれたハイパーパラメータ（metrics.json の selected_hp）と同じ seed で学習し直す．
予測が ex008 の predictions.csv と一致するかも確認する．全データのアノテーションに用いるため，検証用で最良の
エポックの重みも保存する（ex008 の fit_mlp は重みを返さないため，同じ手順の fit_mlp_with_state で学習する）．

出力：outputs/ex009_final/mlp/{教師}__base+rare/{seed}/probs_{eval,rare,check}.npz，best_model.pth，meta.json

実行例:
    .venv_annotate/bin/python programs/ex009_final/mlp_probs.py
    .venv_annotate/bin/python programs/ex009_final/mlp_probs.py --teacher T_v11
"""
import copy
import json
import os
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR  # noqa: E402
from data import EX8_DIR, PROJECT_DIR, TEACHER, _load  # noqa: E402

ST = _load("ex008_students_for_final", os.path.join(EX8_DIR, "students.py"))
# ハイパーパラメータは ex008 で T_v10oJXk の教師ラベルについて選んだものを，他の教師でも用いる
EX8_RUN = os.path.join(PROJECT_DIR, "outputs", "ex008_label_v4", "student_mlp", "T_v10oJXk__base+rare__unweighted")


def fit_mlp_with_state(hp: dict, X_tr: np.ndarray, y_tr: dict, X_va: np.ndarray, y_va: dict, targets: dict, seed: int,
                       epochs: int = 40, batch_size: int = 512) -> tuple:
    """ex008 の fit_mlp と同じ手順（重みなし）で MLP を学習し，最良のエポックの確率と重みを返す．

    Args:
        hp (dict): lr，wd を持つハイパーパラメータ．
        X_tr (np.ndarray): 学習用の特徴量（形状 (N, 1152)）．
        y_tr (dict): 学習用のラベル．
        X_va (np.ndarray): 検証用の特徴量．
        y_va (dict): 検証用のラベル．
        targets (dict[str, np.ndarray]): 予測する集合の名前と特徴量（"tr"，"va" を含む）．
        seed (int): 乱数のシード．
        epochs (int): エポック数．
        batch_size (int): バッチサイズ．

    Returns:
        tuple[dict, dict]: 集合の名前をキーとする確率の辞書と，最良のエポックの重み（state_dict）．
    """
    ST.set_seed(seed)
    dev = "cuda"
    T = {k: torch.tensor(v, device=dev) for k, v in targets.items()}
    Y = {"format": torch.tensor(y_tr["format"], device=dev), **{h: torch.tensor(y_tr[h], device=dev) for h in ST.HEADS}}
    model = ST.ThreeHeadMLP(X_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    g = torch.Generator().manual_seed(seed)
    best, best_out, best_state = -1.0, None, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), generator=g).to(dev)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            loss = ST.three_head_loss(model(T["tr"][b]), {k: v[b] for k, v in Y.items()}, None)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = {k: ST.probs_of(model(v)) for k, v in T.items()}
        f1 = ST.score(probs["va"], y_va)["overall_macro_f1"]
        if f1 > best:
            best, best_out, best_state = f1, probs, copy.deepcopy(model.state_dict())
    return best_out, best_state


def main() -> None:
    """3つの seed で学習し直して確率を保存する．

    Returns:
        None
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=TEACHER)
    teacher = parser.parse_args().teacher
    X_tr, y_tr, X_va, y_va, X_ev = ST.load_xy(teacher, True)
    X_rare, X_check, _, _ = ST.extra_targets(False)
    for seed in ST.SEEDS:
        out_dir = os.path.join(OUTPUT_DIR, "mlp", f"{teacher}__base+rare", str(seed))
        if os.path.exists(os.path.join(out_dir, "best_model.pth")):
            continue
        hp = json.load(open(os.path.join(EX8_RUN, str(seed), "metrics.json")))["selected_hp"]
        out, state = fit_mlp_with_state(hp, X_tr, y_tr, X_va, y_va,
                                        {"tr": X_tr, "va": X_va, "ev": X_ev, "rare": X_rare, "check": X_check}, seed)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(state, os.path.join(out_dir, "best_model.pth"))
        for k, name in [("ev", "eval"), ("rare", "rare"), ("check", "check")]:
            np.savez(os.path.join(out_dir, f"probs_{name}.npz"), **out[k])
        meta = {"teacher": teacher, "hp": hp}
        if teacher == "T_v10oJXk":
            old = pl.read_csv(os.path.join(EX8_RUN, str(seed), "predictions.csv"))
            new = ST.to_frame(out["ev"])
            meta["format_agreement_with_ex008"] = float((old["pred_format"].to_numpy() == new["format"].to_numpy()).mean())
            print(f"seed {seed}: ex008 との形式の一致率 {meta['format_agreement_with_ex008']:.3f}")
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
