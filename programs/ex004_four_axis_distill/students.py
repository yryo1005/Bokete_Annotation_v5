"""保存した画像の埋め込みに対する4軸の生徒（ロジスティック回帰，MLP）を学習し，目視ラベルで評価する．

形式は多クラス分類，属性・媒体・事前知識は項目ごとの2値分類として学習する．
予測では，属性・媒体・事前知識は確率0.5以上を真とし，媒体が1つも真でない場合は最も確率の高い媒体を真とする．
形式から必ず真になる属性は補う．埋め込みは ex002 で計算したもの（datasets/ex002_odai_type_distill/features/）を用いる．
ハイパーパラメータは教師ラベルの検証用分割（4,000枚）で，4軸のマクロ F1 の平均が最大となるものを選ぶ．

結果は outputs/ex004_four_axis_distill/student_{分類器}/{教師}__{特徴}/{seed}/ に保存する．

実行例:
    .venv_annotate/bin/python programs/ex004_four_axis_distill/students.py --teacher Qwen3-VL-8B-Instruct_v5_rules
"""
import argparse
import itertools
import os
import sys

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from common import OUTPUT_DIR, evaluate, is_done, save_run  # noqa: E402
from data import (ATTRIBUTES, FORMAT_IMPLIES, FORMATS, KNOWLEDGE, MEDIA, PROJECT_DIR, label_arrays,  # noqa: E402
                  load_eval_labels, load_pool, load_teacher_labels)
from machine_learning_utils import set_seed  # noqa: E402

FEATURE_DIR = os.path.join(PROJECT_DIR, "datasets", "ex002_odai_type_distill", "features")
SEEDS = [0, 1, 2]
GRIDS = {"logreg": [{"C": c} for c in [0.003, 0.01, 0.03]],
         "mlp": [{"lr": lr, "wd": wd} for lr, wd in itertools.product([1e-3, 3e-4], [1e-4, 1e-2])]}
HEADS = {"attr": ATTRIBUTES, "media": MEDIA, "know": KNOWLEDGE}


def load_features(name: str, split: str) -> np.ndarray:
    """ex002 で保存した埋め込みを float32 で読み込む．

    Args:
        name (str): 特徴名（例 "siglip2-so400m-patch14-384"）．
        split (str): "pool" または "eval"．

    Returns:
        np.ndarray: 形状 (N, D) の配列．
    """
    return np.load(os.path.join(FEATURE_DIR, name, f"{split}.npy")).astype(np.float32)


def load_xy(teacher: str, feat: str) -> tuple:
    """特徴量と教師ラベルを読み込み，学習用・検証用・評価用に分ける．

    Args:
        teacher (str): 教師ラベルの名前．
        feat (str): 特徴名．

    Returns:
        tuple: (X_tr, y_tr, X_va, y_va, X_ev)．y は data.label_arrays の辞書．
    """
    lab = load_teacher_labels(teacher)
    index = {fid: i for i, fid in enumerate(load_pool()["fileId"].to_list())}
    X = load_features(feat, "pool")[[index[f] for f in lab["fileId"]]]
    tr, va = (lab["split"] == "train").to_numpy(), (lab["split"] == "val").to_numpy()
    y = label_arrays(lab)
    return X[tr], {k: v[tr] for k, v in y.items()}, X[va], {k: v[va] for k, v in y.items()}, load_features(feat, "eval")


def to_frame(probs: dict) -> pl.DataFrame:
    """確率から4軸の予測の表を作る．

    Args:
        probs (dict): キー format（形状 (N, 4)），attr（(N, 5)），media（(N, 8)），know（(N, 3)）の確率．

    Returns:
        pl.DataFrame: 列 format と各項目（bool）を持つ表．
    """
    fmt = [FORMATS[i] for i in probs["format"].argmax(1)]
    cols = {"format": fmt}
    for head, items in HEADS.items():
        pred = probs[head] >= 0.5
        if head == "media":
            none = ~pred.any(1)
            pred[none, probs[head][none].argmax(1)] = True
        for j, c in enumerate(items):
            cols[c] = pred[:, j].tolist()
    df = pl.DataFrame(cols)
    for f, attr in FORMAT_IMPLIES.items():
        df = df.with_columns((pl.col(attr) | (pl.col("format") == f)).alias(attr))
    return df


def labels_frame(y: dict) -> pl.DataFrame:
    """ラベルの配列を評価用の表に変換する．

    Args:
        y (dict): data.label_arrays の辞書．

    Returns:
        pl.DataFrame: 列 format と各項目（bool）を持つ表．
    """
    cols = {"format": [FORMATS[i] for i in y["format"]]}
    for head, items in HEADS.items():
        for j, c in enumerate(items):
            cols[c] = y[head][:, j].astype(bool).tolist()
    return pl.DataFrame(cols)


def fit_logreg(C: float, X_tr: np.ndarray, y_tr: dict, targets: dict, seed: int) -> dict:
    """ロジスティック回帰を形式（多クラス）と各項目（2値）について学習し，確率を求める．

    Args:
        C (float): 正則化の強さの逆数．
        X_tr (np.ndarray): 形状 (N, D) の学習用の特徴量．
        y_tr (dict): 学習用のラベル．
        targets (dict[str, np.ndarray]): 予測する集合の名前と特徴量．
        seed (int): 乱数のシード．

    Returns:
        dict: 集合の名前をキー，確率の辞書を値とする辞書．
    """
    sc = StandardScaler().fit(X_tr)
    Xs = sc.transform(X_tr)
    T = {k: sc.transform(v) for k, v in targets.items()}
    fm = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xs, y_tr["format"])
    out = {k: {"format": np.zeros((len(v), len(FORMATS)))} for k, v in T.items()}
    for k, v in T.items():
        out[k]["format"][:, fm.classes_] = fm.predict_proba(v)
    for head in HEADS:
        for k in T:
            out[k][head] = np.zeros((len(T[k]), y_tr[head].shape[1]))
        for j in range(y_tr[head].shape[1]):
            y = y_tr[head][:, j]
            if y.min() == y.max():  # 学習データで常に同じ値の項目は定数で予測する
                for k in T:
                    out[k][head][:, j] = y[0]
                continue
            m = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xs, y)
            for k in T:
                out[k][head][:, j] = m.predict_proba(T[k])[:, 1]
    return out


class FourHeadMLP(torch.nn.Module):
    """1層の隠れ層と，形式・属性・媒体・事前知識の4つの出力層を持つ MLP．"""

    def __init__(self, d_in: int, hidden: int = 1024):
        """
        Args:
            d_in (int): 入力の次元．
            hidden (int): 隠れ層の次元．
        """
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.LayerNorm(d_in), torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
                                        torch.nn.Dropout(0.3))
        self.heads = torch.nn.ModuleDict({"format": torch.nn.Linear(hidden, len(FORMATS)),
                                          **{h: torch.nn.Linear(hidden, len(v)) for h, v in HEADS.items()}})

    def forward(self, x: torch.Tensor) -> dict:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, d_in) の入力．

        Returns:
            dict[str, torch.Tensor]: 出力層ごとのロジット．
        """
        h = self.body(x)
        return {k: m(h) for k, m in self.heads.items()}


def four_head_loss(out: dict, y: dict) -> torch.Tensor:
    """形式の交差エントロピーと，属性・媒体・事前知識の2値交差エントロピーの和を求める．

    Args:
        out (dict): FourHeadMLP の出力．
        y (dict): キー format（形状 (B,) の番号）と各軸（形状 (B, K) の0/1）のテンソル．

    Returns:
        torch.Tensor: スカラーの誤差（1データあたりの平均）．
    """
    loss = torch.nn.functional.cross_entropy(out["format"], y["format"])
    for h in HEADS:
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(out[h], y[h])
    return loss


def probs_of(out: dict) -> dict:
    """ロジットを確率に変換する．

    Args:
        out (dict): FourHeadMLP の出力．

    Returns:
        dict[str, np.ndarray]: 出力層ごとの確率．
    """
    p = {"format": torch.softmax(out["format"], 1).cpu().numpy()}
    p.update({h: torch.sigmoid(out[h]).cpu().numpy() for h in HEADS})
    return p


def score(probs: dict, y: dict) -> dict:
    """確率をラベルと比較して指標を求める．

    Args:
        probs (dict): 出力層ごとの確率．
        y (dict): data.label_arrays の辞書．

    Returns:
        dict: common.evaluate の戻り値．
    """
    return evaluate(labels_frame(y), to_frame(probs))


def fit_mlp(hp: dict, X_tr: np.ndarray, y_tr: dict, X_va: np.ndarray, y_va: dict, targets: dict, seed: int,
            epochs: int = 40, batch_size: int = 512) -> dict:
    """4出力の MLP を学習し，検証用の4軸のマクロ F1 の平均が最良のエポックの確率を返す．

    Args:
        hp (dict): lr，wd を持つハイパーパラメータ．
        X_tr (np.ndarray): 学習用の特徴量．
        y_tr (dict): 学習用のラベル．
        X_va (np.ndarray): 検証用の特徴量．
        y_va (dict): 検証用のラベル．
        targets (dict[str, np.ndarray]): 予測する集合の名前と特徴量（"tr"，"va"，"ev"）．
        seed (int): 乱数のシード．
        epochs (int): エポック数．
        batch_size (int): バッチサイズ．

    Returns:
        dict: 集合の名前をキーとする確率の辞書と，エポックごとの履歴 "history"．
    """
    set_seed(seed)
    dev = "cuda"
    T = {k: torch.tensor(v, device=dev) for k, v in targets.items()}
    Y = {"format": torch.tensor(y_tr["format"], device=dev),
         **{h: torch.tensor(y_tr[h], device=dev) for h in HEADS}}
    Yv = {"format": torch.tensor(y_va["format"], device=dev),
          **{h: torch.tensor(y_va[h], device=dev) for h in HEADS}}
    model = FourHeadMLP(X_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    g = torch.Generator().manual_seed(seed)
    hist = {k: [] for k in ["train_loss", "train_overall_macro_f1", "val_loss", "val_overall_macro_f1"]}
    best, best_out = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), generator=g).to(dev)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            loss = four_head_loss(model(T["tr"][b]), {k: v[b] for k, v in Y.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            outs = {k: model(v) for k, v in T.items()}
            l_tr, l_va = four_head_loss(outs["tr"], Y).item(), four_head_loss(outs["va"], Yv).item()
            probs = {k: probs_of(v) for k, v in outs.items()}
        m_tr, m_va = score(probs["tr"], y_tr), score(probs["va"], y_va)
        for k, v in zip(hist, [l_tr, m_tr["overall_macro_f1"], l_va, m_va["overall_macro_f1"]]):
            hist[k].append(float(v))
        if m_va["overall_macro_f1"] > best:
            best, best_out = m_va["overall_macro_f1"], probs
    best_out["history"] = hist
    return best_out


def run(kind: str, teacher: str, feat: str) -> None:
    """1つの条件（分類器，教師，特徴）を全 seed で実行し，検証用で選んだ設定の結果を保存する．

    Args:
        kind (str): "logreg" または "mlp"．
        teacher (str): 教師ラベルの名前．
        feat (str): 特徴名．

    Returns:
        None
    """
    labels = load_eval_labels()
    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_DIR, f"student_{kind}", f"{teacher}__{feat}", str(seed))
        if is_done(run_dir):
            continue
        X_tr, y_tr, X_va, y_va, X_ev = load_xy(teacher, feat)
        targets = {"tr": X_tr, "va": X_va, "ev": X_ev}
        results = []
        for hp in tqdm(GRIDS[kind], desc=f"{kind}:{teacher}:{feat}:{seed}", leave=False):
            out = (fit_logreg(hp["C"], X_tr, y_tr, targets, seed) if kind == "logreg"
                   else fit_mlp(hp, X_tr, y_tr, X_va, y_va, targets, seed))
            results.append((hp, out, score(out["va"], y_va)))
        hp, out, m_va = max(results, key=lambda r: r[2]["overall_macro_f1"])
        extra = {"selected_hp": hp, "train": score(out["tr"], y_tr), "val": m_va,
                 "grid_val_overall_macro_f1": [{"hp": h, "val": m["overall_macro_f1"]} for h, _, m in results]}
        if "history" in out:
            extra["history"] = out["history"]
        meta = {"method": f"student_{kind}", "teacher": teacher, "features": [feat], "n_train": int(len(X_tr)),
                "seed": seed}
        save_run(run_dir, labels, to_frame(out["ev"]), meta, extra)


def main() -> None:
    """指定した教師と特徴で生徒を比較する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", default=["Qwen3-VL-8B-Instruct_v5_rules"])
    parser.add_argument("--features", nargs="+",
                        default=["siglip2-so400m-patch14-384", "siglip2-base-patch16-224", "clip-vit-base-patch32"])
    parser.add_argument("--kinds", nargs="+", default=["logreg", "mlp"])
    args = parser.parse_args()
    for teacher, feat, kind in itertools.product(args.teachers, args.features, args.kinds):
        run(kind, teacher, feat)


if __name__ == "__main__":
    main()
