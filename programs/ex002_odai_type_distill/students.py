"""保存した特徴量に対する生徒の分類器（ロジスティック回帰，MLP，k 近傍法）を比較する．

教師ラベルの学習用分割（36,000枚）で学習し，検証用分割（4,000枚，教師ラベル）でハイパーパラメータを選び，
目視の評価用データ（640枚）で最終評価する．結果は
outputs/ex002_odai_type_distill/student_{分類器}/{教師}__{特徴}__n{学習枚数}/{seed}/ に保存する．
各ディレクトリの metrics.json は，選んだハイパーパラメータ，全候補の検証用の指標，学習用・検証用・評価用の指標を含む．

実行例:
    .venv_annotate/bin/python programs/ex002_odai_type_distill/students.py --suite all
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from common import OUTPUT_DIR, evaluate, is_done, save_run  # noqa: E402
from data import DOMAIN_LABELS, FORMAT_LABELS, load_eval_labels, load_pool, load_teacher_labels  # noqa: E402
from features import load_features  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

SEEDS = [0, 1, 2]
SO400M, SIGB, CLIP = "siglip2-so400m-patch14-384", "siglip2-base-patch16-224", "clip-vit-base-patch32"
GRIDS = {
    "logreg": [{"C": c} for c in [0.003, 0.01, 0.03, 0.1]],
    "mlp": [{"hidden": h, "lr": lr, "wd": wd} for h, lr, wd in itertools.product([1024], [1e-3, 3e-4], [1e-4, 1e-2])],
    "knn": [{"k": k} for k in [5, 15, 45]],
}


def load_xy(teacher: str, feats: list, n_train: int, seed: int) -> tuple:
    """特徴量と教師ラベルを読み込み，学習用・検証用・評価用に分ける．

    Args:
        teacher (str): 教師ラベルの名前．
        feats (list[str]): 連結する特徴名のリスト．
        n_train (int): 学習に用いる枚数（None で全て）．
        seed (int): 学習用の部分集合を選ぶ乱数のシード．

    Returns:
        tuple: (X_tr, yf_tr, yd_tr, X_va, yf_va, yd_va, X_ev)．X は形状 (N, D) の float32，y はラベル番号の配列．
    """
    lab = load_teacher_labels(teacher)
    pool_ids = load_pool()["fileId"].to_list()
    index = {fid: i for i, fid in enumerate(pool_ids)}
    X_pool = np.concatenate([_norm_block(load_features(f, "pool"), f) for f in feats], axis=1)
    X_ev = np.concatenate([_norm_block(load_features(f, "eval"), f) for f in feats], axis=1)
    rows = np.array([index[f] for f in lab["fileId"].to_list()])
    yf = np.array([FORMAT_LABELS.index(v) for v in lab["format"]])
    yd = np.array([DOMAIN_LABELS.index(v) for v in lab["domain"]])
    tr = np.where(lab["split"].to_numpy() == "train")[0]
    va = np.where(lab["split"].to_numpy() == "val")[0]
    if n_train is not None and n_train < len(tr):
        tr = np.random.default_rng(seed).choice(tr, n_train, replace=False)
    return X_pool[rows[tr]], yf[tr], yd[tr], X_pool[rows[va]], yf[va], yd[va], X_ev


def _norm_block(x: np.ndarray, name: str) -> np.ndarray:
    """手作りの特徴（blankbox）は log(1 + 100x) で変換し，埋め込みはそのまま返す．

    Args:
        x (np.ndarray): 形状 (N, D) の特徴量．
        name (str): 特徴名．

    Returns:
        np.ndarray: 変換後の特徴量．
    """
    return np.log1p(100 * np.maximum(x, 0)) if name == "blankbox" else x


class TwoHeadMLP(torch.nn.Module):
    """1層の隠れ層と，形式と画風の2つの出力層を持つ MLP．"""

    def __init__(self, d_in: int, hidden: int):
        """
        Args:
            d_in (int): 入力の次元．
            hidden (int): 隠れ層の次元．
        """
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.LayerNorm(d_in), torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
                                        torch.nn.Dropout(0.3))
        self.head_f = torch.nn.Linear(hidden, len(FORMAT_LABELS))
        self.head_d = torch.nn.Linear(hidden, len(DOMAIN_LABELS))

    def forward(self, x: torch.Tensor) -> tuple:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, d_in) の入力．

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 形状 (B, 4) の形式のロジットと，形状 (B, 4) の画風のロジット．
        """
        h = self.body(x)
        return self.head_f(h), self.head_d(h)


def fit_predict(kind: str, hp: dict, data: tuple, seed: int) -> dict:
    """1つのハイパーパラメータで分類器を学習し，学習用・検証用・評価用の確率を返す．

    Args:
        kind (str): "logreg"，"mlp"，"knn" のいずれか．
        hp (dict): ハイパーパラメータ．
        data (tuple): load_xy の戻り値．
        seed (int): 乱数のシード．

    Returns:
        dict: キー "tr"，"va"，"ev" に (形式の確率 (N, 4), 画風の確率 (N, 4)) を持ち，mlp の場合は "history" も持つ辞書．
    """
    X_tr, yf_tr, yd_tr, X_va, yf_va, yd_va, X_ev = data
    set_seed(seed)
    if kind == "logreg":
        sc = StandardScaler().fit(X_tr)
        out = {}
        models = [LogisticRegression(C=hp["C"], max_iter=2000, random_state=seed).fit(sc.transform(X_tr), y)
                  for y in (yf_tr, yd_tr)]
        for key, X in (("tr", X_tr), ("va", X_va), ("ev", X_ev)):
            out[key] = tuple(_full_proba(m, sc.transform(X), 4) for m in models)
        return out
    if kind == "knn":
        return _knn(hp["k"], X_tr, yf_tr, yd_tr, {"tr": X_tr, "va": X_va, "ev": X_ev})
    return _mlp(hp, data, seed)


def _full_proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """学習データに現れなかったクラスの列を0で補った確率を返す．

    Args:
        model: 学習済みの sklearn の分類器．
        X (np.ndarray): 形状 (N, D) の入力．
        n_classes (int): クラス数．

    Returns:
        np.ndarray: 形状 (N, n_classes) の確率．
    """
    p = np.zeros((len(X), n_classes))
    p[:, model.classes_] = model.predict_proba(X)
    return p


def _knn(k: int, X_tr: np.ndarray, yf: np.ndarray, yd: np.ndarray, targets: dict) -> dict:
    """コサイン類似度による k 近傍法で確率（近傍の票の割合）を求める．GPU で計算する．

    Args:
        k (int): 近傍の数．
        X_tr (np.ndarray): 形状 (N, D) の学習用の特徴量．
        yf (np.ndarray): 形状 (N,) の形式のラベル番号．
        yd (np.ndarray): 形状 (N,) の画風のラベル番号．
        targets (dict[str, np.ndarray]): 予測する集合の名前と特徴量．

    Returns:
        dict: 集合の名前をキー，(形式の確率, 画風の確率) を値とする辞書．
    """
    base = torch.nn.functional.normalize(torch.tensor(X_tr, device="cuda"), dim=1)
    out = {}
    for key, X in targets.items():
        q = torch.nn.functional.normalize(torch.tensor(X, device="cuda"), dim=1)
        probs_f, probs_d = [], []
        for s in range(0, len(q), 4096):
            sim = q[s:s + 4096] @ base.T
            if key == "tr":  # 学習用データ自身を近傍から除く
                sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -1
            idx = sim.topk(k, dim=1).indices.cpu().numpy()
            probs_f.append(np.stack([np.bincount(r, minlength=4) / k for r in yf[idx]]))
            probs_d.append(np.stack([np.bincount(r, minlength=4) / k for r in yd[idx]]))
        out[key] = (np.concatenate(probs_f), np.concatenate(probs_d))
    return out


def _mlp(hp: dict, data: tuple, seed: int, epochs: int = 40, batch_size: int = 512) -> dict:
    """2出力の MLP を学習し，検証用の7クラスのマクロ F1 が最良のエポックの確率を返す．

    Args:
        hp (dict): hidden，lr，wd を持つハイパーパラメータ．
        data (tuple): load_xy の戻り値．
        seed (int): 乱数のシード．
        epochs (int): エポック数．
        batch_size (int): バッチサイズ．

    Returns:
        dict: "tr"，"va"，"ev" の確率と，エポックごとの学習用・検証用の誤差と正解率の履歴 "history"．
    """
    X_tr, yf_tr, yd_tr, X_va, yf_va, yd_va, X_ev = data
    dev = "cuda"
    T = {k: torch.tensor(v, device=dev) for k, v in
         {"tr": X_tr, "va": X_va, "ev": X_ev}.items()}
    yf, yd = torch.tensor(yf_tr, device=dev), torch.tensor(yd_tr, device=dev)
    model = TwoHeadMLP(X_tr.shape[1], hp["hidden"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    g = torch.Generator(device="cpu").manual_seed(seed)
    ce = torch.nn.CrossEntropyLoss()
    hist = {k: [] for k in ["train_loss", "train_flat_accuracy", "val_loss", "val_flat_accuracy", "val_flat_macro_f1"]}
    best, best_out = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), generator=g).to(dev)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            lf, ld = model(T["tr"][b])
            loss = ce(lf, yf[b]) + ce(ld, yd[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = {k: tuple(torch.softmax(z, 1).cpu().numpy() for z in model(T[k])) for k in T}
            l_tr = sum(ce(z, y).item() for z, y in zip(model(T["tr"]), (yf, yd)))
            l_va = sum(ce(z, torch.tensor(y, device=dev)).item() for z, y in zip(model(T["va"]), (yf_va, yd_va)))
        m_tr = _metrics(out["tr"], yf_tr, yd_tr)
        m_va = _metrics(out["va"], yf_va, yd_va)
        for k, v in zip(hist, [l_tr, m_tr["flat_accuracy"], l_va, m_va["flat_accuracy"], m_va["flat_macro_f1"]]):
            hist[k].append(float(v))
        if m_va["flat_macro_f1"] > best:
            best, best_out = m_va["flat_macro_f1"], out
    best_out["history"] = hist
    return best_out


def _metrics(probs: tuple, yf: np.ndarray, yd: np.ndarray) -> dict:
    """確率とラベル番号から指標を求める．

    Args:
        probs (tuple[np.ndarray, np.ndarray]): 形式と画風の確率．
        yf (np.ndarray): 形式のラベル番号．
        yd (np.ndarray): 画風のラベル番号．

    Returns:
        dict: common.evaluate の戻り値．
    """
    pf, pd_ = probs[0].argmax(1), probs[1].argmax(1)
    return evaluate([FORMAT_LABELS[i] for i in yf], [DOMAIN_LABELS[i] for i in yd],
                    [FORMAT_LABELS[i] for i in pf], [DOMAIN_LABELS[i] for i in pd_])


def run(kind: str, teacher: str, feats: list, n_train: int = None) -> None:
    """1つの条件（分類器，教師，特徴，学習枚数）を全 seed で実行し，検証用で選んだ設定の結果を保存する．

    Args:
        kind (str): 分類器の種類．
        teacher (str): 教師ラベルの名前．
        feats (list[str]): 特徴名のリスト．
        n_train (int): 学習に用いる枚数（None で全て）．

    Returns:
        None
    """
    labels = load_eval_labels()
    tag = f"{teacher}__{'+'.join(feats)}__n{n_train or 'all'}"
    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_DIR, f"student_{kind}", tag, str(seed))
        if is_done(run_dir):
            continue
        data = load_xy(teacher, feats, n_train, seed)
        results = []
        for hp in tqdm(GRIDS[kind], desc=f"{kind}:{tag}:{seed}", leave=False):
            out = fit_predict(kind, hp, data, seed)
            results.append((hp, out, _metrics(out["va"], data[4], data[5])))
        hp, out, m_va = max(results, key=lambda r: r[2]["flat_macro_f1"])
        conf = out["ev"][0].max(1) * out["ev"][1].max(1)
        extra = {"selected_hp": hp, "train": _metrics(out["tr"], data[1], data[2]), "val": m_va,
                 "grid_val_macro_f1": [{"hp": h, "val_flat_macro_f1": m["flat_macro_f1"]} for h, _, m in results]}
        if "history" in out:
            extra["history"] = out["history"]
        meta = {"method": f"student_{kind}", "teacher": teacher, "features": feats, "n_train": len(data[1]),
                "seed": seed}
        save_run(run_dir, labels, [FORMAT_LABELS[i] for i in out["ev"][0].argmax(1)],
                 [DOMAIN_LABELS[i] for i in out["ev"][1].argmax(1)], meta, extra, {"confidence": conf})


def main() -> None:
    """比較の組（suite）を実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", nargs="+", default=["all"],
                        choices=["all", "classifier", "features", "teacher", "size"])
    parser.add_argument("--teachers", nargs="*", default=[])
    parser.add_argument("--main_teacher", default="Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    suites = {"classifier", "features", "teacher", "size"} if "all" in args.suite else set(args.suite)
    t = args.main_teacher
    if "classifier" in suites:
        for kind in ["logreg", "mlp", "knn"]:
            run(kind, t, [SO400M])
    if "features" in suites:
        for feats in [[CLIP], [SIGB], [SO400M, CLIP], [SO400M, "blankbox"], [SIGB, "blankbox"], [SIGB, CLIP]]:
            for kind in ["logreg", "mlp"]:
                run(kind, t, feats)
    if "teacher" in suites:
        for teacher in args.teachers:
            for kind in ["logreg", "mlp"]:
                run(kind, teacher, [SO400M])
    if "size" in suites:
        for n in [1000, 2500, 5000, 10000, 20000]:
            for kind in ["logreg", "mlp"]:
                run(kind, t, [SO400M], n)


if __name__ == "__main__":
    main()
