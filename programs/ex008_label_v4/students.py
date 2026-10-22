"""定義書 v4 の3軸の生徒（SigLIP2-so400m の埋め込みに対する MLP）を学習し，評価する（ex007 の students.py を基にした）．

ex007 からの変更点：
- 項目は定義書 v4（文字ありにロゴ等を含む）であり，data.py（ex008）から読み込む．
- 重み付き損失を廃止し，重みなしのみとする．
- 全 seed で，実データの目視ラベルの画像（無作為抽出160枚と層別抽出220枚）の確率も求めて保存する（check_probs.npz）．
- seed 0 では未使用の実データ（約29.5万枚）にアノテーションする（real_annotation_v4.parquet）．

結果は outputs/ex008_label_v4/student_mlp/{教師}__{データ}__unweighted/{seed}/ に保存する．

実行例:
    .venv_annotate/bin/python programs/ex008_label_v4/students.py --teacher T_v10
"""
import argparse
import itertools
import os
import sys

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from common import OUTPUT_DIR, evaluate, is_done, save_run  # noqa: E402
from data import (ATTRIBUTES, DATASET_DIR, FORMAT_IMPLIES, FORMATS, MEDIA, PROJECT_DIR, label_arrays,  # noqa: E402
                  load_eval_labels, load_pool, load_teacher_labels)
from machine_learning_utils import set_seed  # noqa: E402

FEATURE_DIR = os.path.join(PROJECT_DIR, "datasets", "ex002_odai_type_distill", "features")
SEEDS = [0, 1, 2]
GRID = [{"lr": lr, "wd": wd} for lr, wd in itertools.product([1e-3, 3e-4], [1e-4, 1e-2])]
HEADS = {"attr": ATTRIBUTES, "media": MEDIA}
ALPHA = 0.5


def load_features(name: str, split: str) -> np.ndarray:
    """ex002 で保存した埋め込みを float32 で読み込む．

    Args:
        name (str): 特徴名．
        split (str): "pool" または "eval"．

    Returns:
        np.ndarray: 形状 (N, D) の配列．
    """
    return np.load(os.path.join(FEATURE_DIR, name, f"{split}.npy")).astype(np.float32)


def load_xy(teacher: str, use_rare: bool) -> tuple:
    """SigLIP2-so400m の埋め込みと教師ラベルを読み込み，学習用・検証用・評価用に分ける．

    Args:
        teacher (str): 教師の構成の名前．
        use_rare (bool): 少ない区分の追加分を学習用に加えるか．

    Returns:
        tuple: (X_tr, y_tr, X_va, y_va, X_ev)．検証用は基本の4万枚の検証用分割のみとする．
    """
    lab = load_teacher_labels(f"{teacher}_pool", f"{teacher}_rare" if use_rare else None)
    feats = {fid: ("pool", i) for i, fid in enumerate(load_pool()["fileId"].to_list())}
    X_pool = load_features("siglip2-so400m-patch14-384", "pool")
    if use_rare:
        cand = pl.read_csv(os.path.join(DATASET_DIR, "candidates", "cand.csv"))["fileId"].to_list()
        feats.update({fid: ("cand", i) for i, fid in enumerate(cand)})
        X_cand = np.load(os.path.join(DATASET_DIR, "embeddings", "cand_so400m.npy")).astype(np.float32)
    rows = [feats[f] for f in lab["fileId"]]
    X = np.stack([X_pool[i] if s == "pool" else X_cand[i] for s, i in rows])
    tr = (lab["split"] == "train").to_numpy()
    va = ((lab["split"] == "val") & (lab["source"] == "base")).to_numpy()
    y = label_arrays(lab)
    return X[tr], {k: v[tr] for k, v in y.items()}, X[va], {k: v[va] for k, v in y.items()}, \
        load_features("siglip2-so400m-patch14-384", "eval")


# weighted_sel で重み付けの対象から外す属性（多い区分の適合率を下げた項目．report_007）
EXCLUDE_FROM_WEIGHT = []  # 定義書 v3 では重み付けの対象から外す項目はない


def class_weights(y: dict, device: str, exclude: tuple = ()) -> dict:
    """重み付き損失の重みを学習用データの出現率から求める．

    Args:
        y (dict): 学習用のラベル（label_arrays の辞書）．
        device (str): テンソルを置くデバイス．
        exclude (tuple[str]): 正例の重みを1（重みなし）にする属性の名前．

    Returns:
        dict: キー format（形状 (4,) のクラスの重み），attr・media（形状 (K,) の pos_weight）のテンソル．
    """
    cnt = np.bincount(y["format"], minlength=len(FORMATS)).astype(float)
    w = (cnt.sum() / np.maximum(cnt, 1)) ** ALPHA
    out = {"format": torch.tensor(w / w.mean(), dtype=torch.float32, device=device)}
    for h in HEADS:
        pos = y[h].sum(0)
        w = ((len(y[h]) - pos) / np.maximum(pos, 1)) ** ALPHA
        for name in exclude:
            if name in HEADS[h]:
                w[HEADS[h].index(name)] = 1.0
        out[h] = torch.tensor(w, dtype=torch.float32, device=device)
    return out


def weights_for(mode, y: dict, device: str):
    """重み付けの方式から損失の重みを求める．

    Args:
        mode (bool | str): False または "unweighted"（重みなし），True または "weighted"（全項目に重み），
            "weighted_sel"（合成画像とその他の文字を除いて重み）．
        y (dict): 学習用のラベル．
        device (str): デバイス．

    Returns:
        dict | None: class_weights の戻り値，または重みなしの場合は None．
    """
    mode = {False: "unweighted", True: "weighted"}.get(mode, mode)
    if mode == "unweighted":
        return None
    return class_weights(y, device, tuple(EXCLUDE_FROM_WEIGHT) if mode == "weighted_sel" else ())


def three_head_loss(out: dict, y: dict, weights: dict = None) -> torch.Tensor:
    """形式の交差エントロピーと属性・媒体の2値交差エントロピーの和を求める．

    Args:
        out (dict): 出力層ごとのロジット．
        y (dict): ラベルのテンソル．
        weights (dict): class_weights の戻り値（None で重みなし）．

    Returns:
        torch.Tensor: スカラーの誤差．
    """
    w = weights or {}
    loss = torch.nn.functional.cross_entropy(out["format"], y["format"], weight=w.get("format"))
    for h in HEADS:
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(out[h], y[h], pos_weight=w.get(h))
    return loss


class ThreeHeadMLP(torch.nn.Module):
    """1層の隠れ層と，形式・属性・媒体の3つの出力層を持つ MLP．"""

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


def probs_of(out: dict) -> dict:
    """ロジットを確率に変換する．

    Args:
        out (dict): 出力層ごとのロジット．

    Returns:
        dict[str, np.ndarray]: 出力層ごとの確率．
    """
    p = {"format": torch.softmax(out["format"].float(), 1).cpu().numpy()}
    p.update({h: torch.sigmoid(out[h].float()).cpu().numpy() for h in HEADS})
    return p


def tune_thresholds(probs: dict, y: dict) -> dict:
    """属性と媒体の項目ごとに，検証用データの F1 が最大となる判定のしきい値を求める．

    Args:
        probs (dict): 検証用データの出力層ごとの確率．
        y (dict): 検証用データのラベル．

    Returns:
        dict: キー attr・media に，形状 (K,) のしきい値の配列を持つ辞書．
    """
    from sklearn.metrics import f1_score

    grid = np.arange(0.05, 0.96, 0.05)
    out = {}
    for h in HEADS:
        th = np.full(probs[h].shape[1], 0.5)
        for j in range(probs[h].shape[1]):
            yy = y[h][:, j].astype(bool)
            if yy.sum() == 0:
                continue
            f = [f1_score(yy, probs[h][:, j] >= t, zero_division=0) for t in grid]
            th[j] = grid[int(np.argmax(f))]
        out[h] = th
    return out


def to_frame(probs: dict, thresholds: dict = None) -> pl.DataFrame:
    """確率から3軸の予測の表を作る．

    Args:
        probs (dict): 出力層ごとの確率．
        thresholds (dict): 属性と媒体の項目ごとのしきい値（None で全て0.5）．

    Returns:
        pl.DataFrame: 列 format と各項目（bool）を持つ表．
    """
    cols = {"format": [FORMATS[i] for i in probs["format"].argmax(1)]}
    for head, items in HEADS.items():
        pred = probs[head] >= (thresholds[head] if thresholds else 0.5)
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
        y (dict): label_arrays の辞書．

    Returns:
        pl.DataFrame: 列 format と各項目（bool）を持つ表．
    """
    cols = {"format": [FORMATS[i] for i in y["format"]]}
    for head, items in HEADS.items():
        for j, c in enumerate(items):
            cols[c] = y[head][:, j].astype(bool).tolist()
    return pl.DataFrame(cols)


def score(probs: dict, y: dict) -> dict:
    """確率をラベルと比較して指標を求める．

    Args:
        probs (dict): 出力層ごとの確率．
        y (dict): label_arrays の辞書．

    Returns:
        dict: common.evaluate の戻り値．
    """
    return evaluate(labels_frame(y), to_frame(probs))


def fit_mlp(hp: dict, X_tr: np.ndarray, y_tr: dict, X_va: np.ndarray, y_va: dict, targets: dict, seed: int,
            weighted: bool = False, epochs: int = 40, batch_size: int = 512) -> dict:
    """3出力の MLP を学習し，検証用の3軸のマクロ F1 の平均が最良のエポックの確率を返す．

    Args:
        hp (dict): lr，wd を持つハイパーパラメータ．
        X_tr (np.ndarray): 学習用の特徴量．
        y_tr (dict): 学習用のラベル．
        X_va (np.ndarray): 検証用の特徴量．
        y_va (dict): 検証用のラベル．
        targets (dict[str, np.ndarray]): 予測する集合の名前と特徴量（"tr"，"va" を含む）．
        seed (int): 乱数のシード．
        weighted (bool | str): 重み付けの方式（weights_for を参照）．
        epochs (int): エポック数．
        batch_size (int): バッチサイズ．

    Returns:
        dict: 集合の名前をキーとする確率の辞書と，エポックごとの履歴 "history"．
    """
    set_seed(seed)
    dev = "cuda"
    T = {k: torch.tensor(v, device=dev) for k, v in targets.items()}
    Y = {"format": torch.tensor(y_tr["format"], device=dev), **{h: torch.tensor(y_tr[h], device=dev) for h in HEADS}}
    Yv = {"format": torch.tensor(y_va["format"], device=dev), **{h: torch.tensor(y_va[h], device=dev) for h in HEADS}}
    w = weights_for(weighted, y_tr, dev)
    model = ThreeHeadMLP(X_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    g = torch.Generator().manual_seed(seed)
    hist = {k: [] for k in ["train_loss", "train_overall_macro_f1", "val_loss", "val_overall_macro_f1"]}
    best, best_out = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), generator=g).to(dev)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            loss = three_head_loss(model(T["tr"][b]), {k: v[b] for k, v in Y.items()}, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            outs = {k: model(v) for k, v in T.items()}
            l_tr, l_va = three_head_loss(outs["tr"], Y, w).item(), three_head_loss(outs["va"], Yv, w).item()
            probs = {k: probs_of(v) for k, v in outs.items()}
        m_tr, m_va = score(probs["tr"], y_tr), score(probs["va"], y_va)
        for k, v in zip(hist, [l_tr, m_tr["overall_macro_f1"], l_va, m_va["overall_macro_f1"]]):
            hist[k].append(float(v))
        if m_va["overall_macro_f1"] > best:
            best, best_out = m_va["overall_macro_f1"], probs
    best_out["history"] = hist
    return best_out


def extra_targets(with_real: bool) -> tuple:
    """少ない区分の評価用データ，実データの目視ラベルの画像，未使用の実データの埋め込みを読み込む．

    Args:
        with_real (bool): 未使用の実データの埋め込みも読み込むか．

    Returns:
        tuple: (少ない区分の評価用データの埋め込み (800, D)，目視ラベルの画像の埋め込み (380, D)，
        実データのお題IDのリスト，実データの埋め込み (N, D))．with_real が偽の場合，後の2つは None．
    """
    from data import load_rare_labels

    cand = pl.read_csv(os.path.join(DATASET_DIR, "candidates", "cand.csv"))["fileId"].to_list()
    index = {f: i for i, f in enumerate(cand)}
    E = np.load(os.path.join(DATASET_DIR, "embeddings", "cand_so400m.npy"), mmap_mode="r")
    X_rare = np.asarray(E[[index[f] for f in load_rare_labels()["fileId"]]], dtype=np.float32)
    check_ids = pl.read_csv(os.path.join(DATASET_DIR, "candidates", "real_check_v4.csv"))["fileId"].to_list()
    X_check = np.asarray(E[[index[f] for f in check_ids]], dtype=np.float32)
    if not with_real:
        return X_rare, X_check, None, None
    real_ids = pl.read_parquet(os.path.join(DATASET_DIR, "real_annotation.parquet"))["fileId"].to_list()
    return X_rare, X_check, real_ids, np.asarray(E[[index[f] for f in real_ids]], dtype=np.float32)


def save_real_annotation(run_dir: str, probs: dict, real_ids: list) -> None:
    """未使用の実データへのアノテーション（確率と予測）を保存する．

    Args:
        run_dir (str): 保存先のディレクトリ．
        probs (dict): 実データの出力層ごとの確率．
        real_ids (list[int]): 実データのお題ID．

    Returns:
        None
    """
    pred = to_frame(probs)
    cols = {"fileId": real_ids, **{f"p_format_{c}": probs["format"][:, i] for i, c in enumerate(FORMATS)},
            **{f"p_{c}": probs["attr"][:, i] for i, c in enumerate(ATTRIBUTES)},
            **{f"p_{c}": probs["media"][:, i] for i, c in enumerate(MEDIA)}}
    pl.concat([pl.DataFrame(cols), pred], how="horizontal").write_parquet(os.path.join(run_dir, "real_annotation_v4.parquet"))


def run(teacher: str, use_rare: bool, weighted: bool, tune: bool = False) -> None:
    """1つの条件を全 seed で実行し，検証用で選んだ設定の結果を保存する．

    Args:
        teacher (str): 教師の構成の名前．
        use_rare (bool): 少ない区分の追加分を用いるか．
        weighted (bool | str): 重み付けの方式（weights_for を参照）．
        tune (bool): 属性と媒体の判定のしきい値を検証用データで調整するか．

    Returns:
        None
    """
    labels = load_eval_labels()
    wname = {False: "unweighted", True: "weighted"}.get(weighted, weighted)
    tag = f"{teacher}__{'base+rare' if use_rare else 'base'}__{wname}"
    tag += "__thr" if tune else ""
    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_DIR, "student_mlp", tag, str(seed))
        if is_done(run_dir):
            continue
        X_tr, y_tr, X_va, y_va, X_ev = load_xy(teacher, use_rare)
        X_rare, X_check, real_ids, X_real = extra_targets(seed == 0)
        targets = {"tr": X_tr, "va": X_va, "ev": X_ev, "rare": X_rare, "check": X_check}
        if X_real is not None:
            targets["real"] = X_real
        results = []
        for hp in tqdm(GRID, desc=f"mlp:{tag}:{seed}", leave=False):
            out = fit_mlp(hp, X_tr, y_tr, X_va, y_va, targets, seed, weighted)
            results.append((hp, out, score(out["va"], y_va)))
        hp, out, m_va = max(results, key=lambda r: r[2]["overall_macro_f1"])
        th = tune_thresholds(out["va"], y_va) if tune else None
        extra = {"thresholds": {k: v.tolist() for k, v in th.items()}} if tune else {}
        extra.update({"selected_hp": hp, "train": score(out["tr"], y_tr), "val": m_va, "history": out["history"],
                      "grid_val_overall_macro_f1": [{"hp": h, "val": m["overall_macro_f1"]} for h, _, m in results]})
        meta = {"method": "student_mlp", "teacher": teacher, "use_rare": use_rare, "weighted": weighted, "tune": tune,
                "features": ["siglip2-so400m-patch14-384"], "n_train": int(len(X_tr)), "seed": seed, "alpha": ALPHA}
        save_run(run_dir, labels, to_frame(out["ev"], th), meta, extra)
        np.savez(os.path.join(run_dir, "rare_probs.npz"), **out["rare"])
        np.savez(os.path.join(run_dir, "check_probs.npz"), **out["check"])
        if X_real is not None:
            save_real_annotation(run_dir, out["real"], real_ids)


def main() -> None:
    """少ない区分の追加の有無の2条件を比較する（重み付き損失は廃止した）．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="T_v10")
    parser.add_argument("--no_rare", action="store_true", help="追加分を用いる条件を省略する")
    parser.add_argument("--tune", action="store_true", help="しきい値を調整する条件も実行する")
    args = parser.parse_args()
    for use_rare, weighted, tune in itertools.product([False] if args.no_rare else [False, True], [False],
                                                      [False, True] if args.tune else [False]):
        run(args.teacher, use_rare, weighted, tune)


if __name__ == "__main__":
    main()
