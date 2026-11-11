"""穴埋め・お題文の制約の順守（ボケがお題の空欄・問いに応じているか）の判定方法を比べる．

自然な違反はまれ（検証データでは穴埋め・お題文で0件）なため，検証データの穴埋め・お題文のボケ（正：元のお題）と，
同じ形式の別のお題に付け替えた組（負：swap_pairs.csv）を判別できるかを，ROC の AUC と，しきい値を正の95%が通る値にしたときの
負の検出率で測る．

方法：
- teacher_{モデル}：VLM の教師（第3版）の complies の回答（0/1）
- centroid：ボケの文の埋め込みと，そのお題の他のボケ（最大50件）の埋め込みの平均とのコサイン類似度．
  お題の文を書き起こさなくても，同じお題の他のボケが「お題が求める答えの方向」を表すことを利用する．お題に他のボケが必要である．
- knn5：お題の他のボケのうち最も似た5件との類似度の平均（典型的な答えから外れた正しいボケも，似たボケが数件あれば拾える）．
- pair_mlp：ボケの文の埋め込みとお題画像の埋め込みを連結して，正の組と付け替えた組（swap_train.csv，ex013 の学習用のボケから作成）を
  見分ける MLP（自己教師的に学習．3 seed）．
- pair_mlp+centroid：上の MLP の確率と centroid の類似度の平均（順位の平均）．

出力：outputs/ex014_boke_labels_v2/compliance.json

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/compliance.py
"""
import json
import os
import sys

import numpy as np
import polars as pl
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATASET_DIR, EX13_DIR, OUTPUT_DIR, _load, load_bokes, load_eval  # noqa: E402

E13 = _load("ex013_embed_for_v2", os.path.join(EX13_DIR, "embed.py"))
FEAT = os.path.join(DATASET_DIR, "feat")


def detection(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    """正の95%が通るしきい値で，負を検出できる割合を求める．

    Args:
        score_pos (np.ndarray): 正の組の得点（高いほど順守）．
        score_neg (np.ndarray): 負の組の得点．

    Returns:
        float: 負の検出率．
    """
    th = np.quantile(score_pos, 0.05)
    return float(np.mean(score_neg < th))


def centroid_scores(texts: list, odai_ids: list, own: list) -> tuple:
    """ボケとお題の他のボケの平均の埋め込みとのコサイン類似度と，最も似た5件との類似度の平均を求める．

    Args:
        texts (list[str]): ボケの文．
        odai_ids (list[int]): 比べるお題のID．
        own (list[str]): 除外する文（自分自身）．

    Returns:
        tuple[np.ndarray, np.ndarray]: 平均とのコサイン類似度，最も似た5件との類似度の平均．
    """
    b = load_bokes(sorted(set(odai_ids)), ("odaiId", "text")).filter(pl.col("text").fill_null("") != "")
    cent = {}
    rng = np.random.default_rng(0)
    groups = {int(k[0]): g["text"].to_list() for k, g in b.group_by("odaiId")}
    all_t, idx = [], {}
    for fid, ts in groups.items():
        ts = list(rng.permutation(ts)[:51])
        idx[fid] = (len(all_t), len(ts))
        all_t += ts
    emb = E13.embed_texts("ruri310", all_t)[0].astype(np.float32)
    q = E13.embed_texts("ruri310", texts)[0].astype(np.float32)
    out, knn = [], []
    for i, (fid, t) in enumerate(zip(odai_ids, own)):
        s, n = idx[fid]
        ts = all_t[s:s + n]
        keep = [j for j in range(n) if ts[j] != t][:50]
        e = emb[s:s + n][keep]
        c = e.mean(0)
        out.append(float(q[i] @ c / (np.linalg.norm(c) + 1e-9)))
        knn.append(float(np.sort(e @ q[i])[-5:].mean()))
    return np.array(out), np.array(knn)


def pair_mlp(seed: int) -> tuple:
    """付け替えの組で MLP を学習し，検証の正と負の確率を返す．

    Args:
        seed (int): 乱数のシード．

    Returns:
        tuple[np.ndarray, np.ndarray]: 正の組と負の組の確率．
    """
    torch.manual_seed(seed)
    tr = pl.read_csv(os.path.join(DATASET_DIR, "swap_train.csv"))
    X = np.concatenate([np.load(os.path.join(FEAT, "swap_train_ruri310.npy")), np.load(os.path.join(FEAT, "swap_train_image.npy"))], 1)
    y = tr["label"].to_numpy().astype(np.float32)
    ev = load_eval()
    m = ev["format"].is_in(["anaume", "odaibun"]).to_numpy()
    Xp = np.concatenate([np.load(os.path.join(FEAT, "eval_ruri310.npy"))[m], np.load(os.path.join(FEAT, "eval_image.npy"))[m]], 1)
    Xn = np.concatenate([np.load(os.path.join(FEAT, "swap_ruri310.npy")), np.load(os.path.join(FEAT, "swap_image.npy"))], 1)
    dev = "cuda"
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)  # noqa: E731
    model = torch.nn.Sequential(torch.nn.LayerNorm(X.shape[1]), torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(),
                                torch.nn.Dropout(0.3), torch.nn.Linear(512, 1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt, yt = T(X), T(y)
    for _ in range(20):
        model.train()
        for s in torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)).split(256):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt[s])[:, 0], yt[s])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(T(Xp))[:, 0]).cpu().numpy(), torch.sigmoid(model(T(Xn))[:, 0]).cpu().numpy()


def main() -> None:
    """全ての方法を評価して保存する．

    Returns:
        None
    """
    ev = load_eval().filter(pl.col("format").is_in(["anaume", "odaibun"])).sort("no")
    sw = pl.read_csv(os.path.join(DATASET_DIR, "swap_pairs.csv")).sort("no")
    assert ev["no"].to_list() == sw["no"].to_list()
    res = {"n_pos": ev.height, "n_neg": sw.height, "natural_violations": int((ev["comp"] == "n").sum())}
    for model in ["q32", "q30a3"]:
        d = os.path.join(OUTPUT_DIR, f"teacher_{model}_v3")
        if os.path.exists(os.path.join(d, "swap.csv")):
            p = ev.select("no").join(pl.read_csv(os.path.join(d, "eval.csv")), on="no")["complies"].to_numpy().astype(float)
            n = pl.read_csv(os.path.join(d, "swap.csv")).sort("no")["complies"].to_numpy().astype(float)
            res[f"teacher_{model}"] = {"auc": roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]),
                                       "pos_pass": float(p.mean()), "neg_detected": float(1 - n.mean())}
    cp, kp = centroid_scores(ev["text"].to_list(), ev["odaiId"].to_list(), ev["text"].to_list())
    cn, kn = centroid_scores(sw["text"].to_list(), sw["odaiId"].to_list(), sw["text"].to_list())
    y = np.r_[np.ones(len(cp)), np.zeros(len(cn))]
    res["centroid"] = {"auc": roc_auc_score(y, np.r_[cp, cn]), "neg_detected_at95": detection(cp, cn)}
    res["knn5"] = {"auc": roc_auc_score(y, np.r_[kp, kn]), "neg_detected_at95": detection(kp, kn)}
    mp, mn = zip(*[pair_mlp(s) for s in range(3)])
    aucs = [roc_auc_score(y, np.r_[a, b]) for a, b in zip(mp, mn)]
    res["pair_mlp"] = {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                       "neg_detected_at95": float(np.mean([detection(a, b) for a, b in zip(mp, mn)]))}
    pm, nm = np.mean(mp, 0), np.mean(mn, 0)
    comb = (rankdata(np.r_[pm, nm]) + rankdata(np.r_[cp, cn])) / 2
    res["pair_mlp+centroid"] = {"auc": roc_auc_score(y, comb), "neg_detected_at95": detection(comb[: len(pm)], comb[len(pm):])}
    for fmt in ["anaume", "odaibun"]:
        m = (ev["format"] == fmt).to_numpy()
        yy = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
        res[f"by_format_{fmt}"] = {"centroid_auc": roc_auc_score(yy, np.r_[cp[m], cn[m]]), "knn5_auc": roc_auc_score(yy, np.r_[kp[m], kn[m]]),
                                   "pair_mlp_auc": roc_auc_score(yy, np.r_[pm[m], nm[m]])}
    json.dump(res, open(os.path.join(OUTPUT_DIR, "compliance.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
