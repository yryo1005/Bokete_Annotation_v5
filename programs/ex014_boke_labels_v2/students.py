"""発話の型・必要な知識・画像中の文字への依存の生徒を学習し，検証データ（620件）の予測を保存する．

学習データ（--data）：
- base：ex013 の学習用のボケ3.5万件と教師 30B-A3B（第2版）のラベル
- aug_random：水増し用の集合のうち無作為の5千件と教師 32B（第3版）のラベル
- aug_all：水増し用の集合の全て（無作為5千件＋少ない発話の型の候補約4千件）と教師 32B（第3版）のラベル
- aug_all_w：aug_all に，発話の型の出現の逆数の重みを付けた損失
入力（--feats）：ruri（文の埋め込み）＋fmt（お題の形式・属性・媒体）＋img（お題画像の埋め込み）＋pos（品詞の特徴）の組み合わせ．
補正（--stack）：生徒の確率とお題の特徴・品詞の特徴を入力に，検証データの目視のラベルでロジスティック回帰を学習する
（調整用で学習して最終評価用を予測し，逆も行う2分割）．

出力：outputs/ex014_boke_labels_v2/student/{データ}__{入力}[__stack]/{seed}/eval.csv

実行例:
    .venv_annotate/bin/python programs/ex014_boke_labels_v2/students.py
"""
import itertools
import json
import os
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import DATASET_DIR, EX13_DATASET, EX13_OUTPUT, KNOWS, OUTPUT_DIR, VOICES, load_eval, load_odai_labels  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

FEAT = os.path.join(DATASET_DIR, "feat")
EX13_EMB = os.path.join(EX13_DATASET, "emb")
FORMATS = ["karuta", "anaume", "odaibun", "normal"]
ODAI_FEATS = ["has_blank", "has_karuta_mark", "has_text", "photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
HEADS = {"voice": len(VOICES), "know": len(KNOWS), "uses_text": 1}
SEEDS = [0, 1, 2]
FEAT_SETS = ["ruri_fmt", "ruri_fmt_pos", "ruri_fmt_img", "ruri_fmt_img_pos"]


def odai_feats(df: pl.DataFrame) -> np.ndarray:
    """お題の形式（one-hot）と属性・媒体の特徴を作る．

    Args:
        df (pl.DataFrame): 列 odaiId, format を持つ表．

    Returns:
        np.ndarray: 形状 (N, 14) の特徴．
    """
    lab = df.select("odaiId").join(load_odai_labels(df["odaiId"].unique().to_list()).drop("format"), on="odaiId", how="left",
                                   maintain_order="left")
    f = np.stack([(df["format"] == c).to_numpy() for c in FORMATS], 1)
    return np.concatenate([f, lab.select(ODAI_FEATS).fill_null(False).to_numpy()], 1).astype(np.float32)


def load_train(data: str) -> tuple:
    """学習データの表と，各特徴の配列を読み込む．

    Args:
        data (str): base / aug_random / aug_all / aug_all_w．

    Returns:
        tuple[pl.DataFrame, dict]: 列 odaiId, format, voice, knowledge, uses_image_text を持つ表と，特徴の辞書．
    """
    if data == "base":
        df = pl.read_parquet(os.path.join(EX13_DATASET, "train_pool.parquet")).select("odaiId", "format")
        lab = pl.read_csv(os.path.join(EX13_OUTPUT, "teacher_q30a3_v2", "train.csv"))
        df = pl.concat([df, lab.select("voice", pl.col("knowledge").cast(pl.String), "uses_image_text")], how="horizontal")
        feats = {"ruri": np.load(os.path.join(EX13_EMB, "train_ruri310.npy")), "img": np.load(os.path.join(EX13_EMB, "train_image.npy")),
                 "pos": np.load(os.path.join(FEAT, "train_pos.npy"))}
        return df, feats, np.ones(df.height, bool)
    pool = pl.read_csv(os.path.join(DATASET_DIR, "aug_pool.csv"))
    lab = pl.read_csv(os.path.join(OUTPUT_DIR, "teacher_q32_v3", "aug.csv"))
    df = pl.concat([pool.select("odaiId", "format", "source"), lab.select("voice", pl.col("knowledge").cast(pl.String), "uses_image_text")],
                   how="horizontal")
    feats = {k: np.load(os.path.join(FEAT, f"aug_{n}.npy")) for k, n in [("ruri", "ruri310"), ("img", "image"), ("pos", "pos")]}
    keep = (df["source"] == "random").to_numpy() if data == "aug_random" else np.ones(df.height, bool)
    return df, feats, keep


def assemble(feat_set: str, feats: dict, ofeat: np.ndarray) -> np.ndarray:
    """特徴の組み合わせを連結する．

    Args:
        feat_set (str): FEAT_SETS のいずれか．
        feats (dict): ruri, img, pos の配列．
        ofeat (np.ndarray): お題の特徴．

    Returns:
        np.ndarray: 連結した特徴．
    """
    parts = [feats["ruri"].astype(np.float32), ofeat]
    if "img" in feat_set:
        parts.append(feats["img"].astype(np.float32))
    if "pos" in feat_set:
        parts.append(feats["pos"].astype(np.float32))
    return np.concatenate(parts, 1)


class MultiHead(torch.nn.Module):
    """隠れ層1層と項目ごとの出力層を持つ MLP．"""

    def __init__(self, d_in: int, hidden: int = 512):
        """
        Args:
            d_in (int): 入力の次元．
            hidden (int): 隠れ層の次元．
        """
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.LayerNorm(d_in), torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Dropout(0.3))
        self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(hidden, n) for k, n in HEADS.items()})

    def forward(self, x: torch.Tensor) -> dict:
        """順伝播．

        Args:
            x (torch.Tensor): 形状 (B, d_in) の入力．

        Returns:
            dict[str, torch.Tensor]: 項目ごとのロジット．
        """
        h = self.body(x)
        return {k: m(h) for k, m in self.heads.items()}


def fit(X: np.ndarray, Y: dict, Xe: np.ndarray, seed: int, voice_weight: np.ndarray = None, epochs: int = 30) -> dict:
    """MLP を学習し（1割を検証用にし誤差が最小のエポックを選ぶ），検証データの確率を返す．

    Args:
        X (np.ndarray): 学習用の特徴．
        Y (dict): 学習用の目標（voice, know は番号，uses_text は 0/1）．
        Xe (np.ndarray): 検証データの特徴．
        seed (int): 乱数のシード．
        voice_weight (np.ndarray): 発話の型の重み（None で重みなし）．
        epochs (int): エポック数．

    Returns:
        dict[str, np.ndarray]: 項目ごとの確率．
    """
    set_seed(seed)
    perm = np.random.default_rng(seed).permutation(len(X))
    va, tr = perm[: len(X) // 10], perm[len(X) // 10:]
    T = lambda a: torch.tensor(a, device="cuda")  # noqa: E731
    w = T(voice_weight.astype(np.float32)) if voice_weight is not None else None

    def loss_fn(o: dict, y: dict) -> torch.Tensor:
        """誤差の和を求める．

        Args:
            o (dict): ロジット．
            y (dict): 目標．

        Returns:
            torch.Tensor: スカラーの誤差．
        """
        return (torch.nn.functional.cross_entropy(o["voice"], y["voice"], weight=w)
                + torch.nn.functional.cross_entropy(o["know"], y["know"])
                + torch.nn.functional.binary_cross_entropy_with_logits(o["uses_text"][:, 0], y["uses_text"]))

    Xt, Xv = T(X[tr].astype(np.float32)), T(X[va].astype(np.float32))
    Yt, Yv = {k: T(v[tr]) for k, v in Y.items()}, {k: T(v[va]) for k, v in Y.items()}
    model = MultiHead(X.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state = 1e9, None
    for _ in range(epochs):
        model.train()
        for s in torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed)).split(256):
            loss = loss_fn(model(Xt[s]), {k: v[s] for k, v in Yt.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lv = loss_fn(model(Xv), Yv).item()
        if lv < best:
            best, state = lv, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        o = model(T(Xe.astype(np.float32)))
    return {"voice": torch.softmax(o["voice"], 1).cpu().numpy(), "know": torch.softmax(o["know"], 1).cpu().numpy(),
            "uses_text": torch.sigmoid(o["uses_text"][:, 0]).cpu().numpy()}


def to_pred(ev: pl.DataFrame, p: dict) -> pl.DataFrame:
    """確率から予測の表を作る（穴埋め・お題文の発話の型は回答とする）．

    Args:
        ev (pl.DataFrame): 検証データ．
        p (dict): 項目ごとの確率．

    Returns:
        pl.DataFrame: 列 no, voice, knowledge, uses_image_text の表．
    """
    voice = [VOICES[i] for i in p["voice"].argmax(1)]
    voice = ["A" if f in ("anaume", "odaibun") else v for f, v in zip(ev["format"], voice)]
    return pl.DataFrame({"no": ev["no"], "voice": voice, "knowledge": [KNOWS[i] for i in p["know"].argmax(1)],
                         "uses_image_text": p["uses_text"] > 0.5})


def stack(ev: pl.DataFrame, p: dict, extra: np.ndarray) -> pl.DataFrame:
    """生徒の確率を目視のラベルで補正する（調整用と最終評価用を入れ替えたロジスティック回帰）．

    Args:
        ev (pl.DataFrame): 検証データ．
        p (dict): 生徒の確率．
        extra (np.ndarray): 追加の特徴（お題の特徴と品詞の特徴）．

    Returns:
        pl.DataFrame: 予測の表．
    """
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([p["voice"], p["know"], p["uses_text"][:, None], extra], 1)
    targets = {"voice": np.array([VOICES.index(v) for v in ev["voice"]]), "know": np.array([KNOWS.index(k) for k in ev["know"]]),
               "uses_text": ev["uses_text"].to_numpy().astype(int)}
    out = {"voice": np.zeros((len(ev), len(VOICES))), "know": np.zeros((len(ev), len(KNOWS))), "uses_text": np.zeros(len(ev))}
    for tr_split in ["dev", "test"]:
        tr = (ev["split"] == tr_split).to_numpy()
        te = ~tr
        for k, y in targets.items():
            clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(X[tr], y[tr])
            pr = clf.predict_proba(X[te])
            if k == "uses_text":
                out[k][te] = pr[:, list(clf.classes_).index(1)]
            else:
                full = np.zeros((te.sum(), out[k].shape[1]))
                full[:, clf.classes_] = pr
                out[k][te] = full
    return to_pred(ev, out)


def main() -> None:
    """全ての学習データ・入力の組み合わせで学習・予測し，補正も行う．

    Returns:
        None
    """
    ev = load_eval()
    ev_feats = {k: np.load(os.path.join(FEAT, f"eval_{n}.npy")) for k, n in [("ruri", "ruri310"), ("img", "image"), ("pos", "pos")]}
    ev_ofeat = odai_feats(ev)
    for data in ["base", "aug_random", "aug_all", "aug_all_w"]:
        tr, feats, keep = load_train(data.replace("_w", "") if data == "aug_all_w" else data)
        tr = tr.filter(pl.Series(keep))
        feats = {k: v[keep] for k, v in feats.items()}
        ofeat = odai_feats(tr)
        Y = {"voice": np.array([VOICES.index(v) for v in tr["voice"]]), "know": np.array([KNOWS.index(k) for k in tr["knowledge"]]),
             "uses_text": tr["uses_image_text"].to_numpy().astype(np.float32)}
        vw = None
        if data == "aug_all_w":
            cnt = np.bincount(Y["voice"], minlength=len(VOICES)).astype(float)
            vw = np.where(cnt > 0, cnt.sum() / (len(VOICES) * np.maximum(cnt, 1)), 0.0)
        for fs in FEAT_SETS:
            X, Xe = assemble(fs, feats, ofeat), assemble(fs, ev_feats, ev_ofeat)
            for seed in SEEDS:
                p = fit(X, Y, Xe, seed, vw)
                for name, pred in [(f"{data}__{fs}", to_pred(ev, p)),
                                   (f"{data}__{fs}__stack", stack(ev, p, np.concatenate([ev_ofeat, ev_feats["pos"]], 1)))]:
                    d = os.path.join(OUTPUT_DIR, "student", name, str(seed))
                    os.makedirs(d, exist_ok=True)
                    pred.write_csv(os.path.join(d, "eval.csv"))
            print(data, fs, "voice train dist", np.bincount(Y["voice"], minlength=6).tolist(), flush=True)
    json.dump({"voices": VOICES, "knows": KNOWS}, open(os.path.join(OUTPUT_DIR, "student", "meta.json"), "w"))


if __name__ == "__main__":
    main()
