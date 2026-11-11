"""教師のラベルで，発話の型（6区分）と画像外の知識（2値）の生徒を学習し，検証データ（620件）の予測を保存する．

学習データ：ex014 の水増し用の集合（無作為5千件＋少ない発話の型の候補約4千件，計9,041件）．@extra を付けると，
ex013 の学習用のボケから重複と検証データのお題を除いた約2.6万件を加える．
ラベル（--labels）：
- ex014：ex014 の教師 32B（第3版の指示）．画像外の知識は4段階の O を「はい」とみなす（対照）．
- {モデル}/{方法}：ex015 の教師（teacher.py）のラベル．
- {モデル}/{方法1}+{モデル}/{方法2}：発話の型を方法1，画像外の知識を方法2のラベルとする．
入力：ruri_fmt（文の埋め込み＋お題の形式・属性・媒体），ruri_fmt_img（＋お題画像の埋め込み）．品詞（MeCab）の特徴は用いない．
損失：発話の型は出現の逆数で重みを付けた交差エントロピー，画像外の知識は正例の比で重みを付けた2値交差エントロピー．
補正（__stack）：生徒の確率とお題の特徴を入力に，目視のラベルでロジスティック回帰を学習する（調整用と最終評価用を入れ替える2分割）．

出力：outputs/ex015_boke_labels_v3/student/{ラベル}__{入力}[__stack]/{seed}/eval.csv（列 no, voice, outside, p_outside）

実行例:
    .venv_annotate/bin/python programs/ex015_boke_labels_v3/students.py --labels ex014 q32/dec_fs_cap
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import EX13_DATASET, EX14_DATASET, EX14_OUTPUT, OUTPUT_DIR, load_eval, load_odai_labels  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

FEAT = os.path.join(EX14_DATASET, "feat")
VOICES = ["S", "M", "T", "N", "L", "A"]
FORMATS = ["karuta", "anaume", "odaibun", "normal"]
ODAI_FEATS = ["has_blank", "has_karuta_mark", "has_text", "photo", "anime", "game", "manga", "text_graphic", "painting", "illustration"]
SEEDS = [0, 1, 2]
FEAT_SETS = ["ruri_fmt", "ruri_fmt_img"]


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


def load_train(labels: str) -> tuple:
    """学習用の集合と教師のラベル，文と画像の埋め込みを読み込む．

    Args:
        labels (str): ex014，{モデル}/{方法}，{発話の型の方法}+{画像外の知識の方法}，または {モデル}/{方法}@extra
            （@extra は ex013 の学習用のボケから重複と検証データのお題を除いた約2.6万件を同じ方法のラベルで加える）．

    Returns:
        tuple[pl.DataFrame, dict]: 列 odaiId, format, voice, outside を持つ表と，ruri・img の埋め込み．
    """
    extra = labels.endswith("@extra")
    labels = labels.replace("@extra", "")
    df = load_aug(labels)
    feats = {"ruri": np.load(os.path.join(FEAT, "aug_ruri310.npy")), "img": np.load(os.path.join(FEAT, "aug_image.npy"))}
    if extra:
        model, variant = labels.split("/")
        ex = pl.read_csv(os.path.join(OUTPUT_DIR, f"teacher_{model}", variant, "extra.csv"))
        fmt = pl.read_parquet(os.path.join(EX13_DATASET, "train_pool.parquet")).with_row_index("train_index").select(
            pl.col("train_index").cast(pl.Int64), "format")
        ex = ex.join(fmt, on="train_index", how="left", maintain_order="left")
        idx = ex["train_index"].to_numpy()
        df = pl.concat([df.select("odaiId", "format", "voice", "outside"), ex.select("odaiId", "format", "voice", "outside")])
        for k, n in [("ruri", "train_ruri310"), ("img", "train_image")]:
            feats[k] = np.concatenate([feats[k], np.load(os.path.join(EX13_DATASET, "emb", f"{n}.npy"), mmap_mode="r")[idx]])
    return df, feats


def load_aug(labels: str) -> pl.DataFrame:
    """水増し用の集合と教師のラベルを読み込む．

    Args:
        labels (str): ex014，{モデル}/{方法}，または {発話の型の方法}+{画像外の知識の方法}．

    Returns:
        pl.DataFrame: 列 odaiId, format, voice, outside を持つ表（aug_pool.csv と同じ順）．
    """
    pool = pl.read_csv(os.path.join(EX14_DATASET, "aug_pool.csv")).select("pool_index", "odaiId", "format")
    if labels == "ex014":
        lab = pl.read_csv(os.path.join(EX14_OUTPUT, "teacher_q32_v3", "aug.csv"), schema_overrides={"knowledge": pl.String})
        lab = lab.with_columns((pl.col("knowledge") == "O").alias("outside"))
    else:
        parts = labels.split("+")
        read = lambda x: pl.read_csv(os.path.join(OUTPUT_DIR, f"teacher_{x.split('/')[0]}", x.split("/")[1], "aug.csv"))  # noqa: E731
        lab = read(parts[0])
        if len(parts) == 2:
            lab = lab.with_columns(read(parts[1])["outside"])
    assert lab.height == pool.height
    return pl.concat([pool, lab.select("voice", "outside")], how="horizontal")


def fit(X: np.ndarray, yv: np.ndarray, yo: np.ndarray, Xe: np.ndarray, seed: int, epochs: int = 30) -> tuple:
    """MLP を学習し（1割を検証用にし誤差が最小のエポックを選ぶ），検証データの確率を返す．

    Args:
        X (np.ndarray): 学習用の特徴．
        yv (np.ndarray): 発話の型の番号．
        yo (np.ndarray): 画像外の知識（0/1）．
        Xe (np.ndarray): 検証データの特徴．
        seed (int): 乱数のシード．
        epochs (int): エポック数．

    Returns:
        tuple[np.ndarray, np.ndarray]: 発話の型の確率 (N, 6) と画像外の知識の確率 (N,)．
    """
    set_seed(seed)
    perm = np.random.default_rng(seed).permutation(len(X))
    va, tr = perm[: len(X) // 10], perm[len(X) // 10:]
    T = lambda a: torch.tensor(a, device="cuda")  # noqa: E731
    cnt = np.bincount(yv, minlength=len(VOICES)).astype(float)
    w = T(np.where(cnt > 0, cnt.sum() / (len(VOICES) * np.maximum(cnt, 1)), 0.0).astype(np.float32))
    pw = T(np.float32((len(yo) - yo.sum()) / max(yo.sum(), 1)))
    body = torch.nn.Sequential(torch.nn.LayerNorm(X.shape[1]), torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.3))
    model = torch.nn.ModuleDict({"body": body, "voice": torch.nn.Linear(512, len(VOICES)), "outside": torch.nn.Linear(512, 1)}).cuda()

    def forward(x: torch.Tensor) -> tuple:
        """順伝播．

        Args:
            x (torch.Tensor): 入力．

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 発話の型と画像外の知識のロジット．
        """
        h = model["body"](x)
        return model["voice"](h), model["outside"](h)[:, 0]

    def loss_fn(x: torch.Tensor, v: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """誤差の和を求める．

        Args:
            x (torch.Tensor): 入力．
            v (torch.Tensor): 発話の型の番号．
            o (torch.Tensor): 画像外の知識．

        Returns:
            torch.Tensor: スカラーの誤差．
        """
        lv, lo = forward(x)
        return torch.nn.functional.cross_entropy(lv, v, weight=w) + torch.nn.functional.binary_cross_entropy_with_logits(lo, o, pos_weight=pw)

    Xt, Xv = T(X[tr].astype(np.float32)), T(X[va].astype(np.float32))
    vt, vv, ot, ov = T(yv[tr]), T(yv[va]), T(yo[tr].astype(np.float32)), T(yo[va].astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state = 1e9, None
    for _ in range(epochs):
        model.train()
        for s in torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed)).split(256):
            loss = loss_fn(Xt[s], vt[s], ot[s])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lv = loss_fn(Xv, vv, ov).item()
        if lv < best:
            best, state = lv, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        lv, lo = forward(T(Xe.astype(np.float32)))
    return torch.softmax(lv, 1).cpu().numpy(), torch.sigmoid(lo).cpu().numpy()


def to_pred(ev: pl.DataFrame, pv: np.ndarray, po: np.ndarray) -> pl.DataFrame:
    """確率から予測の表を作る（穴埋め・お題文の発話の型は回答とする）．

    Args:
        ev (pl.DataFrame): 検証データ．
        pv (np.ndarray): 発話の型の確率．
        po (np.ndarray): 画像外の知識の確率．

    Returns:
        pl.DataFrame: 列 no, voice, outside, p_outside の表．
    """
    voice = [VOICES[i] for i in pv.argmax(1)]
    voice = ["A" if f in ("anaume", "odaibun") else v for f, v in zip(ev["format"], voice)]
    return pl.DataFrame({"no": ev["no"], "voice": voice, "outside": po > 0.5, "p_outside": po})


def stack(ev: pl.DataFrame, pv: np.ndarray, po: np.ndarray, extra: np.ndarray) -> pl.DataFrame:
    """生徒の確率を目視のラベルで補正する（調整用と最終評価用を入れ替えたロジスティック回帰）．

    Args:
        ev (pl.DataFrame): 検証データ．
        pv (np.ndarray): 発話の型の確率．
        po (np.ndarray): 画像外の知識の確率．
        extra (np.ndarray): お題の特徴．

    Returns:
        pl.DataFrame: 予測の表．
    """
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([pv, po[:, None], extra], 1)
    yv = np.array([VOICES.index(v) for v in ev["voice"]])
    yo = ev["outside"].to_numpy().astype(int)
    ov, oo = np.zeros((len(ev), len(VOICES))), np.zeros(len(ev))
    for tr_split in ["dev", "test"]:
        tr = (ev["split"] == tr_split).to_numpy()
        te = ~tr
        clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(X[tr], yv[tr])
        full = np.zeros((te.sum(), len(VOICES)))
        full[:, clf.classes_] = clf.predict_proba(X[te])
        ov[te] = full
        clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(X[tr], yo[tr])
        oo[te] = clf.predict_proba(X[te])[:, 1]
    return to_pred(ev, ov, oo)


def main() -> None:
    """指定したラベルと全ての入力の組み合わせで学習・予測し，補正も行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", default=["ex014"])
    args = parser.parse_args()
    ev = load_eval()
    ev_feats = {"ruri": np.load(os.path.join(FEAT, "eval_ruri310.npy")), "img": np.load(os.path.join(FEAT, "eval_image.npy"))}
    ev_ofeat = odai_feats(ev)
    for labels in args.labels:
        tr, tr_feats = load_train(labels)
        ofeat = odai_feats(tr)
        yv = np.array([VOICES.index(v) for v in tr["voice"]])
        yo = tr["outside"].to_numpy().astype(int)
        for fs in FEAT_SETS:
            X = np.concatenate([tr_feats["ruri"], ofeat] + ([tr_feats["img"]] if "img" in fs else []), 1)
            Xe = np.concatenate([ev_feats["ruri"], ev_ofeat] + ([ev_feats["img"]] if "img" in fs else []), 1)
            for seed in tqdm(SEEDS, desc=f"{labels} {fs}", leave=False):
                pv, po = fit(X, yv, yo, Xe, seed)
                tag = labels.replace("/", "-").replace("+", "_").replace("@", "_")
                for name, pred in [(f"{tag}__{fs}", to_pred(ev, pv, po)), (f"{tag}__{fs}__stack", stack(ev, pv, po, ev_ofeat))]:
                    d = os.path.join(OUTPUT_DIR, "student", name, str(seed))
                    os.makedirs(d, exist_ok=True)
                    pred.write_csv(os.path.join(d, "eval.csv"))
        print(labels, "voice", np.bincount(yv, minlength=6).tolist(), "outside", int(yo.sum()), flush=True)


if __name__ == "__main__":
    main()
