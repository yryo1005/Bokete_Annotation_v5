"""教師ラベルで生徒を学習し，検証データで評価する（ボケ約9,250万件に適用できる速さの方法）．

生徒の種類（--kinds）：
- mlp_{文の埋め込み}_{特徴}：埋め込みに対する多出力の MLP．特徴は text（ボケの文の埋め込みのみ），
  text_fmt（＋お題の形式と属性），text_fmt_img（＋お題画像の SigLIP2-so400m の埋め込み）
- logreg_{文の埋め込み}_text_fmt_img：同じ特徴の項目ごとのロジスティック回帰
- ft_ruri：ruri-v3-310m を「形式：…　キャプション：…　ボケ：…」の文で微調整（多出力）
- human_logreg：教師を使わず，検証データの目視のラベルだけでロジスティック回帰を学習する（調整用で学習して最終評価用を予測し，
  逆も行う2分割の交差）
- stack：教師ラベルで学習した MLP の出力の確率とお題の特徴を入力とし，目視のラベルでロジスティック回帰を学習して補正する（同じ2分割）

出力：outputs/ex013_boke_labels/student/{種類}/{seed}/eval.csv，eval_speed.json，meta.json

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/students.py --teacher q30a3
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".ai", "ai-dev-kit", "src"))
from data import DATASET_DIR, KNOWS, OUTPUT_DIR, VOICES, load_eval, load_odai_labels  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

EMB_DIR = os.path.join(DATASET_DIR, "emb")
FORMATS = ["karuta", "anaume", "odaibun", "normal"]
ODAI_FEATS = ["has_blank", "has_karuta_mark", "has_text", "photo", "anime", "game", "manga", "text_graphic", "painting",
              "illustration"]
SEEDS = [0, 1, 2]
HEADS = {"voice": len(VOICES), "know": len(KNOWS), "complies": 1, "uses_text": 1, "inappropriate": 1}


def teacher_targets(teacher: str) -> pl.DataFrame:
    """学習用のボケと教師ラベル，お題のラベルを結合する．

    Args:
        teacher (str): 教師のモデルの名前．

    Returns:
        pl.DataFrame: 学習用の表（train_pool.parquet の順）．
    """
    pool = pl.read_parquet(os.path.join(DATASET_DIR, "train_pool.parquet"))
    lab = pl.read_csv(os.path.join(OUTPUT_DIR, f"teacher_{teacher}", "train.csv"))
    pool = pl.concat([pool, lab.select("complies", "voice", "knowledge", "uses_image_text", "inappropriate")], how="horizontal")
    return pool.join(load_odai_labels(pool["odaiId"].unique().to_list()).drop("format"), on="odaiId", how="left",
                     maintain_order="left")


def odai_feats(df: pl.DataFrame) -> np.ndarray:
    """お題の形式（one-hot）と属性・媒体の特徴を作る．

    Args:
        df (pl.DataFrame): 列 format と ODAI_FEATS を持つ表．

    Returns:
        np.ndarray: 形状 (N, 14) の特徴．
    """
    f = np.stack([(df["format"] == c).to_numpy() for c in FORMATS], 1)
    return np.concatenate([f, df.select(ODAI_FEATS).to_numpy()], 1).astype(np.float32)


def features(sname: str, df: pl.DataFrame, text_model: str, feat: str) -> np.ndarray:
    """生徒の入力の特徴を作る．

    Args:
        sname (str): "train" または "eval"．
        df (pl.DataFrame): ボケの表．
        text_model (str): 文の埋め込みのモデルの名前．
        feat (str): text，text_fmt，text_fmt_img．

    Returns:
        np.ndarray: 特徴の配列．
    """
    parts = [np.load(os.path.join(EMB_DIR, f"{sname}_{text_model}.npy")).astype(np.float32)]
    if "fmt" in feat:
        parts.append(odai_feats(df))
    if "img" in feat:
        parts.append(np.load(os.path.join(EMB_DIR, f"{sname}_image.npy")).astype(np.float32))
    return np.concatenate(parts, 1)


def targets(df: pl.DataFrame) -> dict:
    """教師ラベルを学習の目標の配列にする．

    Args:
        df (pl.DataFrame): 教師ラベルの列を持つ表．

    Returns:
        dict[str, np.ndarray]: 項目ごとの目標．
    """
    return {"voice": np.array([VOICES.index(v) for v in df["voice"]]), "know": np.array([KNOWS.index(v) for v in df["knowledge"]]),
            "complies": df["complies"].to_numpy().astype(np.float32), "uses_text": df["uses_image_text"].to_numpy().astype(np.float32),
            "inappropriate": df["inappropriate"].to_numpy().astype(np.float32)}


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


def loss_fn(out: dict, y: dict) -> torch.Tensor:
    """項目ごとの交差エントロピーの和を求める．

    Args:
        out (dict): ロジット．
        y (dict): 目標のテンソル．

    Returns:
        torch.Tensor: スカラーの誤差．
    """
    loss = sum(torch.nn.functional.cross_entropy(out[k], y[k]) for k in ["voice", "know"])
    return loss + sum(torch.nn.functional.binary_cross_entropy_with_logits(out[k][:, 0], y[k])
                      for k in ["complies", "uses_text", "inappropriate"])


def predict_frame(out: dict) -> pl.DataFrame:
    """ロジットから予測の表を作る．

    Args:
        out (dict[str, np.ndarray]): 項目ごとのロジット．

    Returns:
        pl.DataFrame: 列 complies, voice, knowledge, uses_image_text, inappropriate の表．
    """
    return pl.DataFrame({"complies": out["complies"][:, 0] > 0, "voice": [VOICES[i] for i in out["voice"].argmax(1)],
                         "knowledge": [KNOWS[i] for i in out["know"].argmax(1)],
                         "uses_image_text": out["uses_text"][:, 0] > 0, "inappropriate": out["inappropriate"][:, 0] > 0})


def fit_mlp(X: np.ndarray, Y: dict, Xe: np.ndarray, seed: int, epochs: int = 30) -> tuple:
    """MLP を学習し（学習用の1割を検証用にし，誤差が最小のエポックを選ぶ），検証データの予測と速度を返す．

    Args:
        X (np.ndarray): 学習用の特徴．
        Y (dict): 学習用の目標．
        Xe (np.ndarray): 検証データの特徴．
        seed (int): 乱数のシード．
        epochs (int): エポック数．

    Returns:
        tuple[pl.DataFrame, float, dict]: 予測，1秒あたりの予測の件数，学習の記録．
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    va, tr = perm[: len(X) // 10], perm[len(X) // 10:]
    dev = "cuda"
    T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    Xt, Xv = T(X[tr]), T(X[va])
    Yt = {k: T(v[tr]) for k, v in Y.items()}
    Yv = {k: T(v[va]) for k, v in Y.items()}
    model = MultiHead(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state, hist = 1e9, None, {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        for s in torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed)).split(512):
            loss = loss_fn(model(Xt[s]), {k: v[s] for k, v in Yt.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lt, lv = loss_fn(model(Xt), Yt).item(), loss_fn(model(Xv), Yv).item()
        hist["train_loss"].append(lt)
        hist["val_loss"].append(lv)
        if lv < best:
            best, state = lv, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        Xe_t = T(Xe)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            out = model(Xe_t)
        torch.cuda.synchronize()
        per_sec = 100 * len(Xe) / (time.perf_counter() - start)
    return predict_frame({k: v.cpu().numpy() for k, v in out.items()}), per_sec, hist


def fit_logreg(X: np.ndarray, Y: dict, Xe: np.ndarray) -> pl.DataFrame:
    """項目ごとのロジスティック回帰を学習して予測する．

    Args:
        X (np.ndarray): 学習用の特徴．
        Y (dict): 学習用の目標．
        Xe (np.ndarray): 検証データの特徴．

    Returns:
        pl.DataFrame: 予測．
    """
    from sklearn.linear_model import LogisticRegression

    out = {}
    for k, y in Y.items():
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X, y.astype(int))
        if k in ("voice", "know"):
            full = np.full((len(Xe), HEADS[k]), -1e9)
            full[:, clf.classes_] = clf.decision_function(Xe)
            out[k] = full
        else:
            out[k] = clf.decision_function(Xe)[:, None]
    return predict_frame(out)


def save(name: str, seed: int, ev: pl.DataFrame, pred: pl.DataFrame, per_sec: float, meta: dict) -> None:
    """予測と速度を保存する．

    Args:
        name (str): 生徒の種類．
        seed (int): seed．
        ev (pl.DataFrame): 検証データ．
        pred (pl.DataFrame): 予測．
        per_sec (float): 1秒あたりの予測の件数（特徴の計算を含まない）．
        meta (dict): 条件と学習の記録．

    Returns:
        None
    """
    d = os.path.join(OUTPUT_DIR, "student", name, str(seed))
    os.makedirs(d, exist_ok=True)
    pl.concat([ev.select("no", "odaiId", "format"), pred], how="horizontal").write_csv(os.path.join(d, "eval.csv"))
    json.dump({"per_sec": per_sec}, open(os.path.join(d, "eval_speed.json"), "w"))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)


def run_embedding_students(teacher: str) -> None:
    """埋め込みに対する MLP とロジスティック回帰を学習・評価する．

    Args:
        teacher (str): 教師のモデルの名前．

    Returns:
        None
    """
    tr = teacher_targets(teacher)
    ev = load_eval()
    Y = targets(tr)
    for tm, feat in itertools.product(["ruri310", "e5base"], ["text", "text_fmt", "text_fmt_img"]):
        X, Xe = features("train", tr, tm, feat), features("eval", ev, tm, feat)
        for seed in SEEDS:
            pred, per_sec, hist = fit_mlp(X, Y, Xe, seed)
            save(f"mlp_{tm}_{feat}", seed, ev, pred, per_sec, {"teacher": teacher, "text_model": tm, "feat": feat, **hist})
        print("mlp", tm, feat, flush=True)
    for tm in ["ruri310"]:
        X, Xe = features("train", tr, tm, "text_fmt_img"), features("eval", ev, tm, "text_fmt_img")
        start = time.perf_counter()
        pred = fit_logreg(X, Y, Xe)
        save(f"logreg_{tm}_text_fmt_img", 0, ev, pred, None, {"teacher": teacher, "fit_sec": time.perf_counter() - start})
        print("logreg", tm, flush=True)


def ft_text(df: pl.DataFrame) -> list:
    """微調整する生徒の入力の文を作る．

    Args:
        df (pl.DataFrame): 列 format, caption, text を持つ表．

    Returns:
        list[str]: 入力の文．
    """
    fj = {"karuta": "カルタ", "anaume": "穴埋め", "odaibun": "お題文", "normal": "通常"}
    return [f"形式：{fj.get(f, f)}　画像：{c or 'なし'}　ボケ：{(t or '').replace(chr(10), ' ')[:150]}"
            for f, c, t in df.select("format", "caption", "text").rows()]


def run_finetune(teacher: str, seeds: list, epochs: int = 2, batch_size: int = 64) -> None:
    """ruri-v3-310m を微調整して評価する．

    Args:
        teacher (str): 教師のモデルの名前．
        seeds (list[int]): seed のリスト．
        epochs (int): エポック数．
        batch_size (int): バッチサイズ．

    Returns:
        None
    """
    from transformers import AutoModel, AutoTokenizer

    tr, ev = teacher_targets(teacher), load_eval()
    Y = targets(tr)
    tok = AutoTokenizer.from_pretrained("cl-nagoya/ruri-v3-310m")
    xs, xe = ft_text(tr), ft_text(ev)
    for seed in seeds:
        if os.path.exists(os.path.join(OUTPUT_DIR, "student", "ft_ruri", str(seed), "eval.csv")):
            continue
        set_seed(seed)
        enc = AutoModel.from_pretrained("cl-nagoya/ruri-v3-310m").cuda()
        heads = torch.nn.ModuleDict({k: torch.nn.Linear(enc.config.hidden_size, n) for k, n in HEADS.items()}).cuda()
        params = list(enc.parameters()) + list(heads.parameters())
        opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)

        def forward(texts: list) -> dict:
            """文の束のロジットを求める．

            Args:
                texts (list[str]): 文のリスト．

            Returns:
                dict[str, torch.Tensor]: ロジット．
            """
            b = tok(texts, padding=True, truncation=True, max_length=160, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            pooled = (h.float() * m).sum(1) / m.sum(1)
            return {k: hd(pooled) for k, hd in heads.items()}

        hist = []
        for _ in range(epochs):
            enc.train()
            perm = np.random.default_rng(seed).permutation(len(xs))
            for s in tqdm(range(0, len(perm), batch_size), desc="ft", leave=False):
                idx = perm[s:s + batch_size]
                y = {k: torch.tensor(v[idx], device="cuda") for k, v in Y.items()}
                loss = loss_fn(forward([xs[i] for i in idx]), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                hist.append(loss.item())
        enc.eval()
        outs = {k: [] for k in HEADS}
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            for s in range(0, len(xe), 256):
                o = forward(xe[s:s + 256])
                for k in HEADS:
                    outs[k].append(o[k].float().cpu().numpy())
            torch.cuda.synchronize()
            per_sec = len(xe) / (time.perf_counter() - start)
        pred = predict_frame({k: np.concatenate(v) for k, v in outs.items()})
        save("ft_ruri", seed, ev, pred, per_sec, {"teacher": teacher, "epochs": epochs, "train_loss": hist[::50]})
        print("ft_ruri", seed, flush=True)
        del enc, heads
        torch.cuda.empty_cache()


def labels_human(ev: pl.DataFrame) -> dict:
    """検証データの目視のラベルを目標の配列にする．

    Args:
        ev (pl.DataFrame): 検証データ．

    Returns:
        dict[str, np.ndarray]: 項目ごとの目標．
    """
    return {"voice": np.array([VOICES.index(v) for v in ev["voice"]]), "know": np.array([KNOWS.index(v) for v in ev["know"]]),
            "complies": (ev["comp"] != "n").to_numpy().astype(np.float32), "uses_text": ev["uses_text"].to_numpy().astype(np.float32),
            "inappropriate": ev["inappropriate"].to_numpy().astype(np.float32)}


def two_fold(ev: pl.DataFrame, Xe: np.ndarray, Yh: dict) -> pl.DataFrame:
    """調整用と最終評価用を入れ替えてロジスティック回帰を学習し，それぞれの予測を作る．

    Args:
        ev (pl.DataFrame): 検証データ（列 split）．
        Xe (np.ndarray): 検証データの特徴．
        Yh (dict): 目視のラベルの目標．

    Returns:
        pl.DataFrame: 検証データの順の予測．
    """
    from sklearn.linear_model import LogisticRegression

    out = {k: np.zeros((len(ev), HEADS[k])) for k in HEADS}
    for tr_split in ["dev", "test"]:
        tr = (ev["split"] == tr_split).to_numpy()
        te = ~tr
        for k, y in Yh.items():
            y_tr = y[tr].astype(int)
            if len(np.unique(y_tr)) < 2:
                out[k][te] = (1.0 if y_tr[0] == 1 else -1.0) if HEADS[k] == 1 else 0.0
                continue
            clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
            clf.fit(Xe[tr], y_tr)
            if HEADS[k] > 1:
                full = np.full((te.sum(), HEADS[k]), -1e9)
                full[:, clf.classes_] = clf.decision_function(Xe[te])
                out[k][te] = full
            else:
                out[k][te] = clf.decision_function(Xe[te])[:, None]
    return predict_frame(out)


def run_human(teacher: str) -> None:
    """目視のラベルだけで学習する方法と，教師の生徒の出力を目視のラベルで補正する方法を評価する．

    Args:
        teacher (str): 教師のモデルの名前（stack の土台の生徒）．

    Returns:
        None
    """
    ev = load_eval()
    Yh = labels_human(ev)
    Xe = features("eval", ev, "ruri310", "text_fmt_img")
    save("human_logreg", 0, ev, two_fold(ev, Xe, Yh), None, {"note": "検証データの目視のラベルで2分割の交差"})
    tr = teacher_targets(teacher)
    X = features("train", tr, "ruri310", "text_fmt_img")
    for seed in SEEDS:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        va, trn = perm[: len(X) // 10], perm[len(X) // 10:]
        T = lambda a: torch.tensor(a, device="cuda")  # noqa: E731
        Y = targets(tr)
        model = MultiHead(X.shape[1]).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        Xt, Xv = T(X[trn]), T(X[va])
        Yt, Yv = {k: T(v[trn]) for k, v in Y.items()}, {k: T(v[va]) for k, v in Y.items()}
        best, state = 1e9, None
        for _ in range(30):
            model.train()
            for s in torch.randperm(len(trn), generator=torch.Generator().manual_seed(seed)).split(512):
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
            o = model(T(Xe))
        probs = np.concatenate([torch.softmax(o["voice"], 1).cpu().numpy(), torch.softmax(o["know"], 1).cpu().numpy(),
                                torch.sigmoid(torch.cat([o[k] for k in ["complies", "uses_text", "inappropriate"]], 1)).cpu().numpy()], 1)
        Xs = np.concatenate([probs, odai_feats(ev)], 1)
        save("stack", seed, ev, two_fold(ev, Xs, Yh), None, {"teacher": teacher, "note": "教師の生徒の確率と形式を目視のラベルで補正"})
    print("human/stack done", flush=True)


def main() -> None:
    """生徒を学習・評価する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="q30a3_v2")
    parser.add_argument("--kinds", nargs="+", default=["embedding", "human", "finetune"])
    args = parser.parse_args()
    if "embedding" in args.kinds:
        run_embedding_students(args.teacher)
    if "human" in args.kinds:
        run_human(args.teacher)
    if "finetune" in args.kinds:
        run_finetune(args.teacher, SEEDS)


if __name__ == "__main__":
    main()
