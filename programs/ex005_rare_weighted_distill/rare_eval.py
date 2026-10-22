"""少ない区分（ゲーム，CG，絵画，カルタ）の評価用データを作り，生徒を評価する．

評価用データ640枚には少ない区分の画像が数枚ずつしかないため，以下の手順で区分ごとに50枚，計200枚を追加で抽出する．
1. 収集（mine.py）でラベルを付けなかった候補の画像について，mine.py と同じ MLP で各区分の確率を求める．
2. 区分ごとに確率の高い順に並べ，1〜2,000位から一様に50枚を抽出する（その区分の画像と紛らわしい画像の両方を含む）．
3. 目視で媒体とカルタの有無のラベルを付ける（rare_eval_labels.txt．書式は「お題ID 形式 媒体」で記号は ex004 の data.py と同じ）．
4. 各生徒（MLP と微調整）の予測を求め，区分ごとの適合率・再現率・F1 を求める．

この評価用データは，モデルの確率で絞り込んだ画像からなるため，区分の出現率は実際より高い．値は手法間の比較にのみ用いる．

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/rare_eval.py --make
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/rare_eval.py --evaluate
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, plot_image_grid  # noqa: E402
from data import DATASET_DIR, FORMATS, MEDIA, RARE  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_PATH = os.path.join(THIS_DIR, "rare_eval_labels.txt")
IDS_PATH = os.path.join(DATASET_DIR, "rare_eval_ids.csv")
PER_CLASS, TOP_RANK = 50, 2000
CLASSES = RARE["media"] + RARE["format"]
MEDIA_CODE = {"P": "photo", "A": "anime", "G": "game", "C": "cg", "M": "manga", "T": "text_graphic", "E": "painting",
              "I": "illustration"}


def _ex005_module(name: str):
    """ex005 のモジュールをファイルから読み込む（他の実験の同名のモジュールとの衝突を避ける）．

    Args:
        name (str): モジュール名（ファイル名から拡張子を除いたもの）．

    Returns:
        module: 読み込んだモジュール．
    """
    import importlib.util

    if f"ex005_{name}" in sys.modules:
        return sys.modules[f"ex005_{name}"]
    spec = importlib.util.spec_from_file_location(f"ex005_{name}", os.path.join(THIS_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ex005_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def make() -> None:
    """評価用の画像を抽出し，目視のラベル付け用の一覧画像を作る．

    Returns:
        None
    """
    import mine

    ids = mine.sample_candidates()
    X = mine.candidate_embeddings(ids)
    scores = mine.rare_scores("T_v2_J6M7_pool", X)
    picked = set(pl.read_csv(os.path.join(DATASET_DIR, "teacher_labels", "T_v2_J6M7_rare.csv"))["fileId"].to_list())
    rng = np.random.default_rng(0)
    chosen, rows = set(), []
    for c in CLASSES:
        order = [ids[i] for i in np.argsort(-scores[c]) if ids[i] not in picked and ids[i] not in chosen][:TOP_RANK]
        sel = rng.choice(len(order), PER_CLASS, replace=False)
        for i in sorted(sel):
            chosen.add(order[i])
            rows.append({"fileId": order[i], "pool": c, "rank": int(i) + 1})
    df = pl.DataFrame(rows)
    df.write_csv(IDS_PATH)
    sheets = os.path.join(DATASET_DIR, "rare_eval_sheets")
    os.makedirs(sheets, exist_ok=True)
    for s in range(0, len(df), 20):
        part = df[s:s + 20]
        plot_image_grid(part["fileId"].to_list(), [f"{k:02d}:{f}" for k, f in enumerate(part["fileId"])],
                        f"rare eval sheet {s // 20}", os.path.join(sheets, f"sheet_{s // 20:02d}.png"), ncols=5)
    print(f"{len(df)} 枚を抽出した")


def load_rare_labels() -> pl.DataFrame:
    """目視のラベルを読み込む．

    Returns:
        pl.DataFrame: 列 fileId, pool, karuta（bool）と媒体の各項目（bool）を持つ表．
    """
    ids = pl.read_csv(IDS_PATH)
    recs = []
    for line in open(LABEL_PATH, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        fid, f, m = line.split()
        r = {"fileId": int(fid), "karuta": f == "K"}
        r.update({v: (k in m) for k, v in MEDIA_CODE.items()})
        recs.append(r)
    return ids.join(pl.DataFrame(recs), on="fileId", maintain_order="left")


def predict_mlp(run_dir: str, X: np.ndarray) -> dict:
    """MLP の生徒を，保存した条件で学習し直して予測する（MLP は重みを保存していないため）．

    Args:
        run_dir (str): 生徒の結果のディレクトリ（meta.json と metrics.json を読む）．
        X (np.ndarray): 予測する画像の埋め込み．

    Returns:
        dict: 出力層ごとの確率．
    """
    students = _ex005_module("students")
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    hp = json.load(open(os.path.join(run_dir, "metrics.json")))["selected_hp"]
    X_tr, y_tr, X_va, y_va, _ = students.load_xy(meta["teacher"], meta["use_rare"])
    out = students.fit_mlp(hp, X_tr, y_tr, X_va, y_va, {"tr": X_tr, "va": X_va, "rare": X}, meta["seed"], meta["weighted"])
    if meta.get("tune"):
        th = students.tune_thresholds(out["va"], y_va)
        out["rare"]["media"] = out["rare"]["media"] / th["media"] * 0.5  # しきい値を0.5に揃えるための尺度の変換
    return out["rare"]


@torch.no_grad()
def predict_finetune(run_dir: str, ids: list) -> dict:
    """微調整した生徒の best_model.pth で予測する．

    Args:
        run_dir (str): 生徒の結果のディレクトリ．
        ids (list[int]): お題IDのリスト．

    Returns:
        dict: 出力層ごとの確率．
    """
    T = _ex005_module("train")
    from data import ATTRIBUTES, ThreeAxisDataset

    model = T.load_model(T.SigLIP2BaseThreeHead, os.path.join(run_dir, "best_model.pth")).cuda()
    dummy = {"format": np.zeros(len(ids), int), "attr": np.zeros((len(ids), len(ATTRIBUTES)), np.float32),
             "media": np.zeros((len(ids), len(MEDIA)), np.float32)}
    loader = torch.utils.data.DataLoader(ThreeAxisDataset(ids, dummy, T.SigLIP2BaseThreeHead.transforms()[1]),
                                         batch_size=128, num_workers=16)
    probs, _ = T.collect_probs(model, loader)
    return probs


def scores_of(probs: dict, labels: pl.DataFrame) -> dict:
    """区分ごとの適合率・再現率・F1 を求める．

    Args:
        probs (dict): 出力層ごとの確率．
        labels (pl.DataFrame): load_rare_labels の戻り値．

    Returns:
        dict: 区分の名前をキー，指標の辞書を値とする辞書．
    """
    from sklearn.metrics import precision_recall_fscore_support

    out = {}
    for c in CLASSES:
        pred = probs["format"].argmax(1) == FORMATS.index(c) if c == "karuta" else probs["media"][:, MEDIA.index(c)] >= 0.5
        y = labels["karuta" if c == "karuta" else c].to_numpy()
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        out[c] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(y.sum())}
    out["macro_f1"] = float(np.mean([out[c]["f1"] for c in CLASSES]))
    return out


def evaluate_all(kinds: tuple = ("student_mlp", "finetune_siglip2_base")) -> None:
    """全ての生徒を少ない区分の評価用データで評価し，結果を保存する．

    Args:
        kinds (tuple[str]): 評価する手法の種類．

    Returns:
        None
    """
    import mine

    labels = load_rare_labels()
    cand = mine.sample_candidates()
    index = {f: i for i, f in enumerate(cand)}
    X = mine.candidate_embeddings(cand)[[index[f] for f in labels["fileId"]]]
    results = []
    run_dirs = sorted(d for k in kinds for d in glob.glob(os.path.join(OUTPUT_DIR, k, "*", "*"))
                      if os.path.exists(os.path.join(d, "metrics.json")))
    for run_dir in run_dirs:
        out_path = os.path.join(run_dir, "metrics_rare_eval.json")
        if not os.path.exists(out_path):
            probs = predict_mlp(run_dir, X) if "student_mlp" in run_dir else \
                predict_finetune(run_dir, labels["fileId"].to_list())
            json.dump(scores_of(probs, labels), open(out_path, "w"), indent=2)
        m = json.load(open(out_path))
        results.append({"method": run_dir.split(os.sep)[-3], "setting": run_dir.split(os.sep)[-2],
                        "seed": run_dir.split(os.sep)[-1], "macro_f1": m["macro_f1"],
                        **{f"{c}_f1": m[c]["f1"] for c in CLASSES}, **{f"{c}_recall": m[c]["recall"] for c in CLASSES},
                        **{f"{c}_precision": m[c]["precision"] for c in CLASSES}})
    df = pl.DataFrame(results)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    df.write_csv(os.path.join(OUTPUT_DIR, "figures", "rare_eval_all.csv"))
    print(df.group_by("method", "setting").agg(pl.col("^.*_f1$").mean()).sort("macro_f1", descending=True))


def main() -> None:
    """コマンドライン引数に従って抽出または評価を行う．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--kinds", nargs="+", default=["student_mlp", "finetune_siglip2_base"])
    parser.add_argument("--teachers", nargs="*", default=[])
    args = parser.parse_args()
    if args.teachers:
        evaluate_teachers(args.teachers)
    if args.make:
        make()
    if args.evaluate:
        evaluate_all(tuple(args.kinds))



def evaluate_teachers(names: list) -> None:
    """教師の構成を少ない区分の評価用データで評価し，結果を保存する．

    Args:
        names (list[str]): combine.TEACHERS の構成の名前のリスト．

    Returns:
        None
    """
    combine = _ex005_module("combine")
    labels = load_rare_labels()
    ids = labels["fileId"].to_list()
    rows = []
    for name in names:
        df = combine.build(combine.TEACHERS[name], "rare_eval", ids)
        probs = {"format": np.eye(len(FORMATS))[[FORMATS.index(f) for f in df["format"]]],
                 "media": df.select(MEDIA).to_numpy().astype(float)}
        m = scores_of(probs, labels)
        rows.append({"teacher": name, "macro_f1": m["macro_f1"], **{f"{c}_f1": m[c]["f1"] for c in CLASSES},
                     **{f"{c}_recall": m[c]["recall"] for c in CLASSES},
                     **{f"{c}_precision": m[c]["precision"] for c in CLASSES}})
    out = pl.DataFrame(rows)
    out.write_csv(os.path.join(OUTPUT_DIR, "figures", "rare_eval_teachers.csv"))
    print(out)


if __name__ == "__main__":
    main()
