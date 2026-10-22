"""少ない区分（ゲーム，CG，絵画，カルタ）の画像を，Qwen がその区分と判定した枚数が N 枚以上になるまで集める．

枚数は基本の4万枚の教師ラベルと追加分の合計で数える（N = N_TARGET = 1,000）．

1. 候補の抽出：評価用・教師ラベル用・ex001 の弱ラベル用の画像を除いた全お題から，無作為に N_CANDIDATES 枚を抽出する．
2. 候補の埋め込み：SigLIP2-so400m の埋め込みを計算する．
3. 候補の順位付け：基本の4万枚の教師ラベルで MLP（ex004 の FourHeadMLP と同じ構成の3出力版）を学習し，
   各区分の確率を候補について求める．
4. ラベル付け：区分ごとに確率の高い順に ROUND 枚ずつ選び（既に選んだ画像は除く），教師の部品を適用して教師ラベルを作る．
   Qwen がその区分と判定した画像（形式がカルタ，または媒体が真）が N 枚に達するか，区分あたり MAX_PER_CLASS 枚を
   ラベル付けするまで繰り返す．

出力：datasets/ex005_rare_weighted_distill/candidates/（候補と各回の ID），embeddings/，
teacher_labels/{教師}_rare.csv（追加の教師ラベル）と mining_log.json（各回の集計）．

実行例:
    .venv_annotate/bin/python programs/ex005_rare_weighted_distill/mine.py --teacher T_best
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combine import TEACHERS, build, components_of  # noqa: E402
from data import DATASET_DIR, FORMATS, MEDIA, PROJECT_DIR, RARE, label_arrays, load_eval_labels, load_pool  # noqa: E402
from data import load_teacher_labels  # noqa: E402

N_TARGET = 1000
N_CANDIDATES = 300_000
ROUND = 1000
MAX_PER_CLASS = 8000
EMBED_MODEL = "google/siglip2-so400m-patch14-384"
CAND_DIR = os.path.join(DATASET_DIR, "candidates")


def _load_module(name: str, path: str):
    """モジュール名の衝突を避けてファイルからモジュールを読み込む．

    Args:
        name (str): モジュールに付ける名前．
        path (str): ファイルのパス．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_candidates(seed: int = 0) -> list:
    """候補の画像を抽出する（既にあれば読み込む）．

    Args:
        seed (int): 乱数のシード．

    Returns:
        list[int]: 候補のお題IDのリスト．
    """
    path = os.path.join(CAND_DIR, "cand.csv")
    if os.path.exists(path):
        return pl.read_csv(path)["fileId"].to_list()
    ex001 = _load_module("ex001_data_m", os.path.join(PROJECT_DIR, "programs", "ex001_odai_image_type", "data.py"))
    meta = ex001.load_odai_meta()
    used = set(load_eval_labels()["fileId"].to_list()) | set(load_pool()["fileId"].to_list())
    used |= set(pl.read_csv(os.path.join(PROJECT_DIR, "datasets", "ex001_odai_image_type", "weak_train.csv"))["fileId"])
    ids = meta.filter(~pl.col("fileId").is_in(list(used))).sample(n=N_CANDIDATES, seed=seed + 200)["fileId"].to_list()
    os.makedirs(CAND_DIR, exist_ok=True)
    pl.DataFrame({"fileId": ids}).write_csv(path)
    return ids


def candidate_embeddings(ids: list) -> np.ndarray:
    """候補の SigLIP2-so400m の埋め込みを計算する（既にあれば読み込む）．

    Args:
        ids (list[int]): 候補のお題IDのリスト．

    Returns:
        np.ndarray: 形状 (N, 1152) の float32 の配列．
    """
    path = os.path.join(DATASET_DIR, "embeddings", "cand_so400m.npy")
    if not os.path.exists(path):
        feats = _load_module("ex002_features", os.path.join(PROJECT_DIR, "programs", "ex002_odai_type_distill",
                                                            "features.py"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, feats.extract_embeddings(EMBED_MODEL, ids))
    return np.load(path).astype(np.float32)


def rare_scores(teacher: str, X_cand: np.ndarray, seed: int = 0) -> dict:
    """基本の教師ラベルで MLP を学習し，候補の各少ない区分の確率を求める．

    Args:
        teacher (str): 基本の4万枚の教師ラベルの名前．
        X_cand (np.ndarray): 形状 (N, D) の候補の埋め込み．
        seed (int): 乱数のシード．

    Returns:
        dict[str, np.ndarray]: 区分の名前をキー，形状 (N,) の確率を値とする辞書．
    """
    students = _load_module("ex005_students", os.path.join(os.path.dirname(os.path.abspath(__file__)), "students.py"))
    fit_mlp, load_features = students.fit_mlp, students.load_features

    lab = load_teacher_labels(teacher)
    index = {fid: i for i, fid in enumerate(load_pool()["fileId"].to_list())}
    X = load_features("siglip2-so400m-patch14-384", "pool")[[index[f] for f in lab["fileId"]]]
    tr, va = (lab["split"] == "train").to_numpy(), (lab["split"] == "val").to_numpy()
    y = label_arrays(lab)
    probs = fit_mlp({"lr": 1e-3, "wd": 1e-2}, X[tr], {k: v[tr] for k, v in y.items()}, X[va],
                    {k: v[va] for k, v in y.items()}, {"cand": X_cand, "tr": X[tr], "va": X[va]}, seed)["cand"]
    out = {c: probs["media"][:, MEDIA.index(c)] for c in RARE["media"]}
    out.update({c: probs["format"][:, FORMATS.index(c)] for c in RARE["format"]})
    return out


def confirmed(df: pl.DataFrame, cls: str) -> pl.Series:
    """教師ラベルがその少ない区分であるかを返す．

    Args:
        df (pl.DataFrame): 教師ラベルの表．
        cls (str): 区分の名前．

    Returns:
        pl.Series: 真偽値の列．
    """
    return (df["format"] == cls) if cls in RARE["format"] else df[cls]


def main() -> None:
    """少ない区分の画像を集め，追加の教師ラベルを保存する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    args = parser.parse_args()
    import teacher as T

    ids = sample_candidates()
    X = candidate_embeddings(ids)
    torch.cuda.empty_cache()
    base_name = f"{args.teacher}_pool"
    scores = rare_scores(base_name, X)
    order = {c: np.argsort(-s) for c, s in scores.items()}
    cfg = TEACHERS[args.teacher]
    comps = sorted(components_of(cfg))
    llm = None
    picked, labeled, log = set(), {c: 0 for c in scores}, []
    labels_all = []
    base = pl.read_csv(os.path.join(DATASET_DIR, "teacher_labels", f"{base_name}.csv"))
    base_count = {c: int(confirmed(base, c).sum()) for c in scores}
    count = dict(base_count)
    print("基本の4万枚での枚数:", base_count, flush=True)
    rnd = 0
    while True:
        need = [c for c in scores if count[c] < N_TARGET and labeled[c] < MAX_PER_CLASS]
        if not need:
            break
        batch = []
        for c in need:
            new = [ids[i] for i in order[c] if ids[i] not in picked][:ROUND // len(need)]
            labeled[c] += len(new)
            batch += new
            picked.update(new)
        set_name = f"rare_r{rnd:02d}"
        pl.DataFrame({"fileId": batch}).write_csv(os.path.join(CAND_DIR, f"{set_name}.csv"))
        for comp in comps:
            if os.path.exists(T.output_path(comp, set_name)):
                continue
            if llm is None:
                from prompts import COMPONENTS
                llm = T.build_llm(max(COMPONENTS[x]["max_pixels"] for x in comps))
            df, sec = T.run_component(llm, comp, batch, f"{comp}:{set_name}")
            os.makedirs(os.path.dirname(T.output_path(comp, set_name)), exist_ok=True)
            df.write_csv(T.output_path(comp, set_name))
        lab = pl.concat([pl.DataFrame({"fileId": batch}), build(cfg, set_name, batch)], how="horizontal")
        labels_all.append(lab)
        allv = pl.concat(labels_all)
        count = {c: base_count[c] + int(confirmed(allv, c).sum()) for c in scores}
        log.append({"round": rnd, "n_batch": len(batch), "need": need, "confirmed": count, "labeled_per_class": dict(labeled)})
        print(f"round {rnd}: {len(batch)} 枚にラベル付け，確認された枚数 {count}", flush=True)
        rnd += 1
    out = pl.concat(labels_all)
    out.write_csv(os.path.join(DATASET_DIR, "teacher_labels", f"{args.teacher}_rare.csv"))
    json.dump({"base_count": base_count, "rounds": log}, open(os.path.join(DATASET_DIR, "mining_log.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"追加の教師ラベル {len(out)} 枚を保存した")


if __name__ == "__main__":
    main()
