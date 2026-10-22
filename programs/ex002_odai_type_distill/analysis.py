"""ex002 の追加の分析．

1. baseline：ex001 の最良の方法（弱ラベル用10,500枚に Qwen3-VL-8B が付けたラベルで SigLIP2-so400m のロジスティック回帰，C=0.01）を
   ex002 の評価用データ（640枚）で評価する．
2. rule_blankbox：OpenCV の空欄矩形の特徴のみによる規則（白い一様な四角形があれば穴埋め）．
3. cascade：生徒の確信度（形式と画風の最大確率の積）が低い順に一定の割合の画像を教師 VLM の予測に置き換える．
   処理時間は「生徒で全件＋教師で置き換える割合の画像」として見積もる．
4. teacher_agreement：教師ラベル用の画像における教師どうしの一致率．

実行例:
    .venv_annotate/bin/python programs/ex002_odai_type_distill/analysis.py --cascade_students student_mlp/Qwen3-VL-8B-Instruct__siglip2-so400m-patch14-384__nall
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, TOTAL_IMAGES, save_run  # noqa: E402
from data import DATASET_DIR, PROJECT_DIR, load_eval_labels  # noqa: E402
from features import load_features  # noqa: E402

EX001_DATA = os.path.join(PROJECT_DIR, "datasets", "ex001_odai_image_type")
CASCADE_FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
# 埋め込みの計算速度（ex001 の benchmark.py の実測値，枚/秒）
EMBED_SPEED = {"siglip2-so400m-patch14-384": 177.5, "siglip2-base-patch16-224": 881.5, "clip-vit-base-patch32": 1087.4}


def run_baseline() -> None:
    """ex001 の最良の方法を ex002 の評価用データで評価する（seed ごとに学習データの90%を用いる点も ex001 と同じ）．

    Returns:
        None
    """
    labels = load_eval_labels()
    weak = pl.read_csv(os.path.join(EX001_DATA, "weak_train.csv"))
    vl = pl.read_csv(os.path.join(EX001_DATA, "vlm_labels", "Qwen3-VL-8B-Instruct_v2_px200704.csv"))
    X = np.load(os.path.join(EX001_DATA, "embeddings", "siglip2-so400m-patch14-384", "weak.npy")).astype(np.float32)
    X_ev = load_features("siglip2-so400m-patch14-384", "eval")
    assert weak["fileId"].to_list() == vl["fileId"].to_list()
    for seed in [0, 1, 2]:
        idx = np.random.default_rng(seed).permutation(len(X))[: int(len(X) * 0.9)]
        sc = StandardScaler().fit(X[idx])
        preds = []
        for col in ["format", "domain"]:
            clf = LogisticRegression(C=0.01, max_iter=3000, random_state=seed).fit(sc.transform(X[idx]),
                                                                                   vl[col].to_numpy()[idx])
            preds.append(clf.predict(sc.transform(X_ev)))
        meta = {"method": "baseline_ex001", "description": "ex001 の最良の方法（弱ラベル用10,500枚の8Bラベル）",
                "teacher": "Qwen3-VL-8B-Instruct", "features": ["siglip2-so400m-patch14-384"], "C": 0.01,
                "n_train": int(len(idx)), "seed": seed}
        save_run(os.path.join(OUTPUT_DIR, "baseline_ex001", "logreg_C0.01", str(seed)), labels, list(preds[0]),
                 list(preds[1]), meta)


def run_rule_blankbox() -> None:
    """白い一様な四角形が1つ以上あれば穴埋め，なければ normal とする規則を評価する（画風は photo と仮定）．

    Returns:
        None
    """
    labels = load_eval_labels()
    X = load_features("blankbox", "eval")
    fmt = ["anaume" if v > 0 else "normal" for v in X[:, 1]]
    meta = {"method": "rule_blankbox", "description": "OpenCV で検出した白い一様な四角形の有無のみで穴埋めを判定",
            "note": "画風は判定しないため photo と仮定する"}
    save_run(os.path.join(OUTPUT_DIR, "rule_blankbox", "white_box", "0"), labels, fmt, ["photo"] * len(fmt), meta)


def run_cascade(student_run: str, teacher: str, teacher_speed: float) -> None:
    """生徒の確信度の低い画像を教師の予測に置き換えるカスケードを評価する（seed ごと）．

    Args:
        student_run (str): OUTPUT_DIR からの生徒の結果のパス（seed のディレクトリを除く）．
        teacher (str): 教師の短い名前（teacher_{名前}/v2/0 の予測を用いる）．
        teacher_speed (float): 教師の処理速度（枚/秒）．

    Returns:
        None
    """
    labels = load_eval_labels()
    tp = pl.read_csv(os.path.join(OUTPUT_DIR, f"teacher_{teacher}", "v2", "0", "predictions.csv"))
    s_meta = json.load(open(os.path.join(OUTPUT_DIR, student_run, "0", "meta.json")))
    if "features" in s_meta:
        s_speed = min(EMBED_SPEED[f] for f in s_meta["features"] if f in EMBED_SPEED)
    else:  # 微調整したモデルは seed 0 で測った処理速度を用いる
        s_speed = s_meta["throughput"]["images_per_sec"]
    method = os.path.dirname(student_run).replace("student_", "")
    for seed_dir, frac in itertools.product(sorted(glob.glob(os.path.join(OUTPUT_DIR, student_run, "*"))),
                                            CASCADE_FRACTIONS):
        sp = pl.read_csv(os.path.join(seed_dir, "predictions.csv"))
        order = np.argsort(sp["confidence"].to_numpy())
        replace = np.zeros(len(sp), dtype=bool)
        replace[order[: int(round(frac * len(sp)))]] = True
        fmt = np.where(replace, tp["pred_format"].to_numpy(), sp["pred_format"].to_numpy())
        dom = np.where(replace, tp["pred_domain"].to_numpy(), sp["pred_domain"].to_numpy())
        hours = TOTAL_IMAGES / s_speed / 3600 + frac * TOTAL_IMAGES / teacher_speed / 3600
        meta = {"method": "cascade", "student": student_run, "teacher": teacher, "fraction_to_teacher": frac,
                "seed": int(os.path.basename(seed_dir)),
                "throughput": {"images_per_sec": TOTAL_IMAGES / hours / 3600, "est_hours_all": hours}}
        run_dir = os.path.join(OUTPUT_DIR, f"cascade_{method}", f"{os.path.basename(student_run)}__{teacher}_frac{frac}",
                               os.path.basename(seed_dir))
        save_run(run_dir, labels, list(fmt), list(dom), meta)


def run_agreement() -> None:
    """教師ラベル用の画像における教師どうしの7クラスの一致率を求めて保存する．

    Returns:
        None
    """
    labs = {}
    for p in sorted(glob.glob(os.path.join(DATASET_DIR, "teacher_labels", "*.csv"))):
        df = pl.read_csv(p)
        labs[os.path.basename(p)[:-4]] = [d if f == "normal" else f for f, d in zip(df["format"], df["domain"])]
    out = {f"{a} | {b}": float(np.mean(np.array(labs[a]) == np.array(labs[b])))
           for a, b in itertools.combinations(labs, 2)}
    dist = {k: pl.Series(v).value_counts(normalize=True).sort("proportion", descending=True).to_dicts()
            for k, v in labs.items()}
    d = os.path.join(OUTPUT_DIR, "teacher_agreement")
    os.makedirs(d, exist_ok=True)
    json.dump({"pairwise_flat_agreement": out, "label_distribution": dist}, open(os.path.join(d, "agreement.json"), "w"),
              indent=2, ensure_ascii=False)
    for k, v in out.items():
        print(f"{k}: {v:.3f}")


def run_common_errors(n_show: int = 20) -> None:
    """強い3手法（8B 教師，SigLIP2-so400m の線形分類器，微調整 SigLIP2-B，いずれも seed 0）が
    そろって目視ラベルと異なる7クラスを予測した画像を抽出し，一覧と図を保存する．

    Args:
        n_show (int): 図に示す画像の数．

    Returns:
        None
    """
    runs = {"teacher_8B": "teacher_Qwen3-VL-8B-Instruct/v2/0",
            "logreg_so400m": "student_logreg/Qwen3-VL-8B-Instruct__siglip2-so400m-patch14-384__nall/0",
            "ft_siglip2_b": "finetune_siglip2_base/Qwen3-VL-8B-Instruct_lr3e-05_ep5/0"}
    base = None
    for name, r in runs.items():
        p = pl.read_csv(os.path.join(OUTPUT_DIR, r, "predictions.csv")).select("fileId", "pool", "true_flat",
                                                                              pl.col("pred_flat").alias(name))
        base = p if base is None else base.join(p.select("fileId", name), on="fileId")
    wrong = [pl.col(n) != pl.col("true_flat") for n in runs]
    all_wrong = base.filter(wrong[0] & wrong[1] & wrong[2])
    agree = all_wrong.filter((pl.col("teacher_8B") == pl.col("logreg_so400m")) & (pl.col("logreg_so400m") == pl.col("ft_siglip2_b")))
    d = os.path.join(OUTPUT_DIR, "common_errors")
    os.makedirs(d, exist_ok=True)
    all_wrong.write_csv(os.path.join(d, "all_three_wrong.csv"))
    pairs = agree.group_by("true_flat", "teacher_8B").len().sort("len", descending=True)
    summary = {"n_eval": len(base), "n_all_three_wrong": len(all_wrong), "n_all_three_wrong_same_prediction": len(agree),
               "pairs_true_to_pred": pairs.to_dicts(),
               "n_any_wrong": int(base.filter(wrong[0] | wrong[1] | wrong[2]).height)}
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs_true_to_pred"}, ensure_ascii=False))
    print(pairs)
    show = agree.sample(n=min(n_show, len(agree)), seed=0)
    titles = [f"id={r['fileId']}\nhand: {r['true_flat']} / 3 methods: {r['teacher_8B']}" for r in show.iter_rows(named=True)]
    plot_image_grid(show["fileId"].to_list(), titles,
                    "images where all three strong methods agree with each other but not with the hand label",
                    os.path.join(OUTPUT_DIR, "figures", "fig_common_errors.png"))


def plot_image_grid(ids: list, titles: list, suptitle: str, path: str, ncols: int = 5) -> None:
    """画像を正方形の余白に収めて格子状に並べ，各画像に題を付けて保存する．

    Args:
        ids (list[int]): お題IDのリスト．
        titles (list[str]): 各画像の題．
        suptitle (str): 図全体の題．
        path (str): 保存先のパス．
        ncols (int): 列数．

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image as _Image

    from data import image_path

    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.0 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, fid, title in zip(np.ravel(axes), ids, titles):
        im = _Image.open(image_path(fid)).convert("RGB")
        im.thumbnail((400, 400))
        canvas = _Image.new("RGB", (400, 400), "white")
        canvas.paste(im, ((400 - im.width) // 2, (400 - im.height) // 2))
        ax.imshow(canvas)
        ax.set_title(title, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if nrows > 1 else 0.9))
    fig.suptitle(suptitle)
    fig.savefig(path, bbox_inches="tight", dpi=90)
    plt.close(fig)


def run_teacher_disagreement(n_show: int = 15) -> None:
    """教師 VLM の予測が分かれた評価用画像を，目視ラベルと各教師の予測とともに図示する．

    Qwen3-VL-8B が正しく，他の教師の少なくとも1つが誤った画像から無作為に選ぶ．

    Args:
        n_show (int): 図に示す画像の数．

    Returns:
        None
    """
    teachers = {"8B": "Qwen3-VL-8B-Instruct", "32B": "Qwen3-VL-32B-Instruct-FP8", "30B-A3B": "Qwen3-VL-30B-A3B-Instruct-FP8",
                "InternVL3.5": "InternVL3_5-8B", "MiniCPM-V4.5": "MiniCPM-V-4_5"}
    base = None
    for short, t in teachers.items():
        p = pl.read_csv(os.path.join(OUTPUT_DIR, f"teacher_{t}", "v2", "0", "predictions.csv"))
        p = p.select("fileId", "true_flat", pl.col("pred_flat").alias(short))
        base = p if base is None else base.join(p.select("fileId", short), on="fileId")
    others = [s for s in teachers if s != "8B"]
    cand = base.filter((pl.col("8B") == pl.col("true_flat")) &
                       pl.any_horizontal([pl.col(s) != pl.col("true_flat") for s in others]))
    show = cand.sample(n=min(n_show, len(cand)), seed=0)
    titles = []
    for r in show.iter_rows(named=True):
        wrong = ", ".join(f"{s}:{r[s]}" for s in others if r[s] != r["true_flat"])
        titles.append(f"id={r['fileId']} hand/8B: {r['true_flat']}\n{wrong}")
    plot_image_grid(show["fileId"].to_list(), titles, "teacher disagreements (Qwen3-VL-8B correct; others listed only when wrong)",
                    os.path.join(OUTPUT_DIR, "figures", "fig_teacher_disagreement.png"))
    print(f"8B が正しく他の教師が誤った画像: {len(cand)} 枚")


def main() -> None:
    """コマンドライン引数で指定した分析を実行する．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["baseline", "rule", "agreement"])
    parser.add_argument("--cascade_students", nargs="*", default=[])
    parser.add_argument("--cascade_teacher", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--cascade_teacher_speed", type=float, default=25.2)
    args = parser.parse_args()
    if "baseline" in args.tasks:
        run_baseline()
    if "rule" in args.tasks:
        run_rule_blankbox()
    if "agreement" in args.tasks:
        run_agreement()
    if "common_errors" in args.tasks:
        run_common_errors()
    if "teacher_disagreement" in args.tasks:
        run_teacher_disagreement()
    for s in args.cascade_students:
        run_cascade(s, args.cascade_teacher, args.cascade_teacher_speed)


if __name__ == "__main__":
    main()
