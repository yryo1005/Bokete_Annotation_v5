"""ラベルの設定（形式・属性・媒体のクラス）の統合・削除の案を，既存の予測の読み替えで比較する（order_010）．

予測は新たに学習せずに既存のものを用いる．
- 教師 T_v8（combine_v8 の PCen_cgand）
- 最良の生徒：SigLIP2-so400m の MLP（T_v8，追加あり・weighted，3 seed）
- 微調整した SigLIP2-B（T_v8，追加あり・weighted，3 seed）
評価用データは通常の640枚（dev/test）と少ない区分の800枚（媒体とカルタのみ）である．
少ない区分の評価用データに対する生徒の予測は，ex005 の rare_eval.py の関数で求めてキャッシュする（MLP は保存した条件で学習し直す）．

案は SCHEMES に定義する．統合は各クラスの論理和（媒体・属性）またはクラスの読み替え（形式）で行い，正解と予測の両方に同じ規則を適用する．
出力は outputs/ex006_label_scheme/ の scheme_scores.csv（案×モデル×集合の指標），per_item.csv（項目ごとの F1），
confusion_media.csv（目視と教師の媒体の対応），図である．

実行例:
    .venv_annotate/bin/python programs/ex006_label_scheme/analyze_schemes.py
"""
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from sklearn.metrics import f1_score, precision_recall_fscore_support

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EX5 = os.path.join(PROJECT_DIR, "programs", "ex005_rare_weighted_distill")
OUT = os.path.join(PROJECT_DIR, "outputs", "ex006_label_scheme")
EX5_OUT = os.path.join(PROJECT_DIR, "outputs", "ex005_rare_weighted_distill")
sys.path.insert(0, EX5)
import combine_v8 as C  # noqa: E402
from data import ATTRIBUTES, FORMATS, MEDIA, load_eval_labels  # noqa: E402

RUNS = {"MLP": "student_mlp/T_v8__base+rare__weighted", "FT": "finetune_siglip2_base/T_v8__base+rare__weighted"}
SEEDS = ["0", "1", "2"]

# 媒体の案：新しいクラス名 → 元のクラスのリスト（論理和）
MEDIA_SCHEMES = {
    "M0_現行8": {m: [m] for m in MEDIA},
    "M1_CG→イラスト": {**{m: [m] for m in MEDIA if m not in ("cg", "illustration")}, "illustration": ["illustration", "cg"]},
    "M2_CG・絵画→イラスト": {**{m: [m] for m in MEDIA if m not in ("cg", "painting", "illustration")},
                         "illustration": ["illustration", "cg", "painting"]},
    "M3_CG→写真": {**{m: [m] for m in MEDIA if m not in ("cg", "photo")}, "photo": ["photo", "cg"]},
    "M4_ゲーム・CG統合": {**{m: [m] for m in MEDIA if m not in ("cg", "game")}, "game_cg": ["game", "cg"]},
    "M5_アニメ・ゲーム統合": {**{m: [m] for m in MEDIA if m not in ("anime", "game")}, "anime_game": ["anime", "game"]},
    "M7_CG削除": {m: [m] for m in MEDIA if m != "cg"},
    "M6_CG→イラスト＋アニメ・ゲーム統合": {"photo": ["photo"], "anime_game": ["anime", "game"], "manga": ["manga"],
                                 "text_graphic": ["text_graphic"], "painting": ["painting"],
                                 "illustration": ["illustration", "cg"]},
}
# 属性の案：新しい項目名 → 元の項目のリスト（論理和）
ATTR_SCHEMES = {
    "A0_現行5": {a: [a] for a in ATTRIBUTES},
    "A1_合成削除": {a: [a] for a in ATTRIBUTES if a != "composite"},
    "A2_合成削除＋文字統合": {"has_blank": ["has_blank"], "has_karuta_mark": ["has_karuta_mark"],
                       "has_text": ["has_prompt_text", "has_other_text"]},
    "A3_合成削除＋お題文削除": {a: [a] for a in ATTRIBUTES if a not in ("composite", "has_prompt_text")},
    "A4_合成削除＋その他文字削除": {a: [a] for a in ATTRIBUTES if a not in ("composite", "has_other_text")},
}
# 形式の案：元のクラス → 新しいクラス
FORMAT_SCHEMES = {
    "F0_現行4": {f: f for f in FORMATS},
    "F1_お題文→通常": {"karuta": "karuta", "anaume": "anaume", "odaibun": "normal", "normal": "normal"},
    "F2_カルタ→穴埋め": {"karuta": "anaume", "anaume": "anaume", "odaibun": "odaibun", "normal": "normal"},
}


def _load(name: str, path: str):
    """ファイルからモジュールを読み込む．

    Args:
        name (str): モジュール名．
        path (str): ファイルのパス．

    Returns:
        module: 読み込んだモジュール．
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def eval_predictions(labels: pl.DataFrame) -> dict:
    """通常の評価用データに対する各モデルの予測を読み込む．

    Args:
        labels (pl.DataFrame): 評価用ラベル（640枚）．

    Returns:
        dict[str, list[pl.DataFrame]]: モデル名をキー，seed ごとの予測（列 format と属性・媒体の各項目）のリストを値とする辞書．
    """
    cols = ["format"] + ATTRIBUTES + MEDIA
    out = {"教師T_v8": [C.build_v8(C.TEACHERS_V8["PCen_cgand"], "eval", labels["fileId"].to_list()).select(cols)]}
    for name, run in RUNS.items():
        out[name] = []
        for s in SEEDS:
            p = pl.read_csv(os.path.join(EX5_OUT, run, s, "predictions.csv"))
            out[name].append(p.select([pl.col(f"pred_{c}").alias(c) for c in cols]))
    return out


def rare_predictions(rare: pl.DataFrame) -> dict:
    """少ない区分の評価用データに対する各モデルの予測を求める（キャッシュする）．

    Args:
        rare (pl.DataFrame): combine_v8.load_rare_eval の戻り値（800枚）．

    Returns:
        dict[str, list[pl.DataFrame]]: モデル名をキー，seed ごとの予測（列 format と媒体の各項目）のリストを値とする辞書．
    """
    os.makedirs(OUT, exist_ok=True)
    cols = ["format"] + MEDIA
    parts = [C.build_v8(C.TEACHERS_V8["PCen_cgand"], s, rare.filter(pl.col("set") == s)["fileId"].to_list()).select(cols)
             for s in ["rare_eval", "rare_eval_v2"]]
    out = {"教師T_v8": [pl.concat(parts)]}
    R = _load("ex005_rare_eval", os.path.join(EX5, "rare_eval.py"))
    S = _load("ex005_students", os.path.join(EX5, "students.py"))
    X = None
    for name, run in RUNS.items():
        out[name] = []
        for s in SEEDS:
            cache = os.path.join(OUT, f"rare_pred_{name}_{s}.parquet")
            if not os.path.exists(cache):
                run_dir = os.path.join(EX5_OUT, run, s)
                if name == "FT":
                    probs = R.predict_finetune(run_dir, rare["fileId"].to_list())
                else:
                    if X is None:
                        cand = pl.read_csv(os.path.join(PROJECT_DIR, "datasets", "ex005_rare_weighted_distill", "candidates",
                                                        "cand.csv"))["fileId"].to_list()
                        idx = {f: i for i, f in enumerate(cand)}
                        E = np.load(os.path.join(PROJECT_DIR, "datasets", "ex005_rare_weighted_distill", "embeddings",
                                                 "cand_so400m.npy"), mmap_mode="r")
                        X = np.asarray(E[[idx[f] for f in rare["fileId"]]], dtype=np.float32)
                    probs = R.predict_mlp(run_dir, X)
                S.to_frame(probs).select(cols).write_parquet(cache)
            out[name].append(pl.read_parquet(cache))
    return out


def remap_bool(df: pl.DataFrame, scheme: dict) -> pl.DataFrame:
    """真偽値の項目を案に従って論理和で統合する．

    Args:
        df (pl.DataFrame): 元の項目を持つ表．
        scheme (dict): 新しい項目名 → 元の項目のリスト．

    Returns:
        pl.DataFrame: 新しい項目の表．
    """
    return pl.DataFrame({k: np.logical_or.reduce([df[c].to_numpy().astype(bool) for c in v]) for k, v in scheme.items()})


def bool_scores(true: pl.DataFrame, pred: pl.DataFrame) -> dict:
    """真偽値の項目ごとの F1 と，正例のある項目のマクロ F1，完全一致率を求める．

    Args:
        true (pl.DataFrame): 正解．
        pred (pl.DataFrame): 予測（同じ列）．

    Returns:
        dict: macro_f1，exact_match，per_item（項目 → (F1, 正例数)）．
    """
    per = {}
    for c in true.columns:
        y, q = true[c].to_numpy(), pred[c].to_numpy()
        if y.sum() == 0:
            continue
        per[c] = (float(f1_score(y, q, zero_division=0)), int(y.sum()))
    return {"macro_f1": float(np.mean([v[0] for v in per.values()])),
            "exact_match": float((true.to_numpy() == pred.to_numpy()).all(axis=1).mean()), "per_item": per}


def format_scores(true: pl.Series, pred: pl.Series, scheme: dict) -> dict:
    """形式の案に従って読み替えた正解率とマクロ F1 を求める．

    Args:
        true (pl.Series): 正解の形式．
        pred (pl.Series): 予測の形式．
        scheme (dict): 元のクラス → 新しいクラス．

    Returns:
        dict: accuracy，macro_f1，per_item（クラス → (F1, 正例数)）．
    """
    t, p = true.replace_strict(scheme).to_numpy(), pred.replace_strict(scheme).to_numpy()
    classes = sorted(set(scheme.values()))
    _, _, f, s = precision_recall_fscore_support(t, p, labels=classes, zero_division=0)
    return {"accuracy": float((t == p).mean()), "macro_f1": float(np.mean(f)),
            "per_item": {c: (float(f[i]), int(s[i])) for i, c in enumerate(classes)}}


def rare_media_scores(true: pl.DataFrame, pred: pl.DataFrame, scheme: dict) -> dict:
    """少ない区分の評価用データで，媒体の案に従って統合した後の少ない区分（ゲーム，CG，絵画とそれを含む統合クラス）の F1．

    Args:
        true (pl.DataFrame): 正解の媒体．
        pred (pl.DataFrame): 予測の媒体．
        scheme (dict): 媒体の案．

    Returns:
        dict: 統合後のクラスのうち元の少ない区分を含むものの F1 と，その平均．
    """
    t, p = remap_bool(true, scheme), remap_bool(pred, scheme)
    targets = [k for k, v in scheme.items() if set(v) & {"game", "cg", "painting"}]
    per = {k: float(f1_score(t[k].to_numpy(), p[k].to_numpy(), zero_division=0)) for k in targets}
    return {"macro_f1": float(np.mean(list(per.values()))), "per_item": per}


def media_confusion(labels: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """目視で媒体が1つの画像について，目視の媒体ごとに予測の媒体の分布を数える．

    Args:
        labels (pl.DataFrame): 正解．
        pred (pl.DataFrame): 予測．

    Returns:
        pl.DataFrame: 行が目視の媒体，列が予測の媒体（複数選択は各々に数える）の件数の表．
    """
    T, P = labels.select(MEDIA).to_numpy(), pred.select(MEDIA).to_numpy()
    rows = []
    for m in MEDIA:
        mask = T[:, MEDIA.index(m)] & (T.sum(1) == 1)
        rows.append({"hand": m, "n": int(mask.sum()), **{q: int(P[mask, j].sum()) for j, q in enumerate(MEDIA)}})
    return pl.DataFrame(rows)


def main() -> None:
    """全ての案を評価して保存する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    labels = load_eval_labels()
    rare = C.load_rare_eval()
    ev, rv = eval_predictions(labels), rare_predictions(rare)
    rows, items = [], []
    for model, preds in ev.items():
        for seed, pred in enumerate(preds):
            for split in ["dev", "test", "all"]:
                m = np.ones(len(labels), bool) if split == "all" else (labels["split"] == split).to_numpy()
                L, P = labels.filter(pl.Series(m)), pred.filter(pl.Series(m))
                for sname, sch in MEDIA_SCHEMES.items():
                    r = bool_scores(remap_bool(L, sch), remap_bool(P, sch))
                    rows.append({"axis": "media", "scheme": sname, "model": model, "seed": seed, "set": split,
                                 "macro_f1": r["macro_f1"], "exact": r["exact_match"]})
                    items += [{"axis": "media", "scheme": sname, "model": model, "seed": seed, "set": split, "item": k,
                               "f1": v[0], "support": v[1]} for k, v in r["per_item"].items()]
                for sname, sch in ATTR_SCHEMES.items():
                    r = bool_scores(remap_bool(L, sch), remap_bool(P, sch))
                    rows.append({"axis": "attr", "scheme": sname, "model": model, "seed": seed, "set": split,
                                 "macro_f1": r["macro_f1"], "exact": r["exact_match"]})
                    items += [{"axis": "attr", "scheme": sname, "model": model, "seed": seed, "set": split, "item": k,
                               "f1": v[0], "support": v[1]} for k, v in r["per_item"].items()]
                for sname, sch in FORMAT_SCHEMES.items():
                    r = format_scores(L["format"], P["format"], sch)
                    rows.append({"axis": "format", "scheme": sname, "model": model, "seed": seed, "set": split,
                                 "macro_f1": r["macro_f1"], "exact": r["accuracy"]})
    for model, preds in rv.items():
        for seed, pred in enumerate(preds):
            for split in ["dev", "test"]:
                m = (rare["split"] == split).to_numpy()
                L, P = rare.filter(pl.Series(m)).select(MEDIA), pred.filter(pl.Series(m)).select(MEDIA)
                for sname, sch in MEDIA_SCHEMES.items():
                    r = rare_media_scores(L, P, sch)
                    rows.append({"axis": "media_rare", "scheme": sname, "model": model, "seed": seed, "set": split,
                                 "macro_f1": r["macro_f1"], "exact": None})
                    items += [{"axis": "media_rare", "scheme": sname, "model": model, "seed": seed, "set": split, "item": k,
                               "f1": v, "support": None} for k, v in r["per_item"].items()]
    scores = pl.DataFrame(rows)
    scores.write_csv(os.path.join(OUT, "scheme_scores.csv"))
    pl.DataFrame(items).write_csv(os.path.join(OUT, "per_item.csv"))
    media_confusion(labels, ev["教師T_v8"][0]).write_csv(os.path.join(OUT, "confusion_media_teacher.csv"))
    media_confusion(labels, ev["MLP"][0]).write_csv(os.path.join(OUT, "confusion_media_mlp.csv"))
    summary = scores.group_by("axis", "scheme", "model", "set").agg(pl.col("macro_f1").mean(), pl.col("macro_f1").std()
                                                                    .alias("std"), pl.col("exact").mean())
    summary.sort("axis", "set", "model", "scheme").write_csv(os.path.join(OUT, "scheme_summary.csv"))
    plot(summary)
    print(summary.filter(pl.col("set").is_in(["dev", "test"])).sort("axis", "model", "scheme", "set"))


def plot(summary: pl.DataFrame) -> None:
    """案ごとのマクロ F1 を軸別に描く．

    Args:
        summary (pl.DataFrame): 案×モデル×集合の平均の表．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    axes_def = [("format", "形式（test，マクロ F1）", "test"), ("attr", "属性（test，マクロ F1）", "test"),
                ("media", "媒体（test，マクロ F1）", "test"), ("media_rare", "媒体：少ない区分の評価用データ（test）", "test")]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, (axis, title, split) in zip(axs, axes_def):
        d = summary.filter((pl.col("axis") == axis) & (pl.col("set") == split))
        schemes = sorted(d["scheme"].unique().to_list())
        models = ["教師T_v8", "MLP", "FT"]
        y = np.arange(len(schemes))
        for i, mdl in enumerate(models):
            dd = {r["scheme"]: (r["macro_f1"], r["std"]) for r in d.filter(pl.col("model") == mdl).iter_rows(named=True)}
            ax.barh(y + (i - 1) * 0.27, [dd.get(s, (0, 0))[0] for s in schemes], 0.27,
                    xerr=[dd.get(s, (0, 0))[1] or 0 for s in schemes], capsize=2, label=mdl)
        ax.set_yticks(y, schemes, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.3, 1.0)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axs[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_schemes.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
