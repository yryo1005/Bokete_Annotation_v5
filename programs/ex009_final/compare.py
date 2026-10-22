"""最終版の構成を選ぶため，生徒のモデルと後処理の組み合わせを全ての評価用データで比較する．

モデル（確率の出所．seed ごとに求め，3 seed で平均する）：
- mlp：SigLIP2-so400m の埋め込みに対する MLP（ex008 の最良の条件）
- ft_b224，ft_b384：SigLIP2-B の微調整（224，384 画素）
- ens_*：上記の確率の平均（同じ seed どうし）
- *_x3：同じモデルの3 seed の確率の平均
- ens_mlpx3+b224*：モデルごとに seed の確率を平均してから，モデルの間で平均したもの
- *_T11：教師 T_v11（確認の質問を O_v11_en に替えた教師）のラベルで学習したもの

後処理（順に積み重ねる）：
- raw：確率から予測（形式は最大の確率，属性・媒体は0.5以上．媒体が空なら最大の項目．形式から属性を導く）
- veto：ボケの書き出しによるカルタの取り消し（postprocess.py）
- verify：お題文と予測した画像に Qwen の確認の質問（O_v10_en）を行い，偽ならお題文以外で確率の最も高い形式にする
- verify_t{t}：通常と予測した画像のうち，お題文の確率が t 以上のものも確認の質問の対象とし，真ならお題文にする
  （確認の質問でお題文の誤りを除けるため，お題文の再現率を上げる目的）

評価：通常の評価用データ（dev/test），少ない区分の評価用データ（dev/test），実データの目視ラベル
（無作為160枚，層別220枚，逆確率重みの推定）．確認の質問の対象となる画像の割合（実データでの推定）も求める．

出力：outputs/ex009_final/compare/summary.csv，per_item.json，fig_compare.png（全ての組み合わせ），
fig_models.png（後処理 verify_noveto でのモデルの比較），fig_posts.png（最終のモデルでの後処理の比較）

実行例:
    .venv_annotate/bin/python programs/ex009_final/compare.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, evaluate  # noqa: E402
from data import (ATTRIBUTES, EX8_DIR, FORMATS, MEDIA, _load, check_ids, load_eval_labels, load_rare_labels,  # noqa: E402
                  load_real_labels)

AN = _load("ex008_analyze_for_final", os.path.join(EX8_DIR, "analyze.py"))
TV = _load("ex008_teacher_for_final", os.path.join(EX8_DIR, "teacher_v4.py"))
ST = _load("ex008_students_for_final", os.path.join(EX8_DIR, "students.py"))
import postprocess as PP  # noqa: E402
OUT = os.path.join(OUTPUT_DIR, "compare")
SOURCES = {"mlp": "mlp/T_v10oJXk__base+rare", "ft_b224": "finetune/b224/T_v10oJXk__base+rare",
           "ft_b384": "finetune/b384/T_v10oJXk__base+rare",
           "mlp_T11": "mlp/T_v11__base+rare", "ft_b224_T11": "finetune/b224/T_v11__base+rare"}
ENSEMBLES = {"ens_mlp+b224": ["mlp", "ft_b224"], "ens_mlp+b384": ["mlp", "ft_b384"], "ens_b224+b384": ["ft_b224", "ft_b384"],
             "ens_all": ["mlp", "ft_b224", "ft_b384"]}
# 3 seed のモデルの確率の平均（本番で3つのモデルを使う場合．seed 0 の行として記録する）
SEED_ENSEMBLES = {"ft_b224_x3": "ft_b224", "ft_b384_x3": "ft_b384", "mlp_x3": "mlp"}
# モデルごとに seed の確率を平均してから，モデルの間で平均する組み合わせ（seed 0 の行として記録する）
MULTI = {"ens_mlpx3+b224": {"mlp": [0, 1, 2], "ft_b224": [0]}, "ens_mlpx3+b224x3": {"mlp": [0, 1, 2], "ft_b224": [0, 1, 2]},
         "ens_mlpx3+b224_T11": {"mlp_T11": [0, 1, 2], "ft_b224_T11": [0]}}
POSTS = PP.POSTS


def load_probs(source: str, seed: int, set_name: str) -> dict:
    """モデルの確率を読み込む（組み合わせは平均する）．

    Args:
        source (str): SOURCES または ENSEMBLES のキー．
        seed (int): seed．
        set_name (str): "eval"，"rare"，"check"．

    Returns:
        dict | None: 出力層ごとの確率．ファイルがなければ None．
    """
    if source in MULTI:
        if seed != 0:
            return None
        per_model = [load_probs(f"_avg:{m}:{','.join(map(str, sds))}", 0, set_name) for m, sds in MULTI[source].items()]
        if any(p is None for p in per_model):
            return None
        return {k: np.mean([q[k] for q in per_model], axis=0) for k in per_model[0]}
    if source.startswith("_avg:"):
        _, m, sds = source.split(":")
        parts = [(m, int(x)) for x in sds.split(",")]
    elif source in SEED_ENSEMBLES:
        parts = [(SEED_ENSEMBLES[source], s) for s in [0, 1, 2]] if seed == 0 else [(None, None)]
    else:
        parts = [(p, seed) for p in ENSEMBLES.get(source, [source])]
    probs = []
    for p, sd in parts:
        if p is None:
            return None
        path = os.path.join(OUTPUT_DIR, SOURCES[p], str(sd), f"probs_{set_name}.npz")
        if not os.path.exists(path):
            return None
        probs.append(dict(np.load(path)))
    return {k: np.mean([q[k] for q in probs], axis=0) for k in probs[0]}


def side_info(set_name: str, ids: list) -> dict:
    """後処理に用いる情報（確認の質問の回答，ボケの書き出しの統計）を読み込む．

    Args:
        set_name (str): "eval"，"rare"，"check"．
        ids (list[int]): お題IDのリスト．

    Returns:
        dict: キー verify（確認の質問の回答，bool の配列），max_rate（最も多い書き出しのかなの割合），n_kana．
    """
    ans = {}
    for comp in ["O_v10_en", "O_v11_en"]:
        if set_name == "rare":
            rare = load_rare_labels()
            o = pl.concat([TV._comp(comp, s, rare.filter(pl.col("set") == s)["fileId"].to_list())
                           for s in ["rare_eval", "rare_eval_v2"]])
            o = rare.select("fileId").join(o.unique("fileId"), on="fileId", how="left", maintain_order="left")
        else:
            o = TV._comp(comp, {"eval": "eval", "check": "real_check_v4"}[set_name], ids)
        ans[comp] = o["is_poster_prompt"].fill_null(True).to_numpy().astype(bool)
    stats = TV.RULES.boke_kana_stats(ids)
    return {"verify": ans["O_v10_en"], "verify11": ans["O_v11_en"],
            "max_rate": PP.kana_max_rate(stats["kana_counts"].fill_null("").to_list(), stats["n_kana"].fill_null(0).to_list()),
            "n_kana": stats["n_kana"].fill_null(0).to_numpy()}


def apply_post(probs: dict, info: dict, post: str) -> tuple:
    """後処理を適用して予測の表を作る．

    Args:
        probs (dict): 出力層ごとの確率．
        info (dict): side_info の戻り値．
        post (str): POSTS のいずれか．

    Returns:
        tuple[pl.DataFrame, np.ndarray]: 予測の表と，確認の質問の対象となった画像を表す bool の配列．
    """
    answer = info["verify11"] if post.startswith("verify11") else info["verify"]
    fmt, _, asked = PP.decide_format(probs["format"], info["n_kana"], info["max_rate"], answer, post)
    p2 = dict(probs)
    onehot = np.zeros_like(probs["format"])
    onehot[np.arange(len(fmt)), fmt] = 1.0
    p2["format"] = onehot
    return ST.to_frame(p2), asked


def collect() -> tuple:
    """全てのモデルと後処理の組み合わせの指標を求める．

    Returns:
        tuple[pl.DataFrame, dict]: 1行1（モデル，後処理，seed）の指標の表と，項目ごとの F1．
    """
    labels = load_eval_labels()
    rare = load_rare_labels()
    rnd, strat = load_real_labels("random"), load_real_labels("strat")
    ids = {"eval": labels["fileId"].to_list(), "rare": rare["fileId"].to_list(), "check": check_ids()}
    info = {k: side_info(k, v) for k, v in ids.items()}
    n_pop = int(strat["weight"].sum())
    w_check = pl.DataFrame({"fileId": ids["check"]}).join(
        pl.concat([rnd.select("fileId", pl.lit(n_pop / len(rnd)).alias("w")), strat.select("fileId", pl.col("weight").alias("w"))]),
        on="fileId", how="left", maintain_order="left")["w"].to_numpy()
    rows, items = [], {}
    for source in list(SOURCES) + list(ENSEMBLES) + list(SEED_ENSEMBLES) + list(MULTI):
        for seed in [0, 1, 2]:
            P = {k: load_probs(source, seed, k) for k in ids}
            if any(v is None for v in P.values()):
                continue
            for post in POSTS:
                pred = {k: apply_post(P[k], info[k], post) for k in ids}
                r = {"source": source, "post": post, "seed": seed}
                for sp in ["dev", "test"]:
                    m = labels["split"] == sp
                    e = evaluate(labels.filter(m), pred["eval"][0].filter(m))
                    r[f"{sp}_overall"], r[f"{sp}_robust"] = e["overall_macro_f1"], e["overall_robust_f1"]
                    r[f"{sp}_odaibun"] = e["format_per_class"]["odaibun"]["f1"]
                rs = TV.rare_scores(pred["rare"][0], rare)
                r["rare_dev"], r["rare_test"] = rs["rare_dev"]["macro_f1"], rs["rare_test"]["macro_f1"]
                pc = pl.concat([pl.DataFrame({"fileId": ids["check"]}), pred["check"][0]], how="horizontal")
                cs = AN.check_scores(pc, rnd, strat, n_pop)
                for k, v in cs.items():
                    r[f"{k}_overall"], r[f"{k}_all_correct"] = v["overall_macro_f1"], v["all_correct"]
                    r[f"{k}_odaibun"] = v["per_item"]["odaibun"]
                r["ask_rate"] = float(np.average(pred["check"][1], weights=w_check))
                r["n_vetoed"] = int(sum(PP.decide_format(P[k]["format"], info[k]["n_kana"], info[k]["max_rate"],
                                                         info[k]["verify"], post)[1].sum() for k in ids))
                r["odaibun_rate_real"] = float(np.average(pred["check"][0]["format"].to_numpy() == "odaibun", weights=w_check))
                rows.append(r)
                items.setdefault(f"{source}|{post}", []).append({k: v["per_item"] for k, v in cs.items()})
    return pl.DataFrame(rows), items


def plot(g: pl.DataFrame) -> None:
    """モデルと後処理の組み合わせの比較の図を作る．

    Args:
        g (pl.DataFrame): 組み合わせごとの平均の表．

    Returns:
        None
    """
    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    sources = g["source"].unique(maintain_order=True).to_list()
    posts = [p for p in POSTS if p in g["post"].to_list()]
    for ax, (k, title) in zip(axes, [("dev_overall", "通常 dev：3軸の平均"), ("rare_dev", "少ない区分 dev"),
                                      ("pooled_ipw_all_correct", "実データ 推定：全一致の割合")]):
        for j, post in enumerate(posts):
            vals = [g.filter((pl.col("source") == s) & (pl.col("post") == post))[k].to_list() for s in sources]
            ax.barh(np.arange(len(sources)) + (j - len(posts) / 2 + 0.5) * 0.13,
                    [v[0] if v else 0 for v in vals], 0.13, label=post)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        lo = float(g[k].min()) - 0.02
        ax.set_xlim(lo, min(1.0, float(g[k].max()) + 0.01))
    axes[0].set_yticks(np.arange(len(sources)), sources, fontsize=8)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_compare.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


FINAL_SOURCE, FINAL_POST = "ens_mlpx3+b224", "verify11_noveto"


def plot_report(g: pl.DataFrame) -> None:
    """レポート用に，モデルの比較と後処理の比較の図を作る．

    Args:
        g (pl.DataFrame): 組み合わせごとの平均の表．

    Returns:
        None
    """
    m = g.filter(pl.col("post") == FINAL_POST).sort("select_score")
    keys = [("dev_overall", "通常 dev：3軸の平均"), ("rare_dev", "少ない区分 dev"), ("test_overall", "通常 test：3軸の平均"),
            ("rare_test", "少ない区分 test"), ("pooled_ipw_all_correct", "実データ 推定：全一致")]
    fig, axes = plt.subplots(1, len(keys), figsize=(17, 5), sharey=True)
    y = np.arange(len(m))
    for ax, (k, title) in zip(axes, keys):
        colors = ["tab:red" if s == FINAL_SOURCE else "tab:blue" for s in m["source"]]
        ax.barh(y, m[k], color=colors)
        ax.set_xlim(float(m[k].min()) - 0.02, float(m[k].max()) + 0.01)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, m["source"].to_list(), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_models.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    q = g.filter(pl.col("source") == FINAL_SOURCE)
    keys = [("select_score", "選択の指標（dev）"), ("test_odaibun", "通常 test：お題文の F1"),
            ("pooled_ipw_odaibun", "実データ 推定：お題文の F1"), ("ask_rate", "確認の質問の対象の割合")]
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 3.5), sharey=True)
    y = np.arange(len(q))
    for ax, (k, title) in zip(axes, keys):
        ax.barh(y, q[k], color=["tab:red" if p_ == FINAL_POST else "tab:blue" for p_ in q["post"]])
        lo = 0.0 if k == "ask_rate" else float(q[k].min()) - 0.02
        ax.set_xlim(lo, float(q[k].max()) + 0.01)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, q["post"].to_list(), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_posts.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def main() -> None:
    """比較を行い，表と図を保存する．

    Returns:
        None
    """
    os.makedirs(OUT, exist_ok=True)
    df, items = collect()
    df.write_csv(os.path.join(OUT, "per_seed.csv"))
    cols = [c for c in df.columns if c not in ("source", "post", "seed")]
    g = df.group_by("source", "post", maintain_order=True).agg(
        [pl.len().alias("n")] + [pl.col(c).mean() for c in cols]).with_columns(
        ((pl.col("dev_overall") + pl.col("rare_dev")) / 2).alias("select_score"))
    g.write_csv(os.path.join(OUT, "summary.csv"))
    json.dump({k: {s: {i: float(np.mean([x[s][i] for x in v])) for i in v[0][s]} for s in v[0]} for k, v in items.items()},
              open(os.path.join(OUT, "per_item.json"), "w"), indent=1, ensure_ascii=False)
    plot(g)
    plot_report(g)
    with pl.Config(tbl_rows=100, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(g.sort("select_score", descending=True).select(
            "source", "post", "n", "select_score", "dev_overall", "rare_dev", "test_overall", "test_robust", "rare_test",
            "test_odaibun", "random_all_correct", "strat_uniform_all_correct", "pooled_ipw_all_correct",
            "pooled_ipw_odaibun", "ask_rate"))


if __name__ == "__main__":
    main()
