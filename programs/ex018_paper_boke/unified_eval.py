"""論文（paper_boke/）のため，ex014〜ex017 の主な方法の予測を，同じ正解（検証データ620件の合意ラベル）で評価し直す．

各実験の報告（report_021〜024）では，その時点の目視ラベルで評価していたため，正解が実験ごとに少しずつ異なる．
論文では比較をそろえるため，全ての方法を ex016 の合意ラベル（consensus_labels.txt）で評価する．
生徒の seed が複数ある場合は，seed ごとに評価して平均する．最終評価用の F1 の95%信頼区間（ブートストラップ1,000回）は seed 0 で求める．

出力：outputs/ex018_paper_boke/unified.csv（列 key, group, name, split, voice6_f1, voice3_f1, outside_f1, outside_prec, outside_rec，
outside_lo, outside_hi, voice6_lo, voice6_hi）

実行例:
    .venv_annotate/bin/python programs/ex018_paper_boke/unified_eval.py
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
OUT = os.path.join(PROJECT_DIR, "outputs")
OUTPUT_DIR = os.path.join(OUT, "ex018_paper_boke")
VOICES = ["S", "M", "T", "N", "L", "A"]
VOICE3 = {"S": "C", "M": "C", "T": "X", "N": "X", "L": "X", "A": "A"}


def _load(name: str, path: str):
    """ファイルからモジュールを別名で読み込む．

    Args:
        name (str): モジュール名．
        path (str): ファイルのパス．

    Returns:
        module: 読み込んだモジュール．
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D16 = _load("ex016_data_for_paper", os.path.join(PROJECT_DIR, "programs", "ex016_boke_small_vlm", "data.py"))
_own = sys.modules.get("data")
sys.modules["data"] = _load("ex013_data_for_paper", os.path.join(PROJECT_DIR, "programs", "ex013_boke_labels", "data.py"))
R13 = _load("ex013_rules_for_paper", os.path.join(PROJECT_DIR, "programs", "ex013_boke_labels", "rules.py"))
if _own is not None:
    sys.modules["data"] = _own

# 方法の一覧：キー → (群, 表示名, eval.csv のパターン（glob．seed のディレクトリを含む）)
METHODS = {
    "rules": ("規則", "規則（形式と文末・括弧の手がかり）", None),
    "t32_v3": ("教師", "32B 第3版（4段階を順に問う）", "ex014_boke_labels_v2/teacher_q32_v3/eval.csv"),
    "t30_v3": ("教師", "30B-A3B 第3版", "ex014_boke_labels_v2/teacher_q30a3_v3/eval.csv"),
    "t32_one": ("教師", "32B one（定義書の手順で6区分を選ぶ）", "ex015_boke_labels_v3/teacher_q32/one/eval.csv"),
    "t32_one_cap": ("教師", "32B one_cap（＋キャプション）", "ex015_boke_labels_v3/teacher_q32/one_cap/eval.csv"),
    "t32_kt": ("教師", "32B one_kt_cap（＋知識の種類）", "ex015_boke_labels_v3/teacher_q32/one_kt_cap/eval.csv"),
    "t32_dec": ("教師", "32B dec（問いを分解）", "ex015_boke_labels_v3/teacher_q32/dec/eval.csv"),
    "t32_dec_fs": ("教師", "32B dec_fs（分解＋例示）", "ex015_boke_labels_v3/teacher_q32/dec_fs/eval.csv"),
    "s14_w": ("文の生徒", "文の生徒（水増し＋重み）", "ex014_boke_labels_v2/student/aug_all_w__ruri_fmt/*/eval.csv"),
    "s14_w_stack": ("文の生徒", "文の生徒（水増し＋重み＋補正）", "ex014_boke_labels_v2/student/aug_all_w__ruri_fmt__stack/*/eval.csv"),
    "s15_extra": ("文の生徒", "文の生徒（32B one_kt_cap，3.5万件）", "ex015_boke_labels_v3/student/q32-one_kt_cap_extra__ruri_fmt/*/eval.csv"),
    "zs_q4": ("ゼロショット", "Qwen3-VL-4B", "ex016_boke_small_vlm/zeroshot/q4/eval.csv"),
    "zs_q8": ("ゼロショット", "Qwen3-VL-8B", "ex016_boke_small_vlm/zeroshot/q8/eval.csv"),
    "zs_iv4": ("ゼロショット", "InternVL3.5-4B", "ex016_boke_small_vlm/zeroshot/iv4/eval.csv"),
    "zs_q32t": ("ゼロショット", "Qwen3-VL-32B-Thinking", "ex016_boke_small_vlm/zeroshot/q32t/eval.csv"),
    "ft1_q4": ("VLM 段階1", "Qwen3-VL-4B（教師ラベル）", "ex016_boke_small_vlm/ft/q4/teacher_lr0.0001_ep2_bs16_r16/0/eval.csv"),
    "ft1_iv2": ("VLM 段階1", "InternVL3.5-2B（教師ラベル）", "ex016_boke_small_vlm/ft/iv2/teacher_lr0.0001_ep1_bs16_r16/0/eval.csv"),
    "hum_q4": ("VLM 目視のみ", "Qwen3-VL-4B（目視ラベルのみ）", "ex016_boke_small_vlm/ft/q4/human_only/0/eval.csv"),
    "hum_q2": ("VLM 目視のみ", "Qwen3-VL-2B（目視ラベルのみ）", "ex016_boke_small_vlm/ft/q2/human_only/0/eval.csv"),
    "ft2_q2": ("VLM 段階2", "Qwen3-VL-2B", "ex016_boke_small_vlm/ft/q2/teacher_lr0.0001_ep1_bs16_r16__stage2/0/eval.csv"),
    "ft2_q4": ("VLM 段階2", "Qwen3-VL-4B", "ex016_boke_small_vlm/ft/q4/teacher_lr0.0001_ep2_bs16_r16__stage2/0/eval.csv"),
    "ft2_q8": ("VLM 段階2", "Qwen3-VL-8B", "ex017_boke_outside/ft/q8/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2/0/eval.csv"),
    "ft2_iv1": ("VLM 段階2", "InternVL3.5-1B", "ex016_boke_small_vlm/ft/iv1/teacher_lr0.0001_ep1_bs16_r16__stage2/0/eval.csv"),
    "ft2_iv2": ("VLM 段階2", "InternVL3.5-2B", "ex016_boke_small_vlm/ft/iv2/teacher_lr0.0001_ep1_bs16_r16__stage2/0/eval.csv"),
    "ft2_iv4": ("VLM 段階2", "InternVL3.5-4B", "ex016_boke_small_vlm/ft/iv4/teacher_lr0.0001_ep2_bs16_r16__stage2/0/eval.csv"),
    "ft2_iv8": ("VLM 段階2", "InternVL3.5-8B", "ex017_boke_outside/ft/iv8/ex015_wo3_lr0.0001_ep1_bs16_r16__stage2/0/eval.csv"),
    "ft2_lfm16": ("VLM 段階2", "LFM2-VL-1.6B", "ex016_boke_small_vlm/ft/lfm16/teacher_lr0.0001_ep1_bs16_r16__stage2/0/eval.csv"),
}
ENSEMBLES = {"ens_q4q8": ("VLM 段階2", "Qwen3-VL-4B＋8B（確率の平均）", ["ft2_q4", "ft2_q8"])}


def read_pred(path: str, ev: pl.DataFrame) -> pl.DataFrame:
    """予測の表を読み，列 voice, p_outside（なければ outside か knowledge から作る）を ev の順にそろえる．

    Args:
        path (str): eval.csv のパス．
        ev (pl.DataFrame): 検証データ．

    Returns:
        pl.DataFrame: 列 no, voice, p_outside の表．
    """
    d = pl.read_csv(path, schema_overrides={"knowledge": pl.String}, infer_schema_length=None)
    if "p_outside" not in d.columns:
        o = d["outside"].cast(pl.Boolean) if "outside" in d.columns else (d["knowledge"] == "O")
        d = d.with_columns(o.cast(pl.Float64).alias("p_outside"))
    return ev.select("no").join(d.select("no", "voice", "p_outside"), on="no", how="left", maintain_order="left")


def scores(ev: pl.DataFrame, p: pl.DataFrame, mask: np.ndarray, ci: bool) -> dict:
    """指標を求める．

    Args:
        ev (pl.DataFrame): 検証データ（合意ラベル）．
        p (pl.DataFrame): 予測．
        mask (np.ndarray): 評価する項目．
        ci (bool): 95%信頼区間を求めるか．

    Returns:
        dict: 指標．
    """
    y, yo = ev["voice"].to_numpy()[mask], ev["outside"].to_numpy()[mask]
    q, qo = p["voice"].fill_null("S").to_numpy()[mask], p["p_outside"].fill_null(0).to_numpy()[mask] > 0.5
    v6 = lambda a, b: f1_score(a, b, labels=VOICES, average="macro", zero_division=0)  # noqa: E731
    r = {"voice6_f1": v6(y, q),
         "voice3_f1": f1_score([VOICE3[v] for v in y], [VOICE3[v] for v in q], labels=["C", "X", "A"], average="macro", zero_division=0),
         "outside_f1": f1_score(yo, qo, zero_division=0), "outside_prec": precision_score(yo, qo, zero_division=0),
         "outside_rec": recall_score(yo, qo, zero_division=0)}
    if ci:
        rng = np.random.default_rng(0)
        bo, bv = [], []
        for _ in range(1000):
            b = rng.choice(len(y), len(y))
            bo.append(f1_score(yo[b], qo[b], zero_division=0))
            bv.append(v6(y[b], q[b]))
        r["outside_lo"], r["outside_hi"] = np.percentile(bo, [2.5, 97.5])
        r["voice6_lo"], r["voice6_hi"] = np.percentile(bv, [2.5, 97.5])
    return r


def main() -> None:
    """全ての方法を評価して保存する．

    Returns:
        None
    """
    ev = D16.load_eval()
    preds = {"rules": [pl.DataFrame({"no": ev["no"], "voice": [R13.rule_voice(t, f) for t, f in zip(ev["text"], ev["format"])],
                                     "p_outside": [None] * ev.height}, schema_overrides={"p_outside": pl.Float64})]}
    for key, (_, _, pat) in METHODS.items():
        if pat is not None:
            files = sorted(glob.glob(os.path.join(OUT, pat)))
            assert files, pat
            preds[key] = [read_pred(f, ev) for f in files]
    for key, (_, _, members) in ENSEMBLES.items():
        base = preds[members[0]][0]
        preds[key] = [base.with_columns(pl.Series("p_outside", np.mean([preds[m][0]["p_outside"].to_numpy() for m in members], 0)))]
    rows = []
    names = {**{k: v[:2] for k, v in METHODS.items()}, **{k: v[:2] for k, v in ENSEMBLES.items()}}
    for key, ps in preds.items():
        for split in ["dev", "test"]:
            m = (ev["split"] == split).to_numpy()
            per_seed = [scores(ev, p, m, ci=(split == "test" and i == 0)) for i, p in enumerate(ps)]
            r = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0] if not k.endswith(("_lo", "_hi"))}
            r.update({k: v for k, v in per_seed[0].items() if k.endswith(("_lo", "_hi"))})
            rows.append({"key": key, "group": names[key][0], "name": names[key][1], "split": split, "n_seed": len(ps), **r})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(os.path.join(OUTPUT_DIR, "unified.csv"))
    w = df.pivot(on="split", index=["key", "name"], values=["voice6_f1", "voice3_f1", "outside_f1"])
    with pl.Config(tbl_rows=60, tbl_cols=12, fmt_str_lengths=40, tbl_width_chars=220, float_precision=3):
        print(w)


if __name__ == "__main__":
    main()
