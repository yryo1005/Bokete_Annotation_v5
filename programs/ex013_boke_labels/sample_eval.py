"""ボケのラベルの検証データの追加の抽出（ex012 の140件に加える）．

ex012 と同じ7層から，ex012 で使っていないお題を層ごとに N_ODAI 件抽出し，評価の上位3件と無作為の2件のボケを取る．
出力：datasets/ex013_boke_labels/eval_bokes.csv（ex012 の140件を no 0〜139，追加を 140〜 として結合したもの），
      sheets/sheet_XX.png（追加分の一覧）

実行例:
    .venv_annotate/bin/python programs/ex013_boke_labels/sample_eval.py
"""
import importlib.util
import os

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
spec = importlib.util.spec_from_file_location("ex012_sample", os.path.join(PROJECT_DIR, "programs", "ex012_boke_scheme", "sample.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
OUT = os.path.join(PROJECT_DIR, "datasets", "ex013_boke_labels")
N_ODAI = 8


def main() -> None:
    """追加の抽出と一覧の作成を行う．

    Returns:
        None
    """
    old = pl.read_csv(os.path.join(PROJECT_DIR, "datasets", "ex012_boke_scheme", "sample_bokes.csv"))
    used = set(old["odaiId"].to_list())
    ann = pl.scan_parquet(S.ANN).filter(pl.col("odaiId") == pl.col("rep_odaiId")).collect()
    rng = np.random.default_rng(1)
    recs, sheets, no = [], [], old.height
    for st, cond in S.strata().items():
        picked = 0
        for fid in rng.permutation(ann.filter(cond)["odaiId"].to_numpy()):
            if picked == N_ODAI:
                break
            if int(fid) in used:
                continue
            b = S.load_bokes([int(fid)]).filter(pl.col("text").fill_null("").str.len_chars() > 0)
            if b.height < S.N_MIN:
                continue
            b = b.sort("rateSum", descending=True)
            parts = [("top", b.head(3)), ("random", b.slice(3).sample(n=2, seed=int(fid)))]
            bokes = []
            for kind, part in parts:
                for t, r in zip(part["text"], part["rateSum"]):
                    recs.append({"no": no, "stratum": st, "odaiId": int(fid), "n_bokes": b.height, "kind": kind,
                                 "rateSum": int(r), "text": t})
                    bokes.append((no, kind, int(r), t))
                    no += 1
            sheets.append((int(fid), st, bokes))
            picked += 1
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    pl.concat([old, pl.DataFrame(recs)]).write_csv(os.path.join(OUT, "eval_bokes.csv"))
    for k in range(0, len(sheets), 4):
        S.plot_sheet(sheets[k:k + 4], os.path.join(OUT, "sheets", f"sheet_{k // 4:02d}.png"))
    print(len(recs), len(sheets))


if __name__ == "__main__":
    main()
