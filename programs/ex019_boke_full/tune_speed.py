"""パッケージのボケの推論（stage_boke_predict）の設定ごとの速度を，同じ1シャードで測る．

出力先の一覧（boke/list）を設定ごとの作業用ディレクトリに複写し，予測の段階だけを実行して speed.json を集める．

出力：outputs/ex019_boke_full/tune2/{設定}/speed.json，outputs/ex019_boke_full/tune_speed2.json（起動の時間と定常の速度を分ける）

実行例:
    .venv_annotate/bin/python programs/ex019_boke_full/tune_speed.py --shards part_0003 part_0004
"""
import argparse
import copy
import json
import os
import shutil
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PKG = os.path.join(PROJECT_DIR, "Bokete_Annotation_programs_v5.0")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ex019_boke_full")
sys.path.insert(0, PKG)
from bokete_annotation.boke_pipeline import stage_boke_predict  # noqa: E402

# 設定の名前 → config.json の boke の上書き
SETTINGS = {"w8_b512_s128": {"num_workers": 8, "bokes_per_batch": 512, "sub_batch": 128},
            "w16_b512_s128": {"num_workers": 16, "bokes_per_batch": 512, "sub_batch": 128},
            "w16_b1024_s256": {"num_workers": 16, "bokes_per_batch": 1024, "sub_batch": 256},
            "w16_b1024_s128": {"num_workers": 16, "bokes_per_batch": 1024, "sub_batch": 128}}


def main() -> None:
    """設定ごとに1シャードを予測して速度を集める．

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", default=["part_0003", "part_0004"])
    parser.add_argument("--src", default=os.path.join(PKG, "outputs", "test_boke"))
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    args = parser.parse_args()
    cfg = json.load(open(os.path.join(PKG, "config.json"), encoding="utf-8"))
    res = {}
    for name in args.settings:
        d = os.path.join(OUTPUT_DIR, "tune2", name)
        os.makedirs(os.path.join(d, "boke", "list"), exist_ok=True)
        for sh in args.shards:
            shutil.copy(os.path.join(args.src, "boke", "list", sh + ".parquet"), os.path.join(d, "boke", "list"))
        if not os.path.exists(os.path.join(d, "speed.json")):
            c = copy.deepcopy(cfg)
            c["boke"].update(SETTINGS[name])
            stage_boke_predict(d, args.shards, "/mnt/d/Bokete_Scraping_v3.2/datasets/images", c, PKG)
        sp = json.load(open(os.path.join(d, "speed.json")))
        res[name] = {"all": sp["boke_predict"], "steady": sp["boke_predict_steady"], "startup_sec": sp["boke_predict_startup"]["sec"]}
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(OUTPUT_DIR, "tune_speed2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
