"""
Control: does the loop≫one-shot gap / techno-collapse survive ZERO-padding
instead of tiling short clips? Compares cached tile scores (results/scores/) vs
zero-pad scores (results/scores_zero/) for the given models.

  python scripts/compare_pad.py discogs400-effnet jamendo-track discogs400-maest-5s-pw discogs400-maest-30s-pw
"""
import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import BY_NAME, model_classes
from scripts.mapping import GENRES, genre_label_indices


def acc(scores, rows, gidx, subset=None):
    hits = []
    for r, s in zip(rows, scores):
        if r["ok"] != "1":
            continue
        if subset and r["clip_type"] != subset:
            continue
        pred = max(GENRES, key=lambda g: float(s[gidx[g]].max()))
        hits.append(pred == r["genre_gt"])
    return 100 * np.mean(hits) if hits else float("nan")


def load(scores_dir, name, rows_from=None):
    npy = os.path.join(scores_dir, f"{name}.npy")
    rowf = os.path.join(scores_dir, f"{name}.rows.csv")
    if not os.path.exists(rowf) and rows_from:
        rowf = rows_from
    scores = np.load(npy)
    with open(rowf) as f:
        rows = list(csv.DictReader(f))
    return scores, rows


def main():
    names = sys.argv[1:] or ["discogs400-effnet", "jamendo-track",
                             "discogs400-maest-5s-pw", "discogs400-maest-30s-pw"]
    tile_dir = os.path.join(ROOT, "results", "scores")
    zero_dir = os.path.join(ROOT, "results", "scores_zero")
    print(f"{'model':<26}{'overall t→z':>16}{'loop t→z':>16}{'oneshot t→z':>18}{'dnb':>14}{'garage':>14}")
    for name in names:
        m = BY_NAME[name]
        classes = model_classes(m)
        gidx = genre_label_indices(m["taxonomy"], classes)
        st, rt = load(tile_dir, name)
        sz, rz = load(zero_dir, name, rows_from=os.path.join(tile_dir, f"{name}.rows.csv"))

        def line(sc, rw, sub=None):
            return acc(sc, rw, gidx, sub)
        ov = (line(st, rt), line(sz, rz))
        lo = (line(st, rt, "loop"), line(sz, rz, "loop"))
        on = (line(st, rt, "oneshot"), line(sz, rz, "oneshot"))
        # per-genre one-shot (where padding bites hardest)
        def genre_on(sc, rw, g):
            hits = [max(GENRES, key=lambda gg: float(s[gidx[gg]].max())) == g
                    for r, s in zip(rw, sc) if r["ok"] == "1" and r["genre_gt"] == g and r["clip_type"] == "oneshot"]
            return 100 * np.mean(hits) if hits else float("nan")
        dnb = (genre_on(st, rt, "dnb"), genre_on(sz, rz, "dnb"))
        gar = (genre_on(st, rt, "garage_dubstep"), genre_on(sz, rz, "garage_dubstep"))
        f = lambda p: f"{p[0]:.0f}->{p[1]:.0f}"
        print(f"{name:<26}{f(ov):>16}{f(lo):>16}{f(on):>18}{f(dnb):>14}{f(gar):>14}")
    print("\n(t=tile, z=zero-pad; dnb/garage columns are ONE-SHOT accuracy)")


if __name__ == "__main__":
    main()
