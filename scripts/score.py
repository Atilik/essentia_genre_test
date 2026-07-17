"""
Score one model's cached class scores against the 4-genre ground truth.

Two views (per plan):
  restricted : argmax over ONLY the labels mapped to house/techno/dnb/garage_dubstep
               (max activation per genre) -> 4-way prediction + confusion matrix.
  unrestricted top-1 : the model's single highest label over the FULL taxonomy,
               mapped to a genre or "other" (does an on-genre label even win?).

Accuracy is broken out overall / per clip_type (loop vs one-shot) / per genre.

  python scripts/score.py --model discogs400-effnet
"""
import os
import sys
import csv
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import BY_NAME, model_classes
from scripts.mapping import GENRES, genre_label_indices, genre_labels

SCORES_DIR = os.path.join(ROOT, "results", "scores")
OUT_DIR = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BY_NAME))
    ap.add_argument("--scores", default=SCORES_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    model = BY_NAME[args.model]
    classes = model_classes(model)
    scores = np.load(os.path.join(args.scores, f"{args.model}.npy"))
    with open(os.path.join(args.scores, f"{args.model}.rows.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == scores.shape[0] == 0 or len(rows) == scores.shape[0], "rows/scores misaligned"

    gidx = genre_label_indices(model["taxonomy"], classes)          # genre -> [class idx]
    # reverse lookup for unrestricted top-1
    lab2genre = {}
    for g, labs in genre_labels(model["taxonomy"], classes).items():
        for l in labs:
            lab2genre[l] = g

    gi = {g: i for i, g in enumerate(GENRES)}
    conf = np.zeros((len(GENRES), len(GENRES)), dtype=int)          # true x pred (restricted)
    recs = []
    n_fail = 0
    for r, s in zip(rows, scores):
        if r["ok"] != "1":
            n_fail += 1
            continue
        gt = r["genre_gt"]
        # restricted: max activation per genre, then argmax
        gscore = {g: float(s[gidx[g]].max()) for g in GENRES}
        pred = max(GENRES, key=lambda g: gscore[g])
        # unrestricted top-1
        ti = int(np.argmax(s))
        top1_label = classes[ti]
        top1_genre = lab2genre.get(top1_label, "other")
        conf[gi[gt], gi[pred]] += 1
        recs.append({"gt": gt, "clip_type": r["clip_type"], "pred": pred,
                     "top1_label": top1_label, "top1_genre": top1_genre})

    def acc(subset):
        return round(100 * np.mean([r["pred"] == r["gt"] for r in subset]), 1) if subset else None

    def top1acc(subset):
        return round(100 * np.mean([r["top1_genre"] == r["gt"] for r in subset]), 1) if subset else None

    result = {
        "model": args.model, "taxonomy": model["taxonomy"], "emb_algo": model["emb_algo"],
        "n": len(recs), "n_fail": n_fail,
        "restricted_acc": {
            "overall": acc(recs),
            "loop": acc([r for r in recs if r["clip_type"] == "loop"]),
            "oneshot": acc([r for r in recs if r["clip_type"] == "oneshot"]),
            "per_genre": {g: acc([r for r in recs if r["gt"] == g]) for g in GENRES},
            "per_genre_loop": {g: acc([r for r in recs if r["gt"] == g and r["clip_type"] == "loop"]) for g in GENRES},
            "per_genre_oneshot": {g: acc([r for r in recs if r["gt"] == g and r["clip_type"] == "oneshot"]) for g in GENRES},
        },
        "unrestricted_top1_acc": {
            "overall": top1acc(recs),
            "loop": top1acc([r for r in recs if r["clip_type"] == "loop"]),
            "oneshot": top1acc([r for r in recs if r["clip_type"] == "oneshot"]),
            "on_genre_frac": round(100 * np.mean([r["top1_genre"] != "other" for r in recs]), 1) if recs else None,
        },
        "confusion_restricted": {"genres": GENRES, "matrix_true_by_pred": conf.tolist()},
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(result, open(os.path.join(args.out, f"{args.model}.json"), "w"), indent=2)
    ra = result["restricted_acc"]
    print(f"{args.model:<24} restricted overall={ra['overall']}  loop={ra['loop']}  "
          f"oneshot={ra['oneshot']}  | top1={result['unrestricted_top1_acc']['overall']}  "
          f"(n={len(recs)}, fail={n_fail})")


if __name__ == "__main__":
    main()
