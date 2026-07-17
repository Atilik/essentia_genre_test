"""
Diagnostic: WHY do dnb / garage_dubstep fail — missing labels, a too-narrow
mapping, or genuine model confusion?

For each true genre, reports the labels the model actually activates most
(unrestricted top-1 distribution + mean activation of candidate labels), so we can
see whether e.g. dnb clips are being called "Breakbeat" (a mapping gap) or
"Techno" (real confusion).

  python scripts/analyze_labels.py --model jamendo-track
"""
import os
import sys
import csv
import argparse
import collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import BY_NAME, model_classes
from scripts.mapping import GENRES, genre_labels

SCORES = os.path.join(ROOT, "results", "scores")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BY_NAME))
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    m = BY_NAME[args.model]
    classes = model_classes(m)
    scores = np.load(os.path.join(SCORES, f"{args.model}.npy"))
    with open(os.path.join(SCORES, f"{args.model}.rows.csv")) as f:
        rows = list(csv.DictReader(f))
    mapped = genre_labels(m["taxonomy"], classes)
    mapped_set = {l for ls in mapped.values() for l in ls}

    print(f"\n########## {args.model} ({m['taxonomy']}, {len(classes)} classes) ##########")
    for g in GENRES:
        idx = [i for i, r in enumerate(rows) if r["genre_gt"] == g and r["ok"] == "1"]
        if not idx:
            continue
        S = scores[idx]
        # what does the model actually call these clips (unrestricted)?
        top1 = collections.Counter(classes[int(i)] for i in S.argmax(axis=1))
        print(f"\n=== true genre: {g}  (n={len(idx)}) ===")
        print("  most frequent UNRESTRICTED top-1 labels:")
        for lab, c in top1.most_common(args.top):
            tag = "  <- mapped to " + next((k for k, v in mapped.items() if lab in v), "") \
                  if lab in mapped_set else "  (unmapped)"
            print(f"     {c:>3}x  {lab}{tag}")
        # mean activation of this genre's own mapped labels vs the winner genre's
        print("  mean activation of each genre's mapped labels:")
        for g2 in GENRES:
            cols = [classes.index(l) for l in mapped[g2]]
            print(f"     {g2:<15} {S[:, cols].max(axis=1).mean():.4f}"
                  + ("   <-- true" if g2 == g else ""))


if __name__ == "__main__":
    main()
