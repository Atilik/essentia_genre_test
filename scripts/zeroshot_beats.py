"""
Zero-shot out-of-the-box view (Step 3b): map an off-the-shelf model's genre
predictions onto the 23 Beatport subgenres, no training. Two views:
  (i) direct-match: restricted argmax over the Beatport subgenres the taxonomy can
      express (coverage < 23); accuracy on tracks whose GT is mappable.
  (ii) super-genre: collapse to ~6 coarse families; accuracy over ALL tracks.
Uses cached predictions from discogs400-effnet (400-way) and jamendo-effnet (87-way).

Run in the /scratch sklearn venv (numpy only, no essentia):
  .../sklearn_probe/bin/python scripts/zeroshot_beats.py
"""
import os
import sys
import csv
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import BY_NAME, model_classes
from scripts.beats_mapping import BEATS_GENRES, FAMILY, mapped_indices

BEATS = os.path.join(ROOT, "results", "beats")
HEADS = ["discogs400-effnet", "jamendo-effnet"]


def load_rows():
    with open(os.path.join(BEATS, "rows.csv")) as f:
        rows = list(csv.DictReader(f))
    keep = [i for i, r in enumerate(rows) if r["ok"] == "1"]
    return keep, [rows[i]["genre"] for i in keep]


def score_head(head, keep, gt):
    m = BY_NAME[head]
    classes = model_classes(m)
    P = np.load(os.path.join(BEATS, "pred", f"{head}.npy"))[keep]     # [N, ncls]
    mi = mapped_indices(m["taxonomy"], classes)                       # {genre: [idx]}
    mapped_genres = [g for g in BEATS_GENRES if g in mi]

    # predicted Beatport genre = argmax over per-genre max activation
    def predict(row):
        return max(mapped_genres, key=lambda g: float(row[mi[g]].max()))
    preds = [predict(P[i]) for i in range(len(gt))]

    # (i) direct-match: only tracks whose GT is mappable
    dm_idx = [i for i, g in enumerate(gt) if g in mi]
    dm_acc = 100 * np.mean([preds[i] == gt[i] for i in dm_idx]) if dm_idx else float("nan")
    # per-genre recall on mappable genres
    recall = {}
    for g in mapped_genres:
        gi = [i for i in dm_idx if gt[i] == g]
        recall[g] = round(100 * np.mean([preds[i] == g for i in gi]), 1) if gi else None

    # (ii) super-genre over ALL tracks
    fam_acc = 100 * np.mean([FAMILY.get(preds[i]) == FAMILY.get(gt[i]) for i in range(len(gt))])

    return {
        "head": head, "taxonomy": m["taxonomy"],
        "coverage": f"{len(mapped_genres)}/23", "mapped_genres": mapped_genres,
        "unmapped_genres": [g for g in BEATS_GENRES if g not in mi],
        "direct_match_acc": round(dm_acc, 1), "direct_match_n": len(dm_idx),
        "super_genre_acc": round(fam_acc, 1), "super_genre_n": len(gt),
        "per_genre_recall": recall,
    }


def main():
    keep, gt = load_rows()
    print(f"n={len(gt)} tracks, {len(set(gt))} genres\n")
    out = {}
    for head in HEADS:
        r = score_head(head, keep, gt)
        out[head] = r
        print(f"=== {head} ({r['taxonomy']}) ===")
        print(f"  coverage {r['coverage']}  (unmapped: {r['unmapped_genres']})")
        print(f"  direct-match acc: {r['direct_match_acc']}%  on {r['direct_match_n']} mappable-GT tracks "
              f"(chance ~{100/len(r['mapped_genres']):.1f}%)")
        print(f"  super-genre acc:  {r['super_genre_acc']}%  on all {r['super_genre_n']} "
              f"(chance ~{100/len(set(FAMILY.values())):.1f}%)\n")
    json.dump(out, open(os.path.join(BEATS, "zeroshot.json"), "w"), indent=2)
    print("wrote results/beats/zeroshot.json")


if __name__ == "__main__":
    main()
