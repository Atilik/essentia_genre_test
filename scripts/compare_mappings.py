"""
Mapping robustness check: does the strict label→genre mapping under-sell dnb /
garage_dubstep?

The strict mapping ignores adjacent Discogs styles the models actually reach for
on our clips ('Halftime' on dnb, 'Grime' on garage). This re-scores every model
from the CACHED score vectors (no re-inference) under both mappings and reports
them side by side, so the expanded mapping is a transparent sensitivity check
rather than a silent swap.

  python scripts/compare_mappings.py     -> results/mapping_sensitivity.md
"""
import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import MODELS, model_classes
from scripts.mapping import GENRES, genre_label_indices, genre_labels

SCORES = os.path.join(ROOT, "results", "scores")


def evaluate(scores, rows, taxonomy, classes, expanded):
    gidx = genre_label_indices(taxonomy, classes, expanded=expanded)
    correct, per_genre = [], {g: [] for g in GENRES}
    for r, s in zip(rows, scores):
        if r["ok"] != "1":
            continue
        pred = max(GENRES, key=lambda g: float(s[gidx[g]].max()))
        hit = pred == r["genre_gt"]
        correct.append(hit)
        per_genre[r["genre_gt"]].append(hit)
    acc = 100 * np.mean(correct) if correct else float("nan")
    pg = {g: (100 * np.mean(v) if v else float("nan")) for g, v in per_genre.items()}
    return acc, pg


def main():
    md = ["# Mapping robustness — strict vs expanded label→genre mapping\n"]
    md.append("Re-scored from cached score vectors (no re-inference). **Expanded** adds "
              "`Halftime`→dnb and `Grime`→garage_dubstep (Discogs taxonomies only; "
              "MTG-Jamendo's 87 tags contain no equivalent, so it is unchanged).\n")
    md.append("| model | overall (strict→expanded) | dnb (strict→expanded) | garage_dubstep (strict→expanded) |")
    md.append("|---|--:|--:|--:|")

    print(f"{'model':<26}{'overall':>18}{'dnb':>18}{'garage':>18}")
    for m in MODELS:
        npy = os.path.join(SCORES, f"{m['name']}.npy")
        rowf = os.path.join(SCORES, f"{m['name']}.rows.csv")
        if not (os.path.exists(npy) and os.path.exists(rowf)):
            continue
        scores = np.load(npy)
        with open(rowf) as f:
            rows = list(csv.DictReader(f))
        classes = model_classes(m)
        a0, p0 = evaluate(scores, rows, m["taxonomy"], classes, False)
        a1, p1 = evaluate(scores, rows, m["taxonomy"], classes, True)
        print(f"{m['name']:<26}{a0:>8.1f}->{a1:<8.1f}{p0['dnb']:>8.1f}->{p1['dnb']:<8.1f}"
              f"{p0['garage_dubstep']:>8.1f}->{p1['garage_dubstep']:<8.1f}")
        fmt = lambda a, b: (f"{a:.1f}" if abs(a - b) < 0.05 else f"{a:.1f} → **{b:.1f}**")
        md.append(f"| {m['name']} | {fmt(a0,a1)} | {fmt(p0['dnb'],p1['dnb'])} | "
                  f"{fmt(p0['garage_dubstep'],p1['garage_dubstep'])} |")

    # what the expansion actually adds, per taxonomy
    md.append("\n## Labels added by the expanded mapping\n")
    seen = set()
    for m in MODELS:
        if m["taxonomy"] in seen:
            continue
        seen.add(m["taxonomy"])
        classes = model_classes(m)
        s = genre_labels(m["taxonomy"], classes, False)
        e = genre_labels(m["taxonomy"], classes, True)
        added = {g: sorted(set(e[g]) - set(s[g])) for g in GENRES}
        added = {g: v for g, v in added.items() if v}
        md.append(f"- **{m['taxonomy']}**: {added if added else 'nothing added'}")

    out = os.path.join(ROOT, "results", "mapping_sensitivity.md")
    open(out, "w").write("\n".join(md) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
