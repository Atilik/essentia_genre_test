"""
Assemble the beats full-track results into results/beats/summary.md:
 - learned-probe accuracy per embedding backbone (headline) + best per-genre recall
 - zero-shot direct-match + super-genre accuracy
Reads results/beats/probe_summary.json and results/beats/zeroshot.json.

  .../sklearn_probe/bin/python scripts/report_beats.py   (numpy only)
"""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEATS = os.path.join(ROOT, "results", "beats")


def main():
    probe = json.load(open(os.path.join(BEATS, "probe_summary.json")))
    zs = json.load(open(os.path.join(BEATS, "zeroshot.json")))
    chance = probe[next(iter(probe))]["chance"]

    md = ["# Beatsdataset_yt — full-track subgenre classification\n"]
    md.append("921 full tracks, **23 Beatport subgenres** (~40 each). Two views bridge the "
              "taxonomy mismatch: a **learned probe** (train a classifier on the models' "
              f"embeddings, 5-fold CV) and a **zero-shot** label mapping. Chance = {chance}%.\n")

    md.append("## Learned probe — 23-way accuracy per embedding (5-fold CV)\n")
    md.append("| features | dim | logreg acc | ±std | macro-F1 | nearest-centroid |")
    md.append("|---|--:|--:|--:|--:|--:|")
    order = sorted([k for k in probe if k != "ALL6_concat"],
                   key=lambda k: -probe[k]["logreg"]["acc"]) + ["ALL6_concat"]
    for k in order:
        r = probe[k]
        md.append(f"| {k} | {r['dim']} | **{r['logreg']['acc']}** | {r['logreg']['acc_std']} | "
                  f"{r['logreg']['macro_f1']} | {r['centroid_acc']} |")
    md.append(f"\nRandom baseline = {chance}%.\n")

    # best backbone per-genre recall
    best = max((k for k in probe if k != "ALL6_concat"), key=lambda k: probe[k]["logreg"]["acc"])
    pg = probe[best]["logreg"]["per_genre_recall"]
    md.append(f"## Per-genre recall — best embedding ({best}, probe)\n")
    md.append("| genre | recall | | genre | recall |")
    md.append("|---|--:|---|---|--:|")
    items = sorted(pg.items(), key=lambda x: -x[1])
    half = (len(items) + 1) // 2
    for i in range(half):
        l = f"| {items[i][0]} | {items[i][1]} |"
        r = f" | {items[i+half][0]} | {items[i+half][1]} |" if i + half < len(items) else " | | |"
        md.append(l + r)

    md.append("\n## Zero-shot (off-the-shelf, no training)\n")
    md.append("| model | coverage | direct-match acc | super-genre acc |")
    md.append("|---|---|--:|--:|")
    for head, r in zs.items():
        md.append(f"| {head} | {r['coverage']} | {r['direct_match_acc']}% "
                  f"(n={r['direct_match_n']}) | {r['super_genre_acc']}% |")
    md.append(f"\nUnmapped subgenres (no taxonomy label): "
              f"{zs[list(zs)[0]]['unmapped_genres']}. Super-genre = ~6 coarse families.\n")

    out = os.path.join(BEATS, "summary.md")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
