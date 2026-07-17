"""
Aggregate results/<model>.json into results/summary.md.

Tables: (1) headline restricted accuracy per model, split loop vs one-shot;
(2) per-genre restricted accuracy; (3) unrestricted top-1 accuracy + how often an
on-genre label even wins. Random baseline for the 4-way restricted task = 25%.

  python scripts/report.py
"""
import os
import sys
import glob
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import MODELS
from scripts.mapping import GENRES

RESULTS = os.path.join(ROOT, "results")
ORDER = [m["name"] for m in MODELS]        # manifest order (Discogs400, 519, Jamendo)


def load():
    out = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        r = json.load(open(p))
        if "restricted_acc" in r:
            out[r["model"]] = r
    return out


def cell(v):
    return "–" if v is None else f"{v:.1f}"


def main():
    res = load()
    if not res:
        print("no results yet")
        return
    models = [m for m in ORDER if m in res]

    md = ["# Essentia off-the-shelf genre classification — loops & one-shots\n"]
    md.append("4 genres (house / techno / dnb / garage_dubstep), 50 loops + 50 one-shots "
              "each = 400 clips. Ground truth = source sample pack. **Restricted 4-way "
              "argmax** (over only the labels mapped to the 4 genres); random = **25%**.\n")

    md.append("## Restricted accuracy — overall, loop vs one-shot\n")
    md.append("| model | family | backbone | overall | loop | one-shot | Δ(loop−1shot) |")
    md.append("|---|---|---|--:|--:|--:|--:|")
    for m in models:
        r = res[m]; ra = r["restricted_acc"]
        d = (None if ra["loop"] is None or ra["oneshot"] is None else ra["loop"] - ra["oneshot"])
        fam = {"discogs400": "Discogs400", "discogs519": "Discogs519", "mtg_jamendo": "MTG-Jamendo"}[r["taxonomy"]]
        md.append(f"| {m} | {fam} | {r['emb_algo']} | **{cell(ra['overall'])}** | "
                  f"{cell(ra['loop'])} | {cell(ra['oneshot'])} | {cell(d)} |")

    md.append("\n## Restricted accuracy per genre (recall)\n")
    md.append("| model | " + " | ".join(GENRES) + " |")
    md.append("|---|" + "|".join(["--:"] * len(GENRES)) + "|")
    for m in models:
        pg = res[m]["restricted_acc"]["per_genre"]
        md.append(f"| {m} | " + " | ".join(cell(pg[g]) for g in GENRES) + " |")

    md.append("\n## Unrestricted top-1 (full taxonomy)\n")
    md.append("*top-1 acc = model's single highest label maps to the right genre; "
              "on-genre% = how often the #1 label is any of the 4 genres at all.*\n")
    md.append("| model | top-1 overall | loop | one-shot | on-genre % |")
    md.append("|---|--:|--:|--:|--:|")
    for m in models:
        u = res[m]["unrestricted_top1_acc"]
        md.append(f"| {m} | {cell(u['overall'])} | {cell(u['loop'])} | "
                  f"{cell(u['oneshot'])} | {cell(u['on_genre_frac'])} |")

    out = os.path.join(RESULTS, "summary.md")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
