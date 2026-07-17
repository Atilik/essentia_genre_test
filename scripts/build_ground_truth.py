"""
Build the ground-truth inventory + balanced eval set from SMALL_SAMPLES.zip.

Ground truth = which commercial sample pack a clip came from (4 genres). Clip-type
(loop vs one-shot) is read from the immediate SUBFOLDER name inside each pack
(reliable here: packs ship "Drum Loops", "Drum Oneshots", "One Shots", ...). We do
NOT key off the full path — e.g. the dnb pack is "…Liquid Rollers…" and "roller"
is a loop cue, so only the subfolder is inspected.

Sampling: 50 loops + 50 one-shots per genre (seed=42); ambiguous-type folders
(FX, Atmosphere, Vocals, Pads, Textures, Buildups…) are excluded from the eval set.

Outputs:
  data/ground_truth.csv   every candidate audio: zip_name, genre, clip_type, subfolder
  data/eval_set.csv       the 400 sampled clips: path, genre, clip_type, subfolder, zip_name
  data/audio/<genre>/<clip_type>/...   extracted wavs

Stdlib only — runs on the login node.
"""
import os
import re
import csv
import sys
import random
import zipfile
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.environ.get("SAMPLES_ZIP", "/scratch/mk9649/datasets/SMALL_SAMPLES.zip")
AUDIO_EXT = {".wav", ".aiff", ".aif", ".flac"}
SEED = 42
PER_TYPE = 50                      # 50 loops + 50 one-shots per genre

# genre -> pack folder prefix (unique level-2 dir name start). Matched on the
# exact path component, so mojibake in the garage folder name is irrelevant.
PACKS = {
    "house": "house_Sample Market",
    "techno": "techno_TWHS",
    "dnb": "dnb_KAN",
    "garage_dubstep": "garage_dubstep-",
}
_ONESHOT = re.compile(r"one[ _]?shot|single[ _]?hit|\bhits\b|\bhit\b|\bstab", re.I)
_LOOP = re.compile(r"loop|beats|break|groove|roller|fill", re.I)


def clip_type(subfolder):
    if _ONESHOT.search(subfolder):
        return "oneshot"
    if _LOOP.search(subfolder):
        return "loop"
    return "other"                 # FX / atmos / vocals / pads / textures / buildups


def genre_of(parts):
    """parts = path.split('/'); returns genre if parts[1] matches a pack, else None."""
    if len(parts) < 2:
        return None
    pack = parts[1]
    for g, prefix in PACKS.items():
        if pack.startswith(prefix):
            return g
    return None


def main():
    z = zipfile.ZipFile(ZIP)
    candidates = []                # (zip_name, genre, clip_type, subfolder)
    for info in z.infolist():
        if info.is_dir() or info.filename.startswith("__MACOSX"):
            continue
        if os.path.splitext(info.filename)[1].lower() not in AUDIO_EXT:
            continue
        parts = info.filename.split("/")
        g = genre_of(parts)
        if g is None or len(parts) < 4:      # need pack/subfolder/…/file
            continue
        sub = parts[2]
        candidates.append((info.filename, g, clip_type(sub), sub))

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "ground_truth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zip_name", "genre", "clip_type", "subfolder"])
        w.writerows(candidates)

    # pools per (genre, clip_type)
    pools = collections.defaultdict(list)
    for row in candidates:
        pools[(row[1], row[2])].append(row)
    print("candidate audio counts (genre x clip_type):")
    for g in PACKS:
        cnt = {t: len(pools[(g, t)]) for t in ("loop", "oneshot", "other")}
        print(f"  {g:<15} loop={cnt['loop']:<4} oneshot={cnt['oneshot']:<4} other(excl)={cnt['other']}")

    rng = random.Random(SEED)
    selected = []
    for g in PACKS:
        for t in ("loop", "oneshot"):
            pool = sorted(pools[(g, t)])          # deterministic before shuffle
            if len(pool) < PER_TYPE:
                print(f"  [warn] {g}/{t}: only {len(pool)} < {PER_TYPE}; taking all")
            pick = pool if len(pool) <= PER_TYPE else rng.sample(pool, PER_TYPE)
            selected.extend(pick)

    # extract the selected clips + write eval_set.csv
    audio_root = os.path.join(ROOT, "data", "audio")
    eval_rows = []
    for i, (zn, g, t, sub) in enumerate(selected):
        base = os.path.basename(zn)
        safe_sub = re.sub(r"[^A-Za-z0-9]+", "_", sub).strip("_")
        dest_dir = os.path.join(audio_root, g, t)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{i:04d}__{safe_sub}__{base}")
        with z.open(zn) as src, open(dest, "wb") as out:
            out.write(src.read())
        eval_rows.append([dest, g, t, sub, zn])

    with open(os.path.join(ROOT, "data", "eval_set.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "genre", "clip_type", "subfolder", "zip_name"])
        w.writerows(eval_rows)

    print(f"\neval set: {len(eval_rows)} clips extracted -> data/audio/")
    by = collections.Counter((r[1], r[2]) for r in eval_rows)
    for g in PACKS:
        print(f"  {g:<15} loop={by[(g,'loop')]:<3} oneshot={by[(g,'oneshot')]}")


if __name__ == "__main__":
    main()
