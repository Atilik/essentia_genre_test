"""
Build the ground-truth track list for the beatsdataset_yt full-song experiment.

23 Beatport subgenres, ~40 full tracks each. Ground truth = top-level genre folder.
Unzips the archive once to $BEATS_DIR, drops the 2 corrupt *.part downloads and
.DS_Store, and writes a flat track manifest.

Output: data/beats_tracks.csv  (track_id, path, genre)   ~921 rows
Stdlib only — runs on the login node.
"""
import os
import csv
import zipfile
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.environ.get("BEATS_ZIP", "/scratch/mk9649/datasets/beatsdataset_yt.zip")
BEATS_DIR = os.environ.get("BEATS_DIR", "/scratch/mk9649/datasets/beatsdataset_yt")
AUDIO_EXT = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"}


def main():
    # --- extract once (skip if already unzipped) ---
    z = zipfile.ZipFile(ZIP)
    members = [i for i in z.infolist()
               if not i.is_dir() and not i.filename.startswith("__MACOSX")]
    parent = os.path.dirname(BEATS_DIR)
    if not os.path.isdir(BEATS_DIR):
        print(f"extracting {len(members)} members -> {parent} ...")
        z.extractall(parent, members=[m for m in members])
    else:
        print(f"{BEATS_DIR} already exists; not re-extracting")

    # --- walk extracted tree, build manifest ---
    rows = []
    dropped = collections.Counter()
    for m in members:
        name = m.filename                       # e.g. beatsdataset_yt/House/House040__....mp3
        base = os.path.basename(name)
        ext = os.path.splitext(base)[1].lower()
        parts = name.split("/")
        if base == ".DS_Store" or ext == ".part" or ext not in AUDIO_EXT:
            dropped[ext or base] += 1
            continue
        if len(parts) < 3:                      # need beatsdataset_yt/<genre>/<file>
            dropped["shallow"] += 1
            continue
        genre = parts[1]
        path = os.path.join(parent, name)
        if not os.path.isfile(path):
            dropped["missing_on_disk"] += 1
            continue
        rows.append((os.path.splitext(base)[0], path, genre))

    rows.sort()
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out = os.path.join(ROOT, "data", "beats_tracks.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "path", "genre"])
        w.writerows(rows)

    by_g = collections.Counter(g for _, _, g in rows)
    print(f"\nwrote {out}: {len(rows)} tracks, {len(by_g)} genres")
    print("dropped:", dict(dropped))
    print("per-genre counts:")
    for g, c in sorted(by_g.items()):
        print(f"  {g:<22} {c}")


if __name__ == "__main__":
    main()
