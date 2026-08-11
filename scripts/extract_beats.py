"""
Extract full-track features for beatsdataset_yt (run in the essentia overlay, GPU).

Per track: load the full song once (MonoLoader @ 16 kHz, no tiling — tracks are
minutes long), then run all 6 Discogs-EffNet embedding backbones (mean-pool the
[frames,1280] output over frames) plus two genre heads (discogs400-effnet,
jamendo-effnet) for the zero-shot view. Loading audio once amortises I/O.

Cache (row-aligned to data/beats_tracks.csv order):
  results/beats/emb/<backbone>.npy   float32 [N,1280]   (6 files)
  results/beats/pred/<head>.npy      float32 [N,ncls]   (2 files: 400, 87)
  results/beats/rows.csv             track_id, path, genre, ok

Usage:
  ./sing <<< "python scripts/extract_beats.py"
  ./sing <<< "python scripts/extract_beats.py --limit 6"   # smoke
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import EMB, BY_NAME

TRACKS = os.path.join(ROOT, "data", "beats_tracks.csv")
OUT = os.path.join(ROOT, "results", "beats")
SR = 16000
# 6 EffNet embedding backbones (manifest keys) + 2 heads for the zero-shot view
BACKBONES = ["discogs-effnet-bs64", "discogs_artist", "discogs_label",
             "discogs_release", "discogs_track", "discogs_multi"]
HEADS = ["discogs400-effnet", "jamendo-effnet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import essentia.standard as es

    with open(TRACKS) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(os.path.join(OUT, "emb"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)

    # build the 6 embedding models + 2 heads once
    emb_models = {}
    for key in BACKBONES:
        pb = os.path.join(ROOT, "models", "embeddings", EMB[key][0])
        ejson = EMB[key][2].rsplit("/", 1)[1]
        ejson = os.path.join(ROOT, "models", "heads", ejson)
        out_node = next(o["name"] for o in json.load(open(ejson))["schema"]["outputs"]
                        if (o.get("output_purpose") or "") == "embeddings")
        emb_models[key] = es.TensorflowPredictEffnetDiscogs(graphFilename=pb, output=out_node)

    head_runners = {}   # head_name -> (emb_model, head_model, n_classes)
    for hname in HEADS:
        m = BY_NAME[hname]
        hj = json.load(open(m["head_json"]))
        head_in = hj["schema"]["inputs"][0]["name"]
        head_out = next(o["name"] for o in hj["schema"]["outputs"]
                        if (o.get("output_purpose") or "") == "predictions")
        head_model = es.TensorflowPredict2D(graphFilename=m["head_pb"],
                                            input=head_in, output=head_out)
        head_runners[hname] = (m["emb_key"], head_model, len(hj["classes"]))

    N = len(rows)
    emb_out = {k: np.zeros((N, 1280), dtype=np.float32) for k in BACKBONES}
    pred_out = {h: np.zeros((N, head_runners[h][2]), dtype=np.float32) for h in HEADS}
    ok = []
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            audio = es.MonoLoader(filename=r["path"], sampleRate=SR, resampleQuality=4)()
            if audio.size == 0:
                raise ValueError("empty audio")
            frames = {}
            for k in BACKBONES:
                e = np.asarray(emb_models[k](audio), dtype=np.float32)   # [frames,1280]
                frames[k] = e
                emb_out[k][i] = e.mean(axis=0)
            for h in HEADS:
                ek, head_model, _ = head_runners[h]
                p = np.asarray(head_model(emb_models[ek](audio)), dtype=np.float32)
                pred_out[h][i] = p.mean(axis=0)
            ok.append(1)
        except Exception as ex:
            print(f"[warn] {os.path.basename(r['path'])}: {ex}", flush=True)
            ok.append(0)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N}  ({(i+1)/(time.time()-t0):.2f} tracks/s)", flush=True)

    for k in BACKBONES:
        np.save(os.path.join(OUT, "emb", f"{k}.npy"), emb_out[k])
    for h in HEADS:
        np.save(os.path.join(OUT, "pred", f"{h}.npy"), pred_out[h])
    with open(os.path.join(OUT, "rows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "path", "genre", "ok"])
        for r, o in zip(rows, ok):
            w.writerow([r["track_id"], r["path"], r["genre"], o])
    print(f"done: {sum(ok)}/{N} ok in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
