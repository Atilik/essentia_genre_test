"""
Extract MTG-Jamendo's 512-d PENULTIMATE layer per full track (run in overlay, GPU).

Pipeline: audio -> discogs-effnet-bs64 embedding [frames,1280] -> MTG-Jamendo head
with output = the penultimate dense layer 'model/dense/BiasAdd' [frames,512] ->
mean-pool over frames. This is MTG-Jamendo's genre-TUNED internal representation
(between the raw embedding and the 87-tag output).

Output: results/beats/jamendo_penult512.npy  float32 [N,512]  (row-aligned to rows.csv)
  ./sing <<< "python scripts/extract_penult.py"
"""
import os
import sys
import csv
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import EMB, BY_NAME

ROWS = os.path.join(ROOT, "results", "beats", "rows.csv")
OUT = os.path.join(ROOT, "results", "beats", "jamendo_penult512.npy")
SR = 16000
PENULT_NODE = "model/dense/BiasAdd"     # 512-d, per head JSON


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import essentia.standard as es
    with open(ROWS) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    emb_pb = os.path.join(ROOT, "models", "embeddings", EMB["discogs-effnet-bs64"][0])
    emb_model = es.TensorflowPredictEffnetDiscogs(graphFilename=emb_pb, output="PartitionedCall:1")
    head = BY_NAME["jamendo-effnet"]
    pen_model = es.TensorflowPredict2D(graphFilename=head["head_pb"],
                                       input="model/Placeholder", output=PENULT_NODE)

    N = len(rows)
    out = np.zeros((N, 512), dtype=np.float32)
    ok = []
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            audio = es.MonoLoader(filename=r["path"], sampleRate=SR, resampleQuality=4)()
            if audio.size == 0:
                raise ValueError("empty audio")
            emb = emb_model(audio)                       # [frames,1280]
            pen = np.asarray(pen_model(emb), dtype=np.float32)   # [frames,512]
            out[i] = pen.reshape(-1, 512).mean(axis=0)
            ok.append(1)
        except Exception as ex:
            print(f"[warn] {os.path.basename(r['path'])}: {ex}", flush=True)
            ok.append(0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N}  ({(i+1)/(time.time()-t0):.2f} tracks/s)", flush=True)

    np.save(OUT, out)
    print(f"done: {sum(ok)}/{N} ok in {time.time()-t0:.0f}s -> {OUT}  shape={out.shape}", flush=True)


if __name__ == "__main__":
    main()
