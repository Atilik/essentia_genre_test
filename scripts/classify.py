"""
Run ONE Essentia genre model over the eval set and cache its raw class scores.

Two-stage per Essentia: audio -> embedding backbone -> classifier head. Exact TF
node names, sample rate, and the embedding algorithm are read from the model's
metadata JSONs (schemas differ across heads), so nothing is hard-coded.

Output (aligned to eval_set.csv row order, so scoring can be re-run/re-mapped
without re-inference):
  results/scores/<model>.npy        float32 [N, n_classes]  (mean-pooled over frames)
  results/scores/<model>.rows.csv   path, genre_gt, clip_type, duration_s, ok

Usage (inside the essentia overlay, on a compute node):
  ./sing <<< "python scripts/classify.py --model discogs400-effnet"
  ./sing <<< "python scripts/classify.py --model discogs400-effnet --limit 4"   # smoke
"""
import os
import re
import sys
import csv
import json
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import BY_NAME

EVAL = os.path.join(ROOT, "data", "eval_set.csv")
OUT_DIR = os.path.join(ROOT, "results", "scores")


def load_json(p):
    with open(p) as f:
        return json.load(f)


def pick_output(schema_outputs, purpose):
    """Node name of the output whose output_purpose == purpose (else first)."""
    for o in schema_outputs:
        if (o.get("output_purpose") or "") == purpose:
            return o["name"]
    return schema_outputs[0]["name"]


def min_input_samples(model, sr):
    """Minimum samples to feed the backbone; shorter clips are tiled (looped) up
    to this. EffNet emits 0 patches ('LIST_EMPTY') below ~1 patch (~2 s); MAEST
    raises 'input signal is too short' below its context window (and tiling to the
    exact window is a few mel-frames short). Margins avoid both."""
    if model["emb_algo"] == "effnet":
        return 3 * sr                        # >= one ~2 s EffNet patch
    m = re.search(r"maest-(\d+)s", model["emb_key"])
    ctx = int(m.group(1)) if m else 10
    return (ctx + 2) * sr                     # >= context window + 2 s margin


def build_runner(model):
    """Return (sample_rate, classes, predict_fn, min_samples)."""
    import essentia.standard as es

    head = load_json(model["head_json"])
    emb = load_json(model["emb_json"])
    sr = int(head["inference"]["sample_rate"])
    algo = model["emb_algo"]

    if algo == "effnet":
        # EffNet backbone -> separate TensorflowPredict2D genre head.
        classes = head["classes"]
        head_in = head["schema"]["inputs"][0]["name"]
        head_out = pick_output(head["schema"]["outputs"], "predictions")
        emb_out = pick_output(emb["schema"]["outputs"], "embeddings")   # PartitionedCall:1
        emb_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=model["emb_pb"], output=emb_out)
        head_model = es.TensorflowPredict2D(
            graphFilename=model["head_pb"], input=head_in, output=head_out)

        def predict(audio):
            return np.asarray(head_model(emb_model(audio)))            # [frames, n_classes]

    elif algo == "maest":
        # MAEST transformers were trained on Discogs400/519 and carry that
        # classifier internally: TensorflowPredictMAEST's default output IS the
        # genre prediction (shape (1,1,1,n_classes)). The separate *-maest head
        # expects an intermediate we don't extract, so we use the native output.
        classes = emb["classes"]
        maest = es.TensorflowPredictMAEST(graphFilename=model["emb_pb"])

        def predict(audio):
            p = np.asarray(maest(audio), dtype=np.float32)            # (1,1,1,n_classes)
            return p.reshape(-1, len(classes))
    else:
        raise ValueError(f"unknown emb_algo {algo!r}")

    return sr, classes, predict, min_input_samples(model, sr)


def load_audio(path, sr, min_samples=0, pad="tile"):
    """Load mono @ sr. If min_samples>0 and the clip is shorter, extend it to
    min_samples. pad='tile' loops the clip (default; loops are made to repeat);
    pad='zero' appends silence (control — avoids manufacturing a rhythm)."""
    import essentia.standard as es
    audio = es.MonoLoader(filename=path, sampleRate=sr, resampleQuality=4)()
    if min_samples and audio.size and len(audio) < min_samples:
        if pad == "zero":
            audio = np.concatenate(
                [audio, np.zeros(min_samples - len(audio), dtype=audio.dtype)])
        else:
            reps = int(np.ceil(min_samples / len(audio)))
            audio = np.tile(audio, reps)[:min_samples]
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BY_NAME))
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--limit", type=int, default=None, help="first N clips (smoke test)")
    ap.add_argument("--pad", choices=["tile", "zero"], default="tile",
                    help="how to extend clips below the backbone minimum")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    model = BY_NAME[args.model]
    with open(args.eval) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    sr, classes, predict, min_samples = build_runner(model)
    print(f"model={args.model} algo={model['emb_algo']} sr={sr} "
          f"n_classes={len(classes)} clips={len(rows)} "
          f"min_samples={min_samples}({min_samples//sr if sr else 0}s)", flush=True)

    scores = np.zeros((len(rows), len(classes)), dtype=np.float32)
    ok = []
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            audio = load_audio(r["path"], sr, min_samples, args.pad)
            if audio.size == 0:
                raise ValueError("empty audio")
            frame_scores = predict(audio)                 # [frames, n_classes]
            scores[i] = np.asarray(frame_scores, dtype=np.float32).reshape(
                -1, len(classes)).mean(axis=0)
            ok.append(1)
        except Exception as e:
            print(f"[warn] {os.path.basename(r['path'])}: {e}", flush=True)
            ok.append(0)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  ({(i+1)/(time.time()-t0):.1f} clips/s)", flush=True)

    np.save(os.path.join(args.out, f"{args.model}.npy"), scores)
    with open(os.path.join(args.out, f"{args.model}.rows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "genre_gt", "clip_type", "duration_s", "ok"])
        for r, o in zip(rows, ok):
            w.writerow([r["path"], r["genre"], r["clip_type"], r.get("duration_s", ""), o])
    print(f"wrote {args.model}.npy {scores.shape} | ok={sum(ok)}/{len(rows)} "
          f"in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
