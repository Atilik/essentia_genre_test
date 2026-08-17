"""
Prediction-fingerprint probe: instead of the raw embeddings, use each model's own
genre-OUTPUT vector as features and train the 23-way Beatport classifier on it
(5-fold CV). Tests whether a model's genre-tuned output separates the subgenres
better/worse than the raw embedding it sits on.

Features (both cached from extract_beats.py — no re-extraction):
  - jamendo-effnet  : 87-dim MTG-Jamendo predictions   (the "why not MTG-Jamendo" test)
  - discogs400-effnet: 400-dim Discogs400 predictions
Compared against the embedding probe (best discogs_artist 65.4%).

Run in the /scratch sklearn venv (via cpu_short sbatch — 3.9, not the login node):
  .../sklearn_probe/bin/python scripts/probe_fingerprint.py
"""
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.probe_beats import load_labels, evaluate, BEATS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid

PRED_DIR = os.path.join(BEATS, "pred")
FINGERPRINTS = ["jamendo-effnet", "discogs400-effnet"]


def main():
    keep, y, _ = load_labels()
    print(f"n={len(y)}  genres={len(set(y))}  chance={100/len(set(y)):.1f}%\n")
    logreg = lambda: LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial")
    results = {}
    print(f"{'fingerprint (features)':<28}{'dim':>5}{'logreg acc':>12}{'macroF1':>9}{'centroid':>10}")
    for name in FINGERPRINTS:
        X = np.load(os.path.join(PRED_DIR, f"{name}.npy"))[keep]
        Xs = StandardScaler().fit_transform(X)
        lr = evaluate(Xs, y, logreg())
        nc = evaluate(Xs, y, NearestCentroid())
        results[name] = {"features": f"{name} predictions", "dim": X.shape[1],
                         "logreg": lr, "centroid_acc": nc["acc"], "n": len(y),
                         "chance": round(100/len(set(y)), 1)}
        print(f"{name+' preds':<28}{X.shape[1]:>5}{lr['acc']:>11.1f}%{lr['macro_f1']:>9}{nc['acc']:>9.1f}%")
        json.dump(results[name], open(os.path.join(BEATS, f"probe_fp_{name}.json"), "w"), indent=2)

    print("\nreference (embedding probes): discogs_artist 65.4%, discogs-effnet-bs64 63.8%")
    json.dump(results, open(os.path.join(BEATS, "probe_fingerprint_summary.json"), "w"), indent=2)
    print("wrote results/beats/probe_fp_*.json")


if __name__ == "__main__":
    main()
