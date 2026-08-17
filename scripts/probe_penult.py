"""
Probe MTG-Jamendo's 512-d penultimate layer (5-fold CV), vs the raw embedding
(65.4%) and the 87-d prediction fingerprint (58.8%). Expected to land between them:
genre-tuned but pre-compression.

Run in the /scratch sklearn venv (cpu_short sbatch):
  .../sklearn_probe/bin/python scripts/probe_penult.py
"""
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.probe_beats import load_labels, evaluate, BEATS


def main():
    keep, y, _ = load_labels()
    X = np.load(os.path.join(BEATS, "jamendo_penult512.npy"))[keep]
    Xs = StandardScaler().fit_transform(X)
    lr = evaluate(Xs, y, LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial"))
    nc = evaluate(Xs, y, NearestCentroid())
    res = {"features": "jamendo penultimate (model/dense/BiasAdd)", "dim": X.shape[1],
           "logreg": lr, "centroid_acc": nc["acc"], "n": len(y),
           "chance": round(100 / len(set(y)), 1)}
    json.dump(res, open(os.path.join(BEATS, "probe_jamendo_penult512.json"), "w"), indent=2)
    print(f"n={len(y)}  chance={res['chance']}%")
    print(f"jamendo penultimate (512-d):  logreg {lr['acc']}%  macroF1 {lr['macro_f1']}  centroid {nc['acc']}%")
    print("\nladder:  raw embedding 65.4  >  ?penult?  >  fingerprint 58.8  >  zero-shot 36 (MTG)")
    print("wrote results/beats/probe_jamendo_penult512.json")


if __name__ == "__main__":
    main()
