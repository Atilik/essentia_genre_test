"""
Learned probe (Step 3a): can each model's representation separate the 23 Beatport
subgenres? Trains a cross-validated linear classifier on the cached full-track
embeddings — solves the taxonomy mismatch by LEARNING the label mapping from data.

For each embedding backbone (+ the concatenation of all 6): StandardScaler ->
StratifiedKFold(5) -> LogisticRegression(multinomial). Reports accuracy, macro-F1,
per-genre recall, and a summed 23x23 confusion. Baselines: chance (1/23) and a
nearest-class-centroid classifier.

Runs in the /scratch sklearn venv (no essentia/GPU needed — reads cached .npy):
  /scratch/mk9649/venvs/sklearn_probe/bin/python scripts/probe_beats.py
"""
import os
import csv
import json
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEATS = os.path.join(ROOT, "results", "beats")
EMB_DIR = os.path.join(BEATS, "emb")
SEED = 42
K = 5


def load_labels():
    with open(os.path.join(BEATS, "rows.csv")) as f:
        rows = list(csv.DictReader(f))
    keep = [i for i, r in enumerate(rows) if r["ok"] == "1"]
    y = np.array([rows[i]["genre"] for i in keep])
    return keep, y, rows


def evaluate(X, y, clf):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    pred = cross_val_predict(clf, X, y, cv=skf)
    # per-fold accuracy for a std estimate
    accs = []
    for tr, te in skf.split(X, y):
        accs.append(accuracy_score(y[te], pred[te]))
    genres = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=genres)
    recall = {g: (cm[i, i] / cm[i].sum() * 100 if cm[i].sum() else 0.0)
              for i, g in enumerate(genres)}
    return {
        "acc": round(accuracy_score(y, pred) * 100, 1),
        "acc_std": round(float(np.std(accs)) * 100, 1),
        "macro_f1": round(f1_score(y, pred, average="macro") * 100, 1),
        "per_genre_recall": {g: round(v, 1) for g, v in recall.items()},
        "genres": genres,
        "confusion": cm.tolist(),
    }


def main():
    keep, y, _ = load_labels()
    backbones = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(EMB_DIR, "*.npy")))
    feats = {b: np.load(os.path.join(EMB_DIR, f"{b}.npy"))[keep] for b in backbones}
    print(f"n={len(y)}  genres={len(set(y))}  chance={100/len(set(y)):.1f}%  backbones={backbones}\n")

    logreg = lambda: LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial")
    os.makedirs(BEATS, exist_ok=True)
    results = {}
    print(f"{'features':<26}{'logreg acc':>12}{'±std':>7}{'macroF1':>9}{'centroid':>10}")
    for b in backbones + ["ALL6_concat"]:
        X = np.concatenate([feats[k] for k in backbones], axis=1) if b == "ALL6_concat" else feats[b]
        Xs = StandardScaler().fit_transform(X)
        lr = evaluate(Xs, y, logreg())
        nc = evaluate(Xs, y, NearestCentroid())
        results[b] = {"logreg": lr, "centroid_acc": nc["acc"], "n": len(y),
                      "chance": round(100 / len(set(y)), 1), "dim": X.shape[1]}
        print(f"{b:<26}{lr['acc']:>11.1f}%{lr['acc_std']:>7}{lr['macro_f1']:>9}{nc['acc']:>9.1f}%")
        json.dump({"backbone": b, **results[b]},
                  open(os.path.join(BEATS, f"probe_{b}.json"), "w"), indent=2)

    json.dump(results, open(os.path.join(BEATS, "probe_summary.json"), "w"), indent=2)
    best = max(backbones, key=lambda b: results[b]["logreg"]["acc"])
    print(f"\nbest single backbone: {best} ({results[best]['logreg']['acc']}%); "
          f"wrote results/beats/probe_*.json")


if __name__ == "__main__":
    main()
