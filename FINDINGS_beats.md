# Full-track subgenre classification — beatsdataset_yt

Companion to the loop study ([FINDINGS.md](FINDINGS.md)). Same off-the-shelf Essentia
models, but on **921 full songs** across **23 Beatport subgenres** (~40 each), ground
truth = source subgenre. The models don't output the 23 labels, so we bridge the gap two
ways: a **learned probe** (train a classifier on the models' embeddings, 5-fold CV) and a
**zero-shot** label mapping. Numbers: [results/beats/summary.md](results/beats/summary.md).

## TL;DR

- **The representations carry subgenre — ~65% on a 23-way task** (chance 4.3%). Best
  embedding `discogs_artist` **65.4%**; five of six backbones cluster at 62–65%.
- **The genres that failed on loops now work.** DrumAndBass **90%**, Dubstep 67.5%,
  GlitchHop 80% under the probe. On loops these were ≈chance. The information was always in
  the embeddings — loops lacked context and the off-the-shelf head didn't expose it.
- **Even zero-shot (no training) is decent on full tracks**: Discogs400 hits **49%** on the
  21 mappable subgenres and **69%** at the coarse-family level — vs the near-chance
  dnb/dubstep collapse on loops. Full-track context rescues the off-the-shelf models too.
- **`discogs_track` — the loop-study winner — is the *worst* here (57.5%)**; `discogs_artist`
  wins. Best backbone is task/domain dependent.
- **Errors are dominated by the house family** (DeepHouse/House/TechHouse/ElectroHouse/
  FutureHouse/BigRoom all 42–54% recall), and they're **near-misses within the family** —
  expected for fine, overlapping subgenres.

## Learned probe — 23-way, 5-fold CV (chance 4.3%)

| embedding | dim | logreg acc | macro-F1 | nearest-centroid |
|---|--:|--:|--:|--:|
| **discogs_artist** | 1280 | **65.4** | 65.5 | 63.4 |
| discogs_multi | 1280 | 64.7 | 64.7 | 63.4 |
| discogs_label | 1280 | 64.0 | 63.9 | 61.8 |
| discogs-effnet-bs64 | 1280 | 63.8 | 63.9 | 62.2 |
| discogs_release | 1280 | 61.8 | 61.9 | 62.5 |
| discogs_track | 1280 | 57.5 | 57.5 | 57.3 |
| all-6 concat | 7680 | 64.9 | 65.0 | 65.0 |

Concatenating all six doesn't beat the best single embedding — they're largely redundant
(all are EffNet-Discogs projections). A linear probe already captures most of it (logreg ≈
nearest-centroid), i.e. the genres are close to linearly separable in these spaces.

### Why not train on MTG-Jamendo? Fingerprint probe

MTG-Jamendo isn't a separate representation — it's an 87-tag head on the same Discogs-EffNet
embeddings we already probe. Its only distinct contribution is its genre-OUTPUT vector. Probing
that ("fingerprint") instead of the raw embedding:

A monotonic staircase down through the MTG-Jamendo head — each layer discards subgenre info:

| features (probe) | dim | 23-way acc | what it is |
|---|--:|--:|---|
| discogs_artist embedding | 1280 | **65.4** | best raw embedding |
| discogs-effnet-bs64 embedding | 1280 | 63.8 | the embedding MTG-Jamendo runs on |
| MTG-Jamendo **penultimate** (`model/dense/BiasAdd`) | 512 | 62.3 | its genre-tuned internal layer |
| MTG-Jamendo predictions | 87 | 58.8 | its 87-tag output (fingerprint) |
| Discogs400 predictions | 400 | 58.0 | Discogs400 output (fingerprint) |
| MTG-Jamendo **zero-shot** (fixed head) | — | 36.0 / 57.3 | off-the-shelf argmax |

The key detail: MTG-Jamendo's **penultimate (62.3) is nearly as good as the raw embedding it's
computed from (63.8)** — its genre tuning barely hurts at that stage. The real losses come later:
the **87-genre bottleneck** costs ~3.5 pts, and the **fixed off-the-shelf head** costs the rest
(→36). So it's not the learned representation that's the problem — it's the coarse output layer +
frozen head. Ranking: **raw embedding (65) ≳ penultimate (62) > fingerprint (58) > zero-shot
(36–49)**. Train your own head on the embeddings (or MTG's penultimate); don't use its output.

### Per-genre recall (best embedding, discogs_artist)
Strong: PsyTrance/Trance **92.5**, DrumAndBass **90**, HardDance 87.5, Breaks/Hardcore 85,
GlitchHop 80. Weak: **DeepHouse 42.5, House 50, TechHouse/ElectroHouse/FutureHouse 52.5**.
The confusions are within-family near-misses — DeepHouse→ProgressiveHouse/House,
TechHouse→House, BigRoom↔ElectroHouse — not random errors.

## Zero-shot — off-the-shelf, no training

| model | coverage | direct-match acc | super-genre acc |
|---|---|--:|--:|
| discogs400-effnet | 21/23 | **49.4%** (n=840, chance 4.8%) | **68.6%** (chance 14.3%) |
| jamendo-effnet | 15/23 | 36.0% (n=600, chance 6.7%) | 57.3% |

*direct-match* = restricted argmax over the subgenres the taxonomy can name (BigRoom and
FutureHouse have no Discogs style; MTG-Jamendo can't name 8 of 23). *super-genre* = ~6 coarse
families over all tracks. Discogs400 (rich electronic taxonomy) beats MTG-Jamendo on both.

## Contrast with the loop study

| | loops (4 coarse genres) | full tracks (23 subgenres) |
|---|---|---|
| dnb / dubstep | ≈ chance (collapse into techno) | **DrumAndBass 90%, Dubstep 67.5%** (probe) |
| off-the-shelf zero-shot | near-chance on dnb/garage | 49% direct-match / 69% super-genre |
| conclusion | wrong domain (loops) + wrong head | representations are strong; full-track + learned head works |

This is the direct answer to "should we train our own head": **yes** — the Discogs-EffNet
embeddings clearly separate these subgenres on full tracks, and a simple linear probe already
reaches ~65% on 23 classes. Use **`discogs_artist`** (not `discogs_track`) for full-track
subgenre. Fine, overlapping house subgenres will remain the hard part for any model.

## Caveats

- The probe measures **representation quality** (a mapping is *learned*, 5-fold CV), not
  off-the-shelf correctness — that's the zero-shot view. Both are reported.
- 40 tracks/genre → per-genre recall has wide CIs (±~1–2% overall std across folds).
- YouTube full tracks may include talk/intro; full-track mean-pooling dilutes these.
- Beatport subgenres overlap by construction (DeepHouse vs TechHouse vs House) — within-house
  confusion is a property of the taxonomy, not just the model.

## Reproduce
`scripts/build_beats_gt.py` → `scripts/extract_beats.py` (GPU, full-track mean-pooled
embeddings) → `scripts/probe_beats.py` (sklearn venv) + `scripts/zeroshot_beats.py` →
`scripts/report_beats.py`. Zero-shot label map in `scripts/beats_mapping.py`.
