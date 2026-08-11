# Beatsdataset_yt — full-track subgenre classification

921 full tracks, **23 Beatport subgenres** (~40 each). Two views bridge the taxonomy mismatch: a **learned probe** (train a classifier on the models' embeddings, 5-fold CV) and a **zero-shot** label mapping. Chance = 4.3%.

## Learned probe — 23-way accuracy per embedding (5-fold CV)

| features | dim | logreg acc | ±std | macro-F1 | nearest-centroid |
|---|--:|--:|--:|--:|--:|
| discogs_artist | 1280 | **65.4** | 1.9 | 65.5 | 63.4 |
| discogs_multi | 1280 | **64.7** | 1.8 | 64.7 | 63.4 |
| discogs_label | 1280 | **64.0** | 2.2 | 63.9 | 61.8 |
| discogs-effnet-bs64 | 1280 | **63.8** | 0.8 | 63.9 | 62.2 |
| discogs_release | 1280 | **61.8** | 2.3 | 61.9 | 62.5 |
| discogs_track | 1280 | **57.5** | 1.0 | 57.5 | 57.3 |
| ALL6_concat | 7680 | **64.9** | 1.6 | 65.0 | 65.0 |

Random baseline = 4.3%.

## Per-genre recall — best embedding (discogs_artist, probe)

| genre | recall | | genre | recall |
|---|--:|---|---|--:|
| PsyTrance | 92.5 | | IndieDanceNuDisco | 57.5 |
| Trance | 92.5 | | HipHop | 55.0 |
| DrumAndBass | 90.0 | | BigRoom | 53.7 |
| HardDance | 87.5 | | ElectroHouse | 52.5 |
| Breaks | 85.0 | | FutureHouse | 52.5 |
| HardcoreHardTechno | 85.0 | | Minimal | 52.5 |
| GlitchHop | 80.0 | | TechHouse | 52.5 |
| ReggaeDub | 70.0 | | Dance | 50.0 |
| Dubstep | 67.5 | | House | 50.0 |
| ProgressiveHouse | 65.0 | | ElectronicaDowntempo | 47.5 |
| Techno | 62.5 | | DeepHouse | 42.5 |
| FunkRAndB | 60.0 | | | |

## Zero-shot (off-the-shelf, no training)

| model | coverage | direct-match acc | super-genre acc |
|---|---|--:|--:|
| discogs400-effnet | 21/23 | 49.4% (n=840) | 68.6% |
| jamendo-effnet | 15/23 | 36.0% (n=600) | 57.3% |

Unmapped subgenres (no taxonomy label): ['BigRoom', 'FutureHouse']. Super-genre = ~6 coarse families.

