# Can off-the-shelf Essentia genre models classify loops & one-shots?

15 off-the-shelf [Essentia](https://essentia.upf.edu/models.html) genre models, run on
**400 sample-library clips** (50 loops + 50 one-shots × 4 genres: house / techno / dnb /
garage_dubstep). Ground truth = the commercial sample pack each clip came from. Full
numbers in [results/summary.md](results/summary.md).

## TL;DR

- **Yes, but only just — and only for loops.** Best model reaches **56%** on a 4-way
  choice (random = 25%). Useful signal, far from reliable tagging.
- **One-shots are much harder than loops — universally.** Every one of the 15 models
  scores higher on loops; the gap is **15–29 points** (best model: 69.5% loops vs 42.5%
  one-shots). A single kick or FX hit carries little genre.
- **Only house and techno are actually recognised** (up to 91% / 84%). **dnb and
  garage_dubstep collapse into techno** — they are never reliably found by any model.
  This is *not* a missing-label or mapping problem (checked — see below): the models
  dump these clips into `electronic` / `ambient` and never fire `drumnbass`/`dubstep`.
- **MTG-Jamendo + specialised Discogs embeddings wins** (56.0%); the plain
  `discogs-effnet` embedding is 9 points worse (47.0%).
- **MAEST (transformer) does not help, and longer context actively hurts**: best MAEST
  45.5%, the 30 s model is the **worst overall (34.2%)** and degenerates into calling
  almost everything techno.

## Restricted 4-way accuracy — the headline

*Argmax over only the labels mapped to our 4 genres. Random = 25%.*

| model | backbone | overall | loop | one-shot | Δ |
|---|---|--:|--:|--:|--:|
| **jamendo-artist** | effnet | **56.0** | 68.0 | 44.0 | 24.0 |
| **jamendo-track** | effnet | **56.0** | 69.5 | 42.5 | 27.0 |
| jamendo-label | effnet | 54.8 | 67.0 | 42.5 | 24.5 |
| jamendo-release | effnet | 54.2 | 65.5 | 43.0 | 22.5 |
| jamendo-multi | effnet | 53.2 | 67.0 | 39.5 | 27.5 |
| jamendo-effnet | effnet | 47.0 | 56.0 | 38.0 | 18.0 |
| discogs400-maest-10s-dw | maest | 45.5 | 59.0 | 32.0 | 27.0 |
| discogs400-effnet | effnet | 44.0 | 51.5 | 36.5 | 15.0 |
| discogs519-maest-30s-pw | maest | 41.5 | 50.0 | 33.0 | 17.0 |
| discogs400-maest-30s-pw | maest | **34.2** *(worst)* | 45.0 | 23.5 | 21.5 |

(5 further Discogs400-MAEST variants score 37.5–43.8; see summary.md.)

- **Loops beat one-shots in all 15/15 models.** One-shot accuracy tops out at 44% and
  falls to 23.5% — barely above chance for the weakest models.
- **The embedding matters more than the taxonomy.** Swapping `discogs-effnet` for the
  artist/track/label/release embeddings lifts MTG-Jamendo from 47.0 → 56.0.

## Per-genre: only half the taxonomy works

*Recall, best model (jamendo-track) and a typical Discogs400 model.*

| genre | jamendo-track | discogs400-effnet | verdict |
|---|--:|--:|---|
| house | **91.0** | 61.0 | reliably found |
| techno | **82.0** | 66.0 | reliably found |
| dnb | 25.0 | 32.0 | **fails** |
| garage_dubstep | 26.0 | 17.0 | **fails** |

The confusion matrices show a consistent **techno magnet**: for jamendo-track, 62/100
garage_dubstep clips and 36/100 dnb clips are called techno. It is worst for
`discogs400-maest-30s-pw`, which calls **83% of house, 80% of dnb and 75% of
garage_dubstep "techno"** — 91% techno recall bought by collapsing everything else.
That is a degenerate model on this material, not a good techno detector.

## Restricted vs unrestricted: the models rarely *name* the genre

*top-1 = the model's single highest label over its full taxonomy; on-genre% = how often
that #1 label is any of our 4 genres at all.*

| model | restricted | top-1 | on-genre % |
|---|--:|--:|--:|
| jamendo-track | 56.0 | **7.8** | 7.8 |
| jamendo-artist | 56.0 | **2.2** | 2.8 |
| discogs400-maest-5s-pw | 43.8 | 32.5 | **59.2** |
| discogs400-effnet | 44.0 | 24.2 | 38.5 |

This is the most important caveat on the headline number. The MTG-Jamendo models look
best **only because the restricted view forces a 4-way choice**: left unrestricted their
top label is one of our 4 genres just **2–8%** of the time — they prefer some broader or
unrelated tag. The Discogs400-MAEST models are the opposite: less accurate when
restricted, but they actually commit to an on-genre label ~59% of the time. So "56%"
means *"given that it must pick one of these four, it picks right 56% of the time"* — not
that it would ever spontaneously label a clip "house".

## Is the dnb/garage failure just a bad label mapping? No.

The obvious objection: maybe those genres fail because the taxonomy lacks the labels,
or because our mapping is too narrow. Both were checked.

**The labels exist and are mapped**: Discogs400 has `Drum n Bass` + `Jungle` for dnb and
`Dubstep`, `UK Garage`, `Speed Garage` for garage; MTG-Jamendo has `drumnbass` and
`dubstep`. **The models simply don't fire them** — what they actually predict on these
clips (unrestricted top-1) is a broad bucket:

| true genre | jamendo-track predicts | discogs400-effnet predicts |
|---|---|---|
| house | 82× `electronic`, 18× **house** | 48× **House**, 16× Experimental |
| **dnb** | 75× `electronic`, 11× `ambient`, 10× `classical` | 26× `Ambient`, 19× `Experimental`, **12× Drum n Bass** |
| **garage_dubstep** | 77× `electronic`, 19× `ambient`, **2× dubstep** | 24× `Ambient`, 14× Techno, 7× `Grime` |

Mean activations tell the same story: house/techno mapped labels fire at **0.21–0.33**,
dnb/garage only at **0.03–0.14**. For jamendo-track, dnb's own labels (0.068) score
*below* techno's (0.074) on dnb clips — the model has no confident signal to map.

**Sensitivity check** ([results/mapping_sensitivity.md](results/mapping_sensitivity.md)):
re-scoring from cached vectors with an *expanded* mapping (`Halftime`→dnb, `Grime`→
garage_dubstep — the adjacent styles the models do reach for) moves overall accuracy by
**≤0.5 points**, dnb by at most +1 (and −3 in one model, as garage's new `Grime` label
steals dnb clips), garage by +1–2. All six MTG-Jamendo models are **unchanged** (no
equivalent tags exist). The conclusion is robust to the mapping: **this is genuine model
failure on this material, not a labelling artifact.**

## Recommendations

1. **Don't use these models to tag one-shots.** 23–44% on a 4-way choice is not usable;
   one-shots lack the rhythmic/harmonic context genre models rely on.
2. **For loops, use MTG-Jamendo on a specialised Discogs embedding** (`artist` or
   `track`, 68–69.5% on loops) — and treat the output as a weak prior, not a label.
3. **Don't trust dnb / garage_dubstep predictions at all** — they are absorbed into
   techno by every model.
4. **Skip MAEST here.** It is slower, needs 5–30 s of context these clips don't have, and
   is beaten by a plain CNN embedding. Longer context = worse.
5. **Next**: if genre tagging of sample packs matters, fine-tune on loop-length audio —
   off-the-shelf full-track models transfer poorly to this material.

## Caveats

- **Short clips are tiled (looped) to satisfy model minimums**: EffNet emits no patches
  below ~2 s; MAEST errors below its 5–30 s context. All clips are tiled up to 3 s
  (EffNet) or context+2 s (MAEST). For a 0.3 s hi-hat this manufactures an artificial
  rhythm — the one-shot numbers, especially for long-context MAEST, must be read with
  this in mind. Without it those models cannot process one-shots at all.
- **Ground truth is pack provenance, not perceived genre.** A piano or vocal loop from a
  dnb pack is labelled dnb even if it sounds genre-neutral — part of why dnb scores low.
  Ambiguous-type folders (FX, atmos, pads, textures, vocals) were excluded.
- **MAEST models use their native built-in Discogs400/519 classifier** (the separate
  `-maest` head expects an intermediate representation Essentia doesn't expose).
- 400 clips / 100 per genre: differences of a few points are not meaningful.

## Reproduce
`scripts/build_ground_truth.py` → `scripts/download_assets.sh` → `sbatch/run_all.sbatch`
(classify → score → report). See [README.md](README.md).