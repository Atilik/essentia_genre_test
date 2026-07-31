# Padding control — tile (loop-repeat) vs zero-pad

Short clips must be extended to each backbone's minimum (EffNet ~2 s, MAEST 5–30 s).
The main run **tiles** (loops the clip, since loops are made to repeat). This control
re-runs 4 representative models with **zero-padding** (append silence) instead, to test
whether any conclusion is an artifact of tiling manufacturing a rhythm. Re-inference
(padding changes the audio); accuracy is restricted 4-way argmax.

| model | overall t→z | loop t→z | one-shot t→z | dnb 1-shot t→z | garage 1-shot t→z |
|---|--:|--:|--:|--:|--:|
| discogs400-effnet | 44→42 | 52→52 | 36→32 | 18→10 | 0→0 |
| jamendo-track | 56→52 | 70→70 | 42→36 | 8→2 | 6→10 |
| discogs400-maest-5s-pw | 44→42 | 56→56 | 32→27 | 10→10 | 2→2 |
| discogs400-maest-30s-pw | 34→36 | 45→44 | 24→27 | 2→2 | 0→2 |

*(t=tile, z=zero-pad; dnb/garage columns are ONE-SHOT accuracy — where padding bites hardest.)*

## Conclusions (all robust to the padding choice)

1. **Loops are unchanged** (52→52, 70→70, 56→56, 45→44) — loops rarely need padding, so
   the headline loop numbers do not depend on tile vs zero at all.
2. **Zero-padding does not rescue one-shots — it is slightly worse** (silence drowns the
   signal). The loop ≫ one-shot gap is real and, if anything, widens without tiling.
3. **The dnb / garage collapse persists** under zero-pad (dnb one-shot 2–10%, garage
   0–10%). It is **not** an artifact of tiling injecting a techno-like pulse.

So tiling was marginally generous to one-shots, not misleading, and every qualitative
finding survives the alternative padding. Reproduce:
`classify.py --pad zero --out results/scores_zero` then `compare_pad.py`.
