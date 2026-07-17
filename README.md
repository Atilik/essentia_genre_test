# essentia_genre_test — can off-the-shelf Essentia genre models classify loops & one-shots?

Off-the-shelf [Essentia](https://essentia.upf.edu/models.html) genre classifiers are
trained on **full tracks**. This benchmark asks whether they still work on short,
often genre-ambiguous **sample-library material** — loops and one-shots pulled from
commercial packs — and whether one-shots (single hits, FX) are hopeless vs loops.

## Task
- **4 genres** (ground truth = source sample pack): `house`, `techno`, `dnb`,
  `garage_dubstep`.
- **400 clips**: 50 loops + 50 one-shots per genre (balanced), clip-type read from
  each pack's subfolder names ("Drum Loops" vs "Drum Oneshots"/"One Shots"…).
- Two scoring views:
  - **restricted 4-way argmax** (primary): argmax over only the labels mapped to the
    4 genres → clean 4×4 confusion. Random = **25%**.
  - **unrestricted top-1** (secondary): the model's single highest label over the full
    taxonomy, mapped to a genre or "other" — does an on-genre label even win?
- Headline: **loop vs one-shot accuracy**, split per genre.

## Models (15 versions across 3 families)
Each is `audio → embedding backbone → genre`. Two backbone types:
- **Discogs-EffNet** (CNN): 1× Genre Discogs400 + 6× MTG-Jamendo (different Discogs
  embedding flavors) → separate `TensorflowPredict2D` head.
- **MAEST** (transformer, 5/10/20/30 s context): 7× Genre Discogs400 + 1× Genre
  Discogs519. MAEST carries the Discogs classifier internally, so we use its **native
  output** (the separate `-maest` head expects an intermediate we don't extract).

See `scripts/models.py` for the full manifest.

## Pipeline
```bash
python scripts/build_ground_truth.py          # SMALL_SAMPLES.zip -> data/{ground_truth,eval_set}.csv + data/audio/
bash   scripts/download_assets.sh              # 15 heads + label JSONs -> models/heads/
python scripts/classify.py --model <name>      # -> results/scores/<name>.npy   (GPU)
python scripts/score.py    --model <name>      # -> results/<name>.json
python scripts/report.py                       # -> results/summary.md
```
`classify.py` caches raw class scores as `.npy`, so scoring/label-mapping can be
re-run without re-inference. Needs `essentia-tensorflow`; embedding backbones are
expected under `models/embeddings/`. Add `--limit N` to `classify.py` for a smoke test.

## Caveats
- **MAEST on short clips**: clips shorter than a model's context (5–30 s) are **tiled
  (looped)** up to length. For sub-second one-shots the 20/30 s models mostly see a
  repeated fragment — interpret their one-shot numbers with care.
- **Genre↔content confound**: e.g. a bass/vocal loop from the dnb pack is labelled dnb
  regardless of how "dnb" it sounds. We report loop vs one-shot separately for this
  reason. Ambiguous-type folders (FX, atmos, pads, textures, vocals) are excluded.
- Ground truth is **pack provenance**, a proxy for genre — good for relative model
  comparison, not an absolute accuracy claim.

Results + discussion: **[FINDINGS.md](FINDINGS.md)** (numbers in `results/summary.md`).
