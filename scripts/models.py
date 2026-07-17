"""
Manifest of the 15 off-the-shelf Essentia genre models under test.

Each model is a two-stage pipeline: an audio embedding backbone (already staged
locally under models/embeddings/) feeding a downloaded classifier head. We store,
per model, the head files to download and the local embedding .pb to reuse. Exact
TensorFlow node names + class labels are read at run time from the metadata JSONs
(see classify.py), so nothing fragile is hard-coded here.

Taxonomies: discogs400 (400 styles), discogs519 (519 styles), mtg_jamendo (87 tags).
Embedding algos: "effnet" -> TensorflowPredictEffnetDiscogs, "maest" -> TensorflowPredictMAEST.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADS_DIR = os.path.join(ROOT, "models", "heads")          # downloaded here
EMB_DIR = os.path.join(ROOT, "models", "embeddings")        # symlink to staged weights

HEAD_BASE = "https://essentia.upf.edu/models/classification-heads"
EMB_BASE = "https://essentia.upf.edu/models/feature-extractors"

# embedding backbone -> (local .pb relative to EMB_DIR, algo, metadata-json URL)
EMB = {
    "discogs-effnet-bs64": ("discogs_effnet/discogs-effnet-bs64-1.pb", "effnet",
        f"{EMB_BASE}/discogs-effnet/discogs-effnet-bs64-1.json"),
    "discogs_artist": ("discogs_effnet/discogs_artist_embeddings-effnet-bs64-1.pb", "effnet",
        f"{EMB_BASE}/discogs-effnet/discogs_artist_embeddings-effnet-bs64-1.json"),
    "discogs_label": ("discogs_effnet/discogs_label_embeddings-effnet-bs64-1.pb", "effnet",
        f"{EMB_BASE}/discogs-effnet/discogs_label_embeddings-effnet-bs64-1.json"),
    "discogs_multi": ("discogs_effnet/discogs_multi_embeddings-effnet-bs64-1.pb", "effnet",
        f"{EMB_BASE}/discogs-effnet/discogs_multi_embeddings-effnet-bs64-1.json"),
    "discogs_release": ("discogs_effnet/discogs_release_embeddings-effnet-bs64-1.pb", "effnet",
        f"{EMB_BASE}/discogs-effnet/discogs_release_embeddings-effnet-bs64-1.json"),
    "discogs_track": ("discogs_effnet/discogs_track_embeddings-effnet-bs64-1.pb", "effnet",
        f"{EMB_BASE}/discogs-effnet/discogs_track_embeddings-effnet-bs64-1.json"),
    "maest-5s-pw": ("MAEST/discogs-maest-5s-pw-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-5s-pw-2.json"),
    "maest-10s-pw": ("MAEST/discogs-maest-10s-pw-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-10s-pw-2.json"),
    "maest-10s-fs": ("MAEST/discogs-maest-10s-fs-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-10s-fs-2.json"),
    "maest-10s-dw": ("MAEST/discogs-maest-10s-dw-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-10s-dw-2.json"),
    "maest-20s-pw": ("MAEST/discogs-maest-20s-pw-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-20s-pw-2.json"),
    "maest-30s-pw": ("MAEST/discogs-maest-30s-pw-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-30s-pw-2.json"),
    "maest-30s-pw-ts": ("MAEST/discogs-maest-30s-pw-ts-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-30s-pw-ts-2.json"),
    "maest-30s-pw-519l": ("MAEST/discogs-maest-30s-pw-519l-2.pb", "maest",
        f"{EMB_BASE}/maest/discogs-maest-30s-pw-519l-2.json"),
}


def _m(name, head_file, taxonomy, tax_dir, emb_key):
    return {
        "name": name,
        "taxonomy": taxonomy,
        "head_pb_url": f"{HEAD_BASE}/{tax_dir}/{head_file}.pb",
        "head_json_url": f"{HEAD_BASE}/{tax_dir}/{head_file}.json",
        "head_pb": os.path.join(HEADS_DIR, f"{head_file}.pb"),
        "head_json": os.path.join(HEADS_DIR, f"{head_file}.json"),
        "emb_key": emb_key,
        "emb_pb": os.path.join(EMB_DIR, EMB[emb_key][0]),
        "emb_algo": EMB[emb_key][1],
        "emb_json_url": EMB[emb_key][2],
        "emb_json": os.path.join(HEADS_DIR, EMB[emb_key][2].rsplit("/", 1)[1]),
    }


MODELS = [
    # --- Genre Discogs400 (8) ---
    _m("discogs400-effnet",        "genre_discogs400-discogs-effnet-1",        "discogs400", "genre_discogs400", "discogs-effnet-bs64"),
    _m("discogs400-maest-5s-pw",   "genre_discogs400-discogs-maest-5s-pw-1",   "discogs400", "genre_discogs400", "maest-5s-pw"),
    _m("discogs400-maest-10s-pw",  "genre_discogs400-discogs-maest-10s-pw-1",  "discogs400", "genre_discogs400", "maest-10s-pw"),
    _m("discogs400-maest-10s-fs",  "genre_discogs400-discogs-maest-10s-fs-1",  "discogs400", "genre_discogs400", "maest-10s-fs"),
    _m("discogs400-maest-10s-dw",  "genre_discogs400-discogs-maest-10s-dw-1",  "discogs400", "genre_discogs400", "maest-10s-dw"),
    _m("discogs400-maest-20s-pw",  "genre_discogs400-discogs-maest-20s-pw-1",  "discogs400", "genre_discogs400", "maest-20s-pw"),
    _m("discogs400-maest-30s-pw",  "genre_discogs400-discogs-maest-30s-pw-1",  "discogs400", "genre_discogs400", "maest-30s-pw"),
    _m("discogs400-maest-30s-pw-ts","genre_discogs400-discogs-maest-30s-pw-ts-1","discogs400","genre_discogs400", "maest-30s-pw-ts"),
    # --- Genre Discogs519 (1) ---
    _m("discogs519-maest-30s-pw",  "genre_discogs519-discogs-maest-30s-pw-519l-1", "discogs519", "genre_discogs519", "maest-30s-pw-519l"),
    # --- MTG-Jamendo genre (6) ---
    _m("jamendo-effnet",           "mtg_jamendo_genre-discogs-effnet-1",                    "mtg_jamendo", "mtg_jamendo_genre", "discogs-effnet-bs64"),
    _m("jamendo-artist",           "mtg_jamendo_genre-discogs_artist_embeddings-effnet-1",  "mtg_jamendo", "mtg_jamendo_genre", "discogs_artist"),
    _m("jamendo-label",            "mtg_jamendo_genre-discogs_label_embeddings-effnet-1",   "mtg_jamendo", "mtg_jamendo_genre", "discogs_label"),
    _m("jamendo-multi",            "mtg_jamendo_genre-discogs_multi_embeddings-effnet-1",   "mtg_jamendo", "mtg_jamendo_genre", "discogs_multi"),
    _m("jamendo-release",          "mtg_jamendo_genre-discogs_release_embeddings-effnet-1", "mtg_jamendo", "mtg_jamendo_genre", "discogs_release"),
    _m("jamendo-track",            "mtg_jamendo_genre-discogs_track_embeddings-effnet-1",   "mtg_jamendo", "mtg_jamendo_genre", "discogs_track"),
]

BY_NAME = {m["name"]: m for m in MODELS}


def model_classes(m):
    """Class-label list matching the scores classify.py produces: MAEST models
    emit their native classifier output (emb JSON classes); EffNet models use the
    downloaded head's classes."""
    import json
    key = "emb_json" if m["emb_algo"] == "maest" else "head_json"
    return json.load(open(m[key]))["classes"]


if __name__ == "__main__":
    print(f"{len(MODELS)} models across taxonomies:",
          {t: sum(1 for m in MODELS if m['taxonomy'] == t) for t in ('discogs400','discogs519','mtg_jamendo')})
    for m in MODELS:
        staged = "OK" if os.path.exists(m["emb_pb"]) else "MISSING"
        print(f"  {m['name']:<24} {m['emb_algo']:<6} emb={staged:<8} {os.path.basename(m['emb_pb'])}")
