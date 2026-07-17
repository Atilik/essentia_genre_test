"""
Map each model's large output taxonomy down to our 4 ground-truth genres.

Ground truth = which sample pack a clip came from: house / techno / dnb /
garage_dubstep. To score, we need, per taxonomy, the set of output labels that
correspond to each genre. Rules below are deliberately explicit and easy to edit;
`genre_label_indices()` materialises them against a model's actual class list and
asserts every genre is covered.

Discogs style labels look like "Electronic---Deep House". MTG-Jamendo tags are flat.
"""
GENRES = ["house", "techno", "dnb", "garage_dubstep"]


def _discogs_genre(label):
    """One Discogs style label -> genre or None. Precedence avoids overlaps
    (e.g. 'Garage House' -> house, 'UK Garage' -> garage_dubstep)."""
    c = label.lower()
    if "house" in c:                                   # incl. Garage House, Hip-House
        return "house"
    if "techno" in c:
        return "techno"
    if "drum n bass" in c or "jungle" in c:
        return "dnb"
    if "dubstep" in c or "uk garage" in c or "speed garage" in c:
        return "garage_dubstep"
    return None                                        # breakbeat/grime/etc. -> unmapped


# MTG-Jamendo (87 flat tags): explicit — no 'garage' tag exists, dubstep only.
_MTG = {
    "house": ["house", "deephouse"],
    "techno": ["techno"],
    "dnb": ["drumnbass"],
    "garage_dubstep": ["dubstep"],
}


def genre_labels(taxonomy, classes):
    """taxonomy in {discogs400, discogs519, mtg_jamendo} -> {genre: [labels]}."""
    out = {g: [] for g in GENRES}
    if taxonomy in ("discogs400", "discogs519"):
        for c in classes:
            g = _discogs_genre(c)
            if g:
                out[g].append(c)
    elif taxonomy == "mtg_jamendo":
        cset = set(classes)
        for g, tags in _MTG.items():
            out[g] = [t for t in tags if t in cset]
    else:
        raise ValueError(f"unknown taxonomy {taxonomy!r}")
    missing = [g for g in GENRES if not out[g]]
    if missing:
        raise AssertionError(f"{taxonomy}: no labels for {missing}")
    return out


def genre_label_indices(taxonomy, classes):
    """{genre: [class indices]} for restricted-argmax scoring."""
    idx = {c: i for i, c in enumerate(classes)}
    return {g: [idx[l] for l in labels]
            for g, labels in genre_labels(taxonomy, classes).items()}


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.models import MODELS
    seen = set()
    for m in MODELS:
        if m["taxonomy"] in seen:
            continue
        seen.add(m["taxonomy"])
        classes = json.load(open(m["head_json"]))["classes"]
        gl = genre_labels(m["taxonomy"], classes)
        print(f"\n=== {m['taxonomy']} ({len(classes)} classes) ===")
        for g in GENRES:
            print(f"  {g:<15} ({len(gl[g])}): {gl[g]}")
