"""
Label mapping for the beats zero-shot view: Discogs400 / MTG-Jamendo output labels
-> the 23 Beatport subgenres, and a coarse super-genre grouping.

Uses EXACT label membership (not substring) to avoid overlap — e.g. "Progressive
House" must go to ProgressiveHouse, not House. Two Beatport genres have no Discogs
style (BigRoom, FutureHouse) -> left unmapped; reported as coverage.
"""
BEATS_GENRES = ["BigRoom", "Breaks", "Dance", "DeepHouse", "DrumAndBass", "Dubstep",
                "ElectroHouse", "ElectronicaDowntempo", "FunkRAndB", "FutureHouse",
                "GlitchHop", "HardDance", "HardcoreHardTechno", "HipHop", "House",
                "IndieDanceNuDisco", "Minimal", "ProgressiveHouse", "PsyTrance",
                "ReggaeDub", "TechHouse", "Techno", "Trance"]

E = "Electronic---"
# Beatport subgenre -> exact Discogs400/519 labels (only present ones are kept).
DISCOGS = {
    "BigRoom": [],                                        # no Discogs style
    "Breaks": [E+"Breaks", E+"Breakbeat", E+"Progressive Breaks", E+"Big Beat", E+"Broken Beat"],
    "Dance": [E+"Eurodance", E+"Dance-pop", E+"Italodance", E+"Hands Up"],
    "DeepHouse": [E+"Deep House"],
    "DrumAndBass": [E+"Drum n Bass", E+"Jungle"],
    "Dubstep": [E+"Dubstep"],
    "ElectroHouse": [E+"Electro House"],
    "ElectronicaDowntempo": [E+"Downtempo", E+"Trip Hop", E+"Ambient", E+"IDM", E+"Leftfield", E+"Chillwave"],
    "FunkRAndB": ["Funk / Soul---Funk", "Funk / Soul---Soul", "Funk / Soul---Rhythm & Blues",
                  "Funk / Soul---Contemporary R&B", "Funk / Soul---Neo Soul", "Funk / Soul---Boogie",
                  "Hip Hop---RnB/Swing"],
    "FutureHouse": [],                                   # no Discogs style
    "GlitchHop": [E+"Glitch"],
    "HardDance": [E+"Hard House", E+"Hard Trance", E+"Hardstyle", E+"Hands Up", E+"Jumpstyle", E+"Makina", E+"Donk"],
    "HardcoreHardTechno": [E+"Hardcore", E+"Hard Techno", E+"Gabber", E+"Schranz", E+"Speedcore", E+"Happy Hardcore"],
    "HipHop": [c for c in ()],                            # filled below (all Hip Hop---*)
    "House": [E+"House", E+"Acid House", E+"Garage House", E+"Ghetto House", E+"Italo House",
              E+"Tribal House", E+"Hip-House", E+"Euro House", E+"Tropical House"],
    "IndieDanceNuDisco": [E+"Nu-Disco", E+"Disco", E+"Italo-Disco", E+"Euro-Disco", E+"Synthwave", "Funk / Soul---Disco"],
    "Minimal": [E+"Minimal", E+"Minimal Techno"],
    "ProgressiveHouse": [E+"Progressive House"],
    "PsyTrance": [E+"Psy-Trance", E+"Goa Trance"],
    "ReggaeDub": ["Reggae---Dub", "Reggae---Reggae", "Reggae---Dancehall", "Reggae---Roots Reggae",
                  "Reggae---Ragga", E+"Dub"],
    "TechHouse": [E+"Tech House"],
    "Techno": [E+"Techno", E+"Deep Techno", E+"Dub Techno"],
    "Trance": [E+"Trance", E+"Progressive Trance", E+"Tech Trance"],
}

# MTG-Jamendo 87 flat tags -> Beatport subgenre (only where a tag exists).
MTG = {
    "Breaks": ["breakbeat"], "Dance": ["dance", "eurodance"], "DeepHouse": ["deephouse"],
    "DrumAndBass": ["drumnbass"], "Dubstep": ["dubstep"], "ElectronicaDowntempo": ["downtempo", "triphop", "idm", "chillout"],
    "FunkRAndB": ["funk"], "HipHop": ["hiphop", "rap"], "House": ["house"], "IndieDanceNuDisco": ["disco"],
    "Minimal": ["minimal"], "PsyTrance": ["psychedelic"], "ReggaeDub": ["reggae", "dub"],
    "Techno": ["techno"], "Trance": ["trance"],
}

# coarse super-genre families (both GT and predictions collapse to these)
FAMILY = {
    "House": "house", "DeepHouse": "house", "TechHouse": "house", "ElectroHouse": "house",
    "ProgressiveHouse": "house", "FutureHouse": "house", "BigRoom": "house", "IndieDanceNuDisco": "house",
    "Techno": "techno", "Minimal": "techno", "HardcoreHardTechno": "techno",
    "Trance": "trance", "PsyTrance": "trance", "HardDance": "trance",
    "DrumAndBass": "dnb_breaks", "Breaks": "dnb_breaks",
    "Dubstep": "dubstep_glitch", "GlitchHop": "dubstep_glitch",
    "HipHop": "hiphop", "FunkRAndB": "hiphop",
    "ElectronicaDowntempo": "downtempo", "Dance": "downtempo", "ReggaeDub": "downtempo",
}


def _fill_hiphop():
    # all Hip Hop---* labels map to HipHop (kept generic; filtered to present at use time)
    DISCOGS["HipHop"] = ["Hip Hop---" + s for s in
        ["Boom Bap", "Trap", "Gangsta", "Conscious", "Pop Rap", "Cloud Rap", "G-Funk",
         "Jazzy Hip-Hop", "Instrumental", "Hardcore Hip-Hop", "Crunk", "Bounce",
         "Miami Bass", "Trip Hop", "Thug Rap"]] + [E+"Hip Hop"]


_fill_hiphop()


def mapped_indices(taxonomy, classes):
    """{genre: [class indices present]} for the given taxonomy; only genres with >=1
    present label are included."""
    table = DISCOGS if taxonomy.startswith("discogs") else MTG
    idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for g, labels in table.items():
        present = [idx[l] for l in labels if l in idx]
        if present:
            out[g] = present
    return out
