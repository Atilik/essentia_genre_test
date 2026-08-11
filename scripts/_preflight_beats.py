"""Preflight for the beats full-track experiment (run in the essentia overlay, GPU).
Verifies: (1) MonoLoader decodes mp3, (2) EffNet emb_model returns [frames,1280],
(3) sklearn availability. Extracts one mp3 from the zip itself — no dependency on
the full unzip."""
import os, sys, json, zipfile, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.models import BY_NAME

ZIP = "/scratch/mk9649/datasets/beatsdataset_yt.zip"

# --- grab one mp3 to a temp file ---
z = zipfile.ZipFile(ZIP)
mp3 = next(i.filename for i in z.infolist()
           if i.filename.lower().endswith(".mp3") and not i.filename.startswith("__MACOSX"))
tmp = os.path.join(tempfile.gettempdir(), "preflight_beats.mp3")
with z.open(mp3) as src, open(tmp, "wb") as out:
    out.write(src.read())
print(f"[1] test mp3: {os.path.basename(mp3)} -> {tmp} ({os.path.getsize(tmp)} bytes)")

import essentia.standard as es
audio = es.MonoLoader(filename=tmp, sampleRate=16000, resampleQuality=4)()
print(f"    MonoLoader mp3 OK: {len(audio)} samples ({len(audio)/16000:.1f}s), "
      f"nonzero={np.any(audio!=0)}")

# --- embedding shape from discogs_track ---
m = BY_NAME["jamendo-track"]      # emb_key discogs_track
emb_json = json.load(open(m["emb_json"]))
emb_out = next(o["name"] for o in emb_json["schema"]["outputs"]
               if (o.get("output_purpose") or "") == "embeddings")
print(f"[2] discogs_track emb node = {emb_out}  pb={os.path.basename(m['emb_pb'])}")
emb_model = es.TensorflowPredictEffnetDiscogs(graphFilename=m["emb_pb"], output=emb_out)
e = np.asarray(emb_model(audio))
print(f"    embedding shape = {e.shape}  (expect [frames,1280]); mean-pooled = {e.mean(0).shape}")

# --- sklearn ---
try:
    import sklearn
    print(f"[3] sklearn OK: {sklearn.__version__}")
except Exception as ex:
    print(f"[3] sklearn MISSING ({type(ex).__name__}) -> probe will need a /scratch venv")

os.remove(tmp)
print("PREFLIGHT DONE")
