#!/bin/bash
# Download the 15 classifier heads (.pb) + their metadata/label JSONs, and the
# embedding-backbone metadata JSONs (for exact TF node names). The embedding .pb
# graphs are NOT downloaded here — they are already staged under models/embeddings/.
# Heads are small MLPs + tiny JSONs, so this is safe to run on the login node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p models/heads

# enumerate (url dest) pairs from the manifest so there is a single source of truth
python3 - <<'PY' > /tmp/essentia_dl_list.txt
from scripts.models import MODELS
seen = set()
for m in MODELS:
    print(m["head_pb_url"], m["head_pb"])
    print(m["head_json_url"], m["head_json"])
    if m["emb_json_url"] not in seen:          # embedding JSONs shared across heads
        seen.add(m["emb_json_url"])
        print(m["emb_json_url"], m["emb_json"])
PY

n=0; fail=0
while read -r url dest; do
    [ -z "$url" ] && continue
    if [ -s "$dest" ]; then echo "cached  $(basename "$dest")"; continue; fi
    if curl -fsSL --max-time 120 -o "$dest" "$url"; then
        printf "ok      %s (%s)\n" "$(basename "$dest")" "$(du -h "$dest" | cut -f1)"
        n=$((n+1))
    else
        echo "FAIL    $url"; fail=$((fail+1))
    fi
done < /tmp/essentia_dl_list.txt

echo "---"
echo "downloaded $n new file(s), $fail failure(s)"
echo "heads dir: $(ls models/heads/*.pb 2>/dev/null | wc -l) .pb, $(ls models/heads/*.json 2>/dev/null | wc -l) .json"
[ "$fail" -eq 0 ]
