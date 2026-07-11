#!/usr/bin/env bash
# Build the benchmark database site exactly as the Pages CI does and serve it
# locally, so what you see is byte-for-byte what deploys. Activate the venv first
# (the index builder needs numpy/pandas/plotly): source .venv/bin/activate
#
#   ./serve_site.sh            # build + serve on http://localhost:8321/
#   PORT=9000 ./serve_site.sh  # pick a port
#
# This uses the results cache in your working tree (tasks/*/cache/*.json). CI
# only sees committed files, so for the DEPLOYED site to match, that cache must
# be committed too.
set -euo pipefail

OUT=_site
PORT=${PORT:-8321}

rm -rf "$OUT" && mkdir -p "$OUT"
cp web/benchmarks.html "$OUT/index.html"
cp web/logo.png "$OUT/logo.png"
cp -r web/models "$OUT/models"
python -m yamc_verify.index_builder --out "$OUT/benchmarks-index.json"
# detail views lazy-load per-benchmark cache JSON
find tasks -path '*/cache/*' -name '*.json' -exec cp --parents -t "$OUT" {} +

echo "serving $OUT at http://localhost:${PORT}/  (Ctrl-C to stop)"
python -m http.server "$PORT" --directory "$OUT"
