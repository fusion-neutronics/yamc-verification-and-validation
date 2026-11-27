#!/usr/bin/env bash
# Build the benchmark database site exactly as the Pages CI does and serve it
# locally, so what you see is byte-for-byte what deploys. Activate the venv first
# (the index builder needs numpy/pandas/plotly): source .venv/bin/activate
#
#   ./serve_site.sh            # build + serve on http://localhost:8321/
#   PORT=9000 ./serve_site.sh  # pick a port
#
# It shows the runs published.json names, as CI does, so records that are not
# published do not appear: use `vv compare` to see those.
set -euo pipefail

OUT=_site
PORT=${PORT:-8321}

rm -rf "$OUT" && mkdir -p "$OUT"
cp web/benchmarks.html "$OUT/index.html"
cp web/logo.png "$OUT/logo.png"
cp -r web/models "$OUT/models"
python -m yamc_verify.index_builder "$OUT"

echo "serving $OUT at http://localhost:${PORT}/  (Ctrl-C to stop)"
python -m http.server "$PORT" --directory "$OUT"
