#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
image=${ESDF_EXPERIMENT_IMAGE:-tof-tracker-adaptive-sdf-real-map:torch291}
output=${1:-outputs/esdf_grid_eval/path-integrals-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ "$output" != outputs/* ]]; then
  echo "output must be a repository-relative path under outputs/" >&2
  exit 2
fi
mkdir -p "$repo/$output"
if [[ -e "$repo/$output/run.log" || -e "$repo/$output/results.json" ]]; then
  echo "use a fresh output directory" >&2
  exit 2
fi
image_id=$(docker image inspect --format '{{.Id}}' "$image")
docker run --rm --network none --cpus 2 --memory 4g --memory-swap 4g \
  --user "$(id -u):$(id -g)" --mount "type=bind,source=$repo,target=/workspace" \
  --workdir /workspace --env PYTHONPYCACHEPREFIX=/tmp/esdf-grid-pycache \
  --env "ESDF_EXPERIMENT_IMAGE_ID=$image_id" "$image" \
  python -m docs.explorations.esdf.esdf_grid_eval.code.evaluate --output "$output" 2>&1 | tee "$repo/$output/run.log"
