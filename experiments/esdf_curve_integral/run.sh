#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
image=${CURVE_INTEGRAL_IMAGE:-tof-tracker-main-sync-d9fc:torch291}
output=${1:-outputs/esdf_curve_integral/real-map-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ "$output" != outputs/* || -e "$repo/$output" ]]; then
  echo "use a new repository-relative output under outputs/" >&2
  exit 2
fi
mkdir -p "$repo/$output"
docker image inspect --format '{{.RepoTags}} {{.Id}}' "$image" > "$repo/$output/images.txt"
docker run --rm --gpus all --network none --memory 4g --memory-swap 4g --cpus 2 \
  --user "$(id -u):$(id -g)" --env PYTHONPYCACHEPREFIX=/tmp/esdf-curve-integral-pycache \
  --mount "type=bind,source=$repo,target=/workspace" --workdir /workspace "$image" \
  python -m experiments.esdf_curve_integral.evaluate --output "$output/comparison" \
  --grid "${CURVE_INTEGRAL_GRID:-outputs/esdf/full-0p2-20261018}" \
  --source-raw "${CURVE_INTEGRAL_SOURCE:-outputs/esdf_reference_eval/integrated-20261018-v2/comparison/raw.npz}" \
  --device cuda --repeats "${CURVE_INTEGRAL_REPEATS:-5}" 2>&1 | tee "$repo/$output/evaluate.log"
