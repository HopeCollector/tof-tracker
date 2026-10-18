#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
build_image=${REFERENCE_BUILD_IMAGE:-tof-tracker-esdf-eval:nvblox-24eee49-cu128}
gpu_image=${REFERENCE_GPU_IMAGE:-tof-tracker-main-sync-d9fc:torch291}
map=${REFERENCE_MAP:-docs/explorations/esdf/artifacts/outputs/mesh_distance_eval/reviewed-20261015/pillar_omni.obj}
grid=${REFERENCE_GRID:-docs/explorations/esdf/artifacts/outputs/esdf/full-0p2-20261018}
output=${1:-outputs/esdf_reference_eval/comparison-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ "$output" != outputs/* || -e "$repo/$output" ]]; then
  echo "use a new repository-relative output under outputs/" >&2
  exit 2
fi
mkdir -p "$repo/$output"
docker image inspect --format '{{.RepoTags}} {{.Id}}' "$build_image" "$gpu_image" > "$repo/$output/images.txt"
common=(--rm --network none --memory 4g --memory-swap 4g --cpus 2
  --user "$(id -u):$(id -g)" --env PYTHONPYCACHEPREFIX=/tmp/esdf-reference-pycache
  --mount "type=bind,source=$repo,target=/workspace" --workdir /workspace)
docker run "${common[@]}" "$build_image" python -m docs.explorations.esdf.esdf_reference_eval.code.build_maps \
  --map "$map" --output "$output/reference-map" 2>&1 | tee "$repo/$output/build.log"
docker run "${common[@]}" --gpus all "$gpu_image" python -m docs.explorations.esdf.esdf_reference_eval.code.evaluate \
  --map "$map" --grid "$grid" --reference-map "$output/reference-map" \
  --output "$output/comparison" --device cuda 2>&1 | tee "$repo/$output/evaluate.log"
