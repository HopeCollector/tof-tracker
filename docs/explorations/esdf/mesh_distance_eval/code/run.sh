#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
output=${1:-outputs/mesh_distance_eval/run}
if [[ $# -gt 0 ]]; then shift; fi
image=tof-tracker-mesh-eval:portable-warp1.15-cu128
docker build -f docs/explorations/esdf/mesh_distance_eval/code/Dockerfile.portable -t "$image" .
image_id=$(docker image inspect "$image" --format '{{.Id}}')
source_revision=$(git rev-parse HEAD)
host_gpu_status=$(nvidia-smi)
docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --env "MESH_EVAL_IMAGE_ID=$image_id" --env "MESH_EVAL_SOURCE_REVISION=$source_revision" \
  --env "MESH_EVAL_HOST_GPU_STATUS=$host_gpu_status" \
  --mount "type=bind,source=$(pwd),target=/workspace" "$image" bash -c '
set -euo pipefail
cmake -S tools/mapgen -B outputs/mesh_distance_eval/build-portable -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/mesh_distance_eval/build-portable --parallel 2
python -m docs.explorations.esdf.mesh_distance_eval.code.evaluate \
  --binary outputs/mesh_distance_eval/build-portable/offline_mapgen "$@"
' -- --output "$output" "$@"
