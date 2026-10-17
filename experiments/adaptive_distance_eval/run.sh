#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=${1:-outputs/adaptive_distance_eval/reproduce}
if [[ $# -gt 0 ]]; then shift; fi
image=tof-tracker-adaptive-sdf:8db373e-torch291
variant=${ADAPTIVE_SDF_VARIANT:-original}
if [[ "$variant" != original && "$variant" != shared-nodes ]]; then
    printf '%s\n' 'ADAPTIVE_SDF_VARIANT must be original or shared-nodes' >&2
    exit 2
fi
build_options=()
if [[ -n ${ADAPTIVE_SDF_BASE_IMAGE:-} ]]; then
    build_options+=(--build-arg "BASE_IMAGE=$ADAPTIVE_SDF_BASE_IMAGE")
fi
docker build "${build_options[@]}" -f experiments/adaptive_distance_eval/Dockerfile -t "$image" .
image_id=$(docker image inspect "$image" --format '{{.Id}}')
source_revision=$(git rev-parse HEAD)
docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
    --env "ADAPTIVE_SDF_IMAGE_ID=$image_id" \
    --env "ADAPTIVE_SDF_SOURCE_REVISION=$source_revision" \
    --env "ADAPTIVE_SDF_VARIANT=$variant" \
    --mount "type=bind,source=$(pwd),target=/workspace" "$image" bash -c '
set -euo pipefail
upstream=/opt/SdfLib
build_dir=outputs/adaptive_distance_eval/build
if [[ "$ADAPTIVE_SDF_VARIANT" == shared-nodes ]]; then
    mkdir -p outputs/adaptive_distance_eval/vendor
    task_vendor_dir=$(mktemp -d outputs/adaptive_distance_eval/vendor/shared-nodes-XXXXXX)
    python -m experiments.adaptive_distance_eval.nodal_patch \
        --source /opt/SdfLib --destination "$task_vendor_dir/SdfLib"
    upstream="/workspace/$task_vendor_dir/SdfLib"
    build_dir="$task_vendor_dir/build"
fi
cmake -S experiments/adaptive_distance_eval -B "$build_dir" \
    -DSDFLIB_SOURCE_DIR="$upstream" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel 2
python -m experiments.adaptive_distance_eval.evaluate \
    --binary "$build_dir/export_sdflib" --variant "$ADAPTIVE_SDF_VARIANT" "$@"
' -- --output "$output" "$@"
