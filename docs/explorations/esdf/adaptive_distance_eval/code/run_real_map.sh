#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../../../.."
output=${1:-outputs/adaptive_distance_eval/real-map-reproduce}
map=${2:-outputs/adaptive_distance_eval/real-map-assets/pillars-23.obj}
full_error=${ADAPTIVE_SDF_FULL_ERROR_M:-.002}
full_depth=${ADAPTIVE_SDF_FULL_MAX_DEPTH:-9}
image=tof-tracker-adaptive-sdf-real-map:torch291
if [[ -e "$output" ]]; then
    printf '%s\n' "Output already exists: $output" >&2
    exit 2
fi
docker build -f docs/explorations/esdf/adaptive_distance_eval/code/Dockerfile -t tof-tracker-adaptive-sdf:8db373e-torch291 .
docker build -f docs/explorations/esdf/adaptive_distance_eval/code/Dockerfile.real_map -t "$image" .
image_id=$(docker image inspect "$image" --format '{{.Id}}')
source_revision=$(git rev-parse HEAD)
docker run --rm --gpus all --network none --memory 14g --memory-swap 20g --cpus 4 \
    --user "$(id -u):$(id -g)" --env "ADAPTIVE_SDF_IMAGE_ID=$image_id" \
    --env "ADAPTIVE_SDF_SOURCE_REVISION=$source_revision" \
    --env PYTHONPYCACHEPREFIX=/tmp/adaptive-real-map-pycache --env TORCHINDUCTOR_COMPILE_THREADS=2 \
    --mount "type=bind,source=$(pwd),target=/workspace" "$image" bash -c '
set -euo pipefail
output=$1
map=$2
full_error=$3
full_depth=$4
if [[ ! -f "$map" ]]; then
    mkdir -p "$(dirname "$map")"
    cmake -S tools/mapgen -B outputs/adaptive_distance_eval/mapgen-build -DCMAKE_BUILD_TYPE=Release
    cmake --build outputs/adaptive_distance_eval/mapgen-build --parallel 2
    outputs/adaptive_distance_eval/mapgen-build/offline_mapgen --type pillars --seed 23 \
        --size-x 60 --size-y 60 --size-z 15 --count 500 --width-min .3 --width-max 1 --output "$map"
fi
mkdir -p outputs/adaptive_distance_eval/vendor
task_build=$(mktemp -d outputs/adaptive_distance_eval/vendor/real-map-XXXXXX)
python -m docs.explorations.esdf.adaptive_distance_eval.code.nodal_patch --source /opt/SdfLib \
    --destination "$task_build/SdfLib" --node-key-step-m .0125 --robust-mesh-values
cmake -S docs/explorations/esdf/adaptive_distance_eval/code -B "$task_build/build" \
    -DSDFLIB_SOURCE_DIR="/workspace/$task_build/SdfLib" -DCMAKE_BUILD_TYPE=Release
cmake --build "$task_build/build" --parallel 2
python -m docs.explorations.esdf.adaptive_distance_eval.code.real_map_evaluate --stage prepare --map "$map" --output "$output" \
    --adaptation-manifest "$task_build/SdfLib/ADAPTATION.json" --full-termination-error-m "$full_error" --full-max-depth "$full_depth"
python -m docs.explorations.esdf.adaptive_distance_eval.code.raw_value_checks --binary "$task_build/build/mesh_value_probe" \
    --union-mesh "$output/union.obj" --source-map "$output/source-map.obj" --output "$output/raw-value-checks"
python -m docs.explorations.esdf.adaptive_distance_eval.code.real_map_evaluate --stage build \
    --binary "$task_build/build/export_sdflib" --output "$output"
python -m docs.explorations.esdf.adaptive_distance_eval.code.real_map_evaluate --stage evaluate --output "$output"
python -m docs.explorations.esdf.adaptive_distance_eval.code.real_map_plot --output "$output"
' -- "$output" "$map" "$full_error" "$full_depth"
