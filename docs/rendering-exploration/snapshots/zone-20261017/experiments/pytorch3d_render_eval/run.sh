#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "$0")/../.." && pwd)
task_image=${PYTORCH3D_EVAL_IMAGE:-tof-tracker-pytorch3d-eval:torch2.7-cu128}
task_image_id=$(docker --host unix:///var/run/docker.sock image inspect --format '{{.Id}}' "$task_image")
task_output=${1:-outputs/pytorch3d_render_eval/run}
if (($#)); then shift; fi
case "$task_output" in outputs/*) ;; *) echo 'Output must be below outputs/' >&2; exit 2 ;; esac
mkdir -p "$task_root/$task_output"
docker --host unix:///var/run/docker.sock run --rm \
    --name "tof-pytorch3d-eval-$$" --gpus all --shm-size=2g \
    --user "$(id -u):$(id -g)" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    -e "PYTORCH3D_EVAL_IMAGE_ID=$task_image_id" \
    -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib-cache \
    --mount "type=bind,src=$task_root,dst=/workspace,readonly" \
    --mount "type=bind,src=$task_root/$task_output,dst=/workspace/$task_output" \
    --workdir /workspace "$task_image" \
    python experiments/pytorch3d_render_eval/run_eval.py --output "$task_output" "$@"
