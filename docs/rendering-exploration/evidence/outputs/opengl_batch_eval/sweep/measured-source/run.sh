#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "$0")/../.." && pwd)
task_image=${OPENGL_BATCH_EVAL_IMAGE:-tof-tracker-pytorch3d-eval:torch2.7-cu128}
task_image_id=$(docker --host unix:///var/run/docker.sock image inspect --format '{{.Id}}' "$task_image")
task_output=${1:-outputs/opengl_batch_eval/run}
if (($#)); then shift; fi
case "$task_output" in outputs/opengl_batch_eval/*) ;; *) echo 'Output must be below outputs/opengl_batch_eval/' >&2; exit 2 ;; esac
task_output_absolute=$(realpath -m "$task_root/$task_output")
case "$task_output_absolute" in "$task_root/outputs/opengl_batch_eval/"*) ;; *) echo 'Output path escapes outputs/opengl_batch_eval/' >&2; exit 2 ;; esac
mkdir -p "$task_output_absolute"
docker --host unix:///var/run/docker.sock run --rm \
    --name "tof-opengl-batch-eval-$$" --gpus all --shm-size=2g \
    --user "$(id -u):$(id -g)" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    -e "OPENGL_BATCH_EVAL_IMAGE_ID=$task_image_id" \
    -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib-cache \
    --mount "type=bind,src=$task_root,dst=/workspace,readonly" \
    --mount "type=bind,src=$task_output_absolute,dst=/workspace/$task_output" \
    --workdir /workspace "$task_image" \
    python experiments/opengl_batch_eval/run_eval.py --output "$task_output" "$@"
