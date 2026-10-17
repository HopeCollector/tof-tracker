#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${TOF_TRACKER_DEV_IMAGE:-tof-tracker-dev:ubuntu24.04}"
docker_cli=(docker --host unix:///var/run/docker.sock)

usage() {
    echo "Usage: $0 build | run [--web] [--gpu] [--] [command [args...]]" >&2
}

case "${1:-}" in
    build)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        exec "${docker_cli[@]}" build \
            --file "$project_root/docker/Dockerfile" \
            --tag "$image" "$project_root"
        ;;
    run)
        shift
        run_args=(--rm --init --user "$(id -u):$(id -g)" \
            --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/.cache \
            --workdir /workspace \
            --mount "type=bind,source=$project_root,target=/workspace")
        if [[ -t 0 ]]; then run_args+=(--interactive); fi
        if [[ -t 0 && -t 1 ]]; then run_args+=(--tty); fi
        while [[ "${1:-}" == --web || "${1:-}" == --gpu ]]; do
            case "$1" in
                --web) run_args+=(--publish 127.0.0.1:8080:8080) ;;
                --gpu) run_args+=(--gpus all --env NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility) ;;
            esac
            shift
        done
        if [[ "${1:-}" == -- ]]; then shift; fi
        if [[ $# -eq 0 ]]; then set -- bash; fi
        exec "${docker_cli[@]}" run "${run_args[@]}" "$image" "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
