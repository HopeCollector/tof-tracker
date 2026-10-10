#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="tof-tracker-dev:ubuntu24.04"
docker_cli=(docker --host unix:///var/run/docker.sock)

usage() {
    echo "Usage: $0 build | run [command [args...]] | view input.ply [viewer options...]" >&2
}

case "${1:-}" in
    build)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        exec "${docker_cli[@]}" build \
            --file "$project_root/docker/Dockerfile" \
            --tag "$image" "$project_root"
        ;;
    run|view)
        action="$1"
        shift
        run_args=(--rm --init --user "$(id -u):$(id -g)" \
            --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/.cache \
            --workdir /workspace \
            --mount "type=bind,source=$project_root,target=/workspace")
        if [[ -t 0 ]]; then run_args+=(--interactive); fi
        if [[ -t 0 && -t 1 ]]; then run_args+=(--tty); fi
        if [[ "$action" == view ]]; then
            [[ $# -gt 0 ]] || { usage; exit 2; }
            run_args+=(--publish 127.0.0.1:8080:8080)
            set -- python tools/map_viewer/view_map.py "$@" --serve --host 0.0.0.0 --port 8080
        elif [[ $# -eq 0 ]]; then
            set -- bash
        fi
        exec "${docker_cli[@]}" run "${run_args[@]}" "$image" "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
