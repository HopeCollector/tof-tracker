#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="${TOF_TRACKER_DEV_IMAGE:-tof-tracker-dev:ubuntu24.04}"
docker_cli=(docker --host unix:///var/run/docker.sock)

usage() {
    echo "Usage: $0 build | run [--] [command [args...]]" >&2
}

# Docker chooses the host port. Announce it only after an HTTP service responds.
# The foreground process stays `docker run`, retaining its signal/exit handling.
announce_web() {
    local run_pid="$1" run_directory="$2" container_id address="" response
    trap 'rm -rf -- "$run_directory"' EXIT
    while kill -0 "$run_pid" 2>/dev/null; do
        if [[ -z "$address" && -s "$run_directory/id" ]]; then
            container_id="$(<"$run_directory/id")"
            address="$("${docker_cli[@]}" port "$container_id" 8080/tcp 2>/dev/null)" || return 0
        fi
        if [[ "$address" == 127.0.0.1:* ]] && (
            exec 3<>"/dev/tcp/127.0.0.1/${address##*:}" &&
            printf 'HEAD / HTTP/1.0\r\nHost: localhost\r\n\r\n' >&3 &&
            IFS= read -r -t 1 response <&3 &&
            [[ "$response" == HTTP/* ]]
        ) 2>/dev/null; then
            printf '\n浏览器访问 http://%s/ （Ctrl+C 退出）\n' "$address"
            return 0
        fi
        sleep .2
    done
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
            --gpus all --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
            --publish 127.0.0.1::8080 \
            --workdir /workspace \
            --mount "type=bind,source=$project_root,target=/workspace")
        if [[ -t 0 ]]; then run_args+=(--interactive); fi
        if [[ -t 0 && -t 1 ]]; then run_args+=(--tty); fi
        if [[ "${1:-}" == -- ]]; then shift; fi
        if [[ $# -eq 0 ]]; then set -- bash; fi
        run_directory="$(mktemp -d "${TMPDIR:-/tmp}/tof-tracker-run.XXXXXXXX")"
        announce_web "$$" "$run_directory" &
        run_args+=(--cidfile "$run_directory/id")
        exec "${docker_cli[@]}" run "${run_args[@]}" "$image" "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
