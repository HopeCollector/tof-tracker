#!/usr/bin/env bash
# Export image dependencies for host-side Pylance without installing on the host.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="tof-tracker-dev:ubuntu24.04"
cache_parent="$project_root/.vscode"
cache_dir="$cache_parent/pylance-site-packages"
docker_cli=(docker --host unix:///var/run/docker.sock)

mkdir -p -- "$cache_parent"
staging_dir="$(mktemp -d "$cache_parent/.pylance-site-packages.XXXXXX")"
trap 'rm -rf -- "$staging_dir"' EXIT

"${docker_cli[@]}" run --rm --pull never --network none "$image" bash -euc '
    site_packages="$(python -c "import sysconfig; print(sysconfig.get_path(\"purelib\"))")"
    exec tar -C "$site_packages" --exclude=__pycache__ -cf - .
' | tar -xf - -C "$staging_dir" --no-same-owner

# Keep the previous cache until the complete export succeeds.
rm -rf -- "$cache_dir"
mv -- "$staging_dir" "$cache_dir"
printf 'Pylance dependencies exported to %s\n' "$cache_dir"
