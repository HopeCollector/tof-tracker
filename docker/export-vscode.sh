#!/usr/bin/env bash
# Export image dependencies and configure Pylance for local project sources.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="tof-tracker-dev:ubuntu24.04"
cache_parent="$project_root/.vscode"
cache_dir="$cache_parent/pylance-site-packages"
config_path="$cache_parent/pyrightconfig.json"
config_input=/dev/null
if [[ -f "$config_path" ]]; then config_input="$config_path"; fi
docker_cli=(docker --host unix:///var/run/docker.sock)

mkdir -p -- "$cache_parent"
staging_dir="$(mktemp -d "$cache_parent/.pylance-site-packages.XXXXXX")"
trap 'rm -rf -- "$staging_dir"' EXIT

config_writer='
import json, sys

text = sys.stdin.read()
config = json.loads(text) if text.strip() else {}
config["pythonVersion"] = f"{sys.version_info.major}.{sys.version_info.minor}"
paths = config.setdefault("extraPaths", [])
if not isinstance(paths, list):
    raise ValueError("pyrightconfig.json extraPaths must be an array")
# Editable installs contain container paths; resolve project packages locally.
paths[:] = ["../src", ".."] + [path for path in paths if path not in ("../src", "..")]
if "pylance-site-packages" not in paths:
    paths.append("pylance-site-packages")
print(json.dumps(config, indent=2))
'

"${docker_cli[@]}" run --rm --interactive --pull never --network none "$image" bash -euc '
    python -c "$1" > /tmp/pyrightconfig.json
    site_packages="$(python -c "import sysconfig; print(sysconfig.get_path(\"purelib\"))")"
    exec tar -C "$site_packages" --exclude=__pycache__ -cf - . -C /tmp pyrightconfig.json
' _ "$config_writer" < "$config_input" | tar -xf - -C "$staging_dir" --no-same-owner

# Keep the previous cache until the complete export succeeds.
if cmp -s "$staging_dir/pyrightconfig.json" "$config_path"; then
    rm -- "$staging_dir/pyrightconfig.json"
else
    mv -f -- "$staging_dir/pyrightconfig.json" "$config_path"
fi
rm -rf -- "$cache_dir"
mv -- "$staging_dir" "$cache_dir"
printf 'Pylance dependencies exported to %s\n' "$cache_dir"
printf 'Pylance target configuration synced to %s\n' "$config_path"
