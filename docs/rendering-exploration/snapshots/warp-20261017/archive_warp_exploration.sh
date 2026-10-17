#!/usr/bin/env bash
# Filesystem-only archival: no imports, package installation, tests, or GPU use.
set -euo pipefail
project_root="${1:-$(git rev-parse --show-toplevel)}"
cd "$project_root"
archive="docs/rendering-exploration"
snapshot="$archive/snapshots/warp-20261017"
manifest="$archive/warp-20261017.json"
index="$archive/warp-20261017.sha256"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
: > "$scratch/files.jsonl"
: > "$scratch/excluded.jsonl"
: > "$scratch/reused.jsonl"

entry() {
    local path="$1" source="$2" category="$3" output="$4" hash bytes
    hash="$(sha256sum "$path" | cut -d ' ' -f 1)"
    bytes="$(stat -c %s "$path")"
    jq -cn --arg path "${path#"$archive/"}" --arg source "$source" \
        --arg category "$category" --arg hash "$hash" --argjson bytes "$bytes" \
        '{path:$path,source_path:$source,category:$category,bytes:$bytes,sha256:$hash}' >> "$output"
}

copy_file() {
    local source="$1" destination="$2" category="$3"
    mkdir -p "$(dirname "$destination")"
    cp -p "$source" "$destination"
    cmp -s "$source" "$destination"
    entry "$destination" "$source" "$category" "$scratch/files.jsonl"
}

for directory in outputs/warp_ray_perf outputs/warp_postprocess_profile outputs/ray_unified outputs/ray_sensors; do
    while IFS= read -r -d '' source; do
        case "$source" in
            *.trace.json) reason="Large reproducible Chrome trace; profiler statistics and raw cell JSON retained" ;;
            outputs/ray_unified/example/one_frame.html) reason="Large generated offline visualization; compact arrays and summary retained" ;;
            outputs/ray_unified/before-worktree.tar) reason="Recovery tar kept locally; relevant pre-unified tests and review sources extracted below" ;;
            outputs/ray_unified/before.patch) reason="Workspace recovery patch kept locally; measured and delivered source snapshots retained" ;;
            *) copy_file "$source" "$archive/evidence/$source" "evidence"; continue ;;
        esac
        entry "$source" "$source" "$reason" "$scratch/excluded.jsonl"
    done < <(find "$directory" -type f -not -path '*/__pycache__/*' -print0 | sort -z)
done

for directory in experiments/warp_ray_perf experiments/warp_postprocess_profile experiments/unified_ray_perf; do
    if test -d "$directory"; then
        while IFS= read -r -d '' source; do
            copy_file "$source" "$snapshot/harnesses/$source" "harness"
        done < <(find "$directory" -type f -not -path '*/__pycache__/*' -print0 | sort -z)
    else
        # The active historical directory may be retired after its first copy.
        test -d "$snapshot/harnesses/$directory"
        while IFS= read -r -d '' source; do
            entry "$source" "${source#"$snapshot/harnesses/"}" "harness" "$scratch/files.jsonl"
        done < <(find "$snapshot/harnesses/$directory" -type f -print0 | sort -z)
    fi
done

# Supplement exact measured snapshots with their environment/recovery inputs.
# These files are archive-time support, not a claim of hash-exact timed builds.
for source in pyproject.toml docker/Dockerfile docker/dev.sh docker/fetch_review_assets.py; do
    copy_file "$source" "$snapshot/support/$source" "archive_time_environment"
done

# Preserve the pending origins text before the live file is simplified.
# The caller copied this first, and this generator must never replace it.
test -f "$snapshot/pending/docs/origins.md"
entry "$snapshot/pending/docs/origins.md" "docs/origins.md (pending text before live-doc cleanup)" \
    "pending_provenance" "$scratch/files.jsonl"

# This recovery snapshot predates unified sampling and contains the old Warp
# validation/UI state. Do not retain the large tar or duplicate old GL files.
while IFS= read -r source; do
    case "$source" in
        tests/verify_ray_sensors.py|tests/verify_ray_review.py|tools/sensor_review/*.py|tools/sensor_review/README.md|tools/sensor_review/web/*.js|tools/sensor_review/web/*.html|tools/sensor_review/web/*.css)
            destination="$snapshot/pre-unified/$source"
            mkdir -p "$(dirname "$destination")"
            tar -xOf outputs/ray_unified/before-worktree.tar "$source" > "$destination"
            entry "$destination" "outputs/ray_unified/before-worktree.tar::$source" \
                "pre_unified_recovery_source" "$scratch/files.jsonl"
            ;;
    esac
done < <(tar -tf outputs/ray_unified/before-worktree.tar)

# The pilot source directory was later overwritten by the completed sweep.
# Restore only the two explicitly recorded stream-lifetime additions and
# accept the reconstruction solely if the pilot's recorded SHA matches.
prestream="$snapshot/pre-stream-record/devtools/sensors.py"
mkdir -p "$(dirname "$prestream")"
awk '
    /^def / { process = ($0 ~ /^def process_samples\(/) }
    /^    # The caller establishes readiness when crossing streams\./ { skip = 3 }
    process && /^    stream = torch.cuda.current_stream\(poses.device\)$/ { skip = 6 }
    skip > 0 { skip--; next }
    { print }
' \
    outputs/ray_unified/perf-with-counter-wrapper/pillars5000-measurements-b64-source/devtools/sensors.py > "$prestream"
expected="$(jq -r '.source_sha256["devtools/sensors.py"]' outputs/ray_unified/perf-with-counter-wrapper/pilot-before-stream-record.json)"
test "$(sha256sum "$prestream" | cut -d ' ' -f1)" = "$expected"
entry "$prestream" "reconstructed from counter-wrapper source by removing the two recorded record_stream additions; pilot SHA matched" \
    "hash_verified_reconstruction" "$scratch/files.jsonl"

copy_file outputs/archive_warp_exploration.sh "$snapshot/archive_warp_exploration.sh" "archive_generator"

# Shared source and maps are already frozen by bb5ca53: reference instead of
# duplicating them. They provide a base workspace for the overlays above.
for relative in src/tof_tracker/__init__.py src/tof_tracker/geometry.py devtools/__init__.py devtools/trajectory.py \
    outputs/sensor_perf/maps/pillars-20.obj outputs/sensor_perf/maps/pillars-5000.obj; do
    source="$archive/snapshots/zone-20261017/$relative"
    test -f "$source"
    entry "$source" "$relative" "reuse_zone_snapshot" "$scratch/reused.jsonl"
done

jq -s 'sort_by(.path)' "$scratch/files.jsonl" > "$scratch/files.json"
jq -s 'sort_by(.path)' "$scratch/excluded.jsonl" > "$scratch/excluded.json"
jq -s 'sort_by(.path)' "$scratch/reused.jsonl" > "$scratch/reused.json"
jq -n --slurpfile files "$scratch/files.json" --slurpfile excluded "$scratch/excluded.json" \
    --slurpfile reused "$scratch/reused.json" \
    '{schema:1,captured_date:"2026-10-17",base_archive_commit:"bb5ca53c21031e69019d0e9f2d808178ad8b8974",
      purpose:"Warp ray, postprocess profiling, and unified sampling evidence; original outputs remain unchanged",
      snapshot_root:"snapshots/warp-20261017",evidence_root:"evidence",
      notes:[
        "Original JSON/CSV/log/report bytes and adjacent measured-source paths are retained without rewriting",
        "Warp timed_source differs from current_source; use provenance.json to recover the measured code",
        "Postprocess Graph is a prepared-bounds scheduling diagnostic, not production direct Graph support",
        "perf-with-counter-wrapper is superseded timing methodology; its historical source_snapshot paths predate relocation",
        "The pilot pre-stream sensors.py is restored by removing only the two known record_stream additions; its recorded SHA matches exactly",
        "runtime-probe.json and runtime-probe-final.json failed to establish pending stream pressure; stream-final is the later valid probe",
        "Use exact per-cell source overlays; harnesses/support and pre-unified are archive-time/recovery inputs, not substitutes for source hashes",
        "The early eight-group verification result has no test-source hash; its recovery test is preserved without asserting exact timed provenance",
        "Large traces, binaries, duplicate maps and offline HTML are not added; excluded local artifacts remain at their original paths",
        "Native reference and zone evidence already belongs to the base archive and is not copied again"
      ],files:$files[0],excluded_local_artifacts:$excluded[0],reused:$reused[0],
      totals:{files:($files[0]|length),bytes:($files[0]|map(.bytes)|add),
        excluded_files:($excluded[0]|length),excluded_bytes:($excluded[0]|map(.bytes)|add)}}' > "$manifest"
jq -r '.files[] | .sha256 + "  " + .path' "$manifest" > "$index"
manifest_hash="$(sha256sum "$manifest" | cut -d ' ' -f 1)"
index_hash="$(sha256sum "$index" | cut -d ' ' -f 1)"
jq --arg manifest_hash "$manifest_hash" --arg index_hash "$index_hash" \
    '.additional_archives["warp-20261017"]={manifest:"warp-20261017.json",sha256:$manifest_hash,
      index:"warp-20261017.sha256",index_sha256:$index_hash}' \
    "$archive/manifest.json" > "$scratch/base-manifest.json"
cp "$scratch/base-manifest.json" "$archive/manifest.json"
(cd "$archive" && sha256sum --check --quiet warp-20261017.sha256)
jq '.totals' "$manifest"
