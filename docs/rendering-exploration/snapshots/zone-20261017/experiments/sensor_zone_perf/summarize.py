"""Collect independently measured cells into portable JSON and CSV tables."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def command_option(command, name):
    """Recover workload metadata for cells killed before writing their JSON."""
    try:
        index = command.index(name)
        return command[index + 1]
    except (ValueError, IndexError):
        return None


def mib(value):
    """Missing measurements remain unavailable; a measured zero stays zero."""
    return value / 2**20 if type(value) in (int, float) else None


def make_row(value, path, manifest=None, manifest_path=None):
    manifest = manifest or {}
    command = manifest.get("command", [])
    map_path = value.get("map", {}).get("path") or command_option(command, "--map")
    mode = value.get("mode") or command_option(command, "--mode")
    batch = value.get("batch")
    if batch is None:
        argument = command_option(command, "--batch")
        batch = int(argument) if argument is not None else None
    row = {
        "map": Path(map_path).stem if map_path else "unknown",
        "mode": mode,
        "sampling": value.get("sampling") or command_option(command, "--sampling"),
        "batch": batch,
        "unit": value.get("unit") or ("body_rigs" if mode == "rig" else "views" if mode else None),
        "pose_pattern": value.get("pose_pattern") or command_option(command, "--pose-pattern"),
        "status": manifest.get("status") or value.get("status") or "missing_result",
        "synchronized_first_forward": value.get(
            "synchronized_first_forward", "--synchronize-first-forward" in command),
        "cuda_stream": value.get("cuda_stream", "default"),
        "source_file": str(path),
    }
    if manifest_path is not None:
        row.update(manifest_file=str(manifest_path))
        for key in ("returncode", "elapsed_s", "last_progress_stage", "reason", "timeout_s"):
            row[key] = manifest.get(key)
    for stage in ("pipeline", "postprocess", "end_to_end", "mount"):
        data = value.get(stage) or {}
        for key in ("wall_median_ms", "wall_p95_ms", "repeats", "units_per_second"):
            row[f"{stage}_{key}"] = data.get(key)
    row["gl_draw_gpu_ms"] = value.get("gl_draw_probe", {}).get("per_action_sum_median_ms")
    e2e = value.get("end_to_end") or {}
    torch_memory = e2e.get("torch_memory_after") or {}
    gl_memory = e2e.get("gl_memory_after") or {}
    row["torch_peak_allocated_mib"] = mib(torch_memory.get("peak_allocated_bytes"))
    row["torch_reserved_mib"] = mib(torch_memory.get("reserved_bytes"))
    row["tracked_gl_mib"] = mib(gl_memory.get("tracked_total_bytes"))
    row["triangle_count"] = value.get("geometry", {}).get("triangle_count")
    row["production_parity"] = value.get("production_parity", {}).get("passed")
    row["first_forward_ms"] = value.get("first_forward_ms")
    return row


def collect(folders):
    rows, sources, seen = [], {}, set()

    def append(path, manifest=None, manifest_path=None):
        identity = path.resolve()
        if identity in seen:
            return
        seen.add(identity)
        value = json.loads(path.read_text()) if path.is_file() else {}
        if not isinstance(value, dict):
            value = {}
        if manifest is None and "mode" not in value:
            return
        rows.append(make_row(value, path, manifest, manifest_path))
        for name, digest in value.get("source", {}).get("files_sha256", {}).items():
            sources.setdefault(name, set()).add(digest)

    for folder in folders:
        manifest_path = folder / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, list):
                raise ValueError(f"expected a list of timing cells in {manifest_path}")
            for entry in manifest:
                # Sweep cells are flat files beside their manifest. Basename
                # lookup also works when a container recorded /workspace paths.
                filename = Path(entry.get("result") or (entry["label"] + ".json")).name
                append(folder / filename, entry, manifest_path)
        for path in sorted(folder.glob("*.json")):
            if path != manifest_path:
                append(path)
    return rows, sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows, sources = collect(args.folders)
    args.output.mkdir(parents=True,exist_ok=True)
    names=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/"results.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    summary={"rows":rows,"cells":len(rows),"successful":sum(row["status"]=="ok" for row in rows),
             "status_counts":dict(sorted(Counter(row["status"] for row in rows).items())),
             "source_hashes":{name:sorted(values) for name,values in sources.items()},
             "scope":"prepared CUDA input to GPU output; independent stages are not additive; no UI downloads",
             "memory_scope":"Torch allocated peak and tracked GL exclude driver; reserved is allocator cache, not live tensors"}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"cells":summary["cells"],"successful":summary["successful"],"output":str(args.output)}))


if __name__=="__main__":
    main()
