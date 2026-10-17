"""Run timing cells sequentially in fresh processes inside the project Docker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def last_progress_stage(log_path):
    """Read the last emitted benchmark stage, ignoring other log messages."""
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("stage"), str):
            return value["stage"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, nargs="+", default=[20, 500, 5000])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--modes", nargs="+", choices=["tof", "camera", "rig"], default=["tof", "camera", "rig"])
    parser.add_argument("--sampling", choices=["pixel", "zone"], default="zone")
    parser.add_argument("--pose-pattern", choices=["varied", "repeat"], default="varied")
    parser.add_argument("--budget-ms", type=float, default=500)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-repeats", type=int, default=7)
    parser.add_argument("--max-repeats", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=240)
    parser.add_argument("--synchronize-first-forward", action="store_true")
    parser.add_argument("--skip-larger-after-failure", action="store_true",
                        help="after a non-ok cell, skip larger batches for the same map/mode")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    total = len(args.maps) * len(args.batches) * len(args.modes)
    for count in args.maps:
        for mode in args.modes:
            failed_batch = None
            failed_label = None
            failed_status = None
            for batch in args.batches:
                label = f"pillars{count}-{mode}-{args.sampling}-b{batch}-{args.pose_pattern}"
                result_path = args.output / (label + ".json")
                command = [sys.executable, str(ROOT / "experiments/sensor_zone_perf/benchmark.py"),
                           "--map", str(ROOT / f"outputs/sensor_perf/maps/pillars-{count}.obj"),
                           "--mode", mode, "--batch", str(batch), "--sampling", args.sampling,
                           "--pose-pattern", args.pose_pattern, "--output", str(result_path),
                           "--target-budget-ms", str(args.budget_ms), "--warmup", str(args.warmup),
                           "--min-repeats", str(args.min_repeats), "--max-repeats", str(args.max_repeats),
                           "--probe-gl-draw"]
                if args.synchronize_first_forward:
                    command.append("--synchronize-first-forward")
                if (args.skip_larger_after_failure and failed_batch is not None
                        and batch > failed_batch):
                    record = {"label": label, "command": command, "returncode": None,
                              "elapsed_s": 0., "result": str(result_path),
                              "status": "skipped_after_failure", "timeout_s": args.timeout_s,
                              "last_progress_stage": None,
                              "reason": {"failed_cell": failed_label,
                                         "failed_batch": failed_batch,
                                         "failed_status": failed_status,
                                         "message": "larger batch skipped for the same map/mode"}}
                    records.append(record)
                    (args.output / "manifest.json").write_text(json.dumps(records, indent=2)+"\n")
                    print(f"SKIP {len(records)}/{total} {label}: {failed_label} {failed_status}", flush=True)
                    continue
                print(f"START {len(records)+1}/{total} {label}", flush=True)
                start = time.perf_counter()
                failure = None
                log_path = args.output / (label + ".log")
                with log_path.open("w") as log:
                    try:
                        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                   timeout=args.timeout_s, check=False)
                        returncode = completed.returncode
                    except subprocess.TimeoutExpired:
                        returncode = None
                        failure = "timeout"
                value = json.loads(result_path.read_text()) if result_path.exists() else {}
                record = {"label": label, "command": command, "returncode": returncode,
                          "elapsed_s": time.perf_counter()-start, "result": str(result_path),
                          "status": failure or value.get("status", "missing_result"),
                          "timeout_s": args.timeout_s,
                          "last_progress_stage": last_progress_stage(log_path)}
                if (returncode != 0 or record["status"] != "ok") and (
                        failed_batch is None or batch < failed_batch):
                    failed_batch = batch
                    failed_label = label
                    failed_status = record["status"]
                records.append(record)
                (args.output / "manifest.json").write_text(json.dumps(records, indent=2)+"\n")
                print(f"END {label} {record['status']} {record['elapsed_s']:.2f}s", flush=True)
    failed = [item for item in records if item["returncode"] != 0 or item["status"] != "ok"]
    print(json.dumps({"cells": len(records), "failed": len(failed), "output": str(args.output)}), flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
