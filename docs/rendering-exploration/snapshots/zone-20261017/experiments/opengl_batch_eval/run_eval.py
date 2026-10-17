"""Isolated, source-fingerprinted OpenGL multi-view experiment orchestration."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.pytorch3d_render_eval.run_eval import command, ensure_maps


def source_hashes():
    suffixes = {".py", ".glsl", ".vert", ".frag", ".geom", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".cuh", ".sh"}
    sources = [p for folder in (HERE, ROOT / "experiments/pytorch3d_render_eval", ROOT / "devtools")
               for p in folder.rglob("*") if p.is_file() and p.suffix in suffixes]
    sources += [ROOT / "src/tof_tracker/geometry.py", ROOT / "configs/devtools.json",
                ROOT / "experiments/sensor_reference.py", ROOT / "tests/verify_sensor_reference.py"]
    sources += [p for p in HERE.glob("Dockerfile*") if p.is_file()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))}


def validations_in(output):
    results = {}
    for path in sorted(output.glob("validation*.json")):
        result = json.loads(path.read_text())
        if "backend" in result and "cases" in result:
            results[result["backend"]] = {"status": result.get("status", "unknown"), "file": path.name}
    return results


def flatten(records, validations):
    rows = []
    for r in records:
        timing, memory = r.get("end_to_end", {}), r.get("warm_memory", {})
        explicit_gl = r.get("gl_warm_memory", {})
        gl_bytes = explicit_gl.get("tracked_total_bytes")
        rows.append({"pillars": r["pillars"], "mode": r["mode"], "backend": r["backend"],
                     "batch_views": r["batch_views"], "status": r["status"],
                     "validation_status": validations.get(r["backend"], {}).get("status", "not_validated"),
                     "validation_file": validations.get(r["backend"], {}).get("file"),
                     "median_ms": timing.get("wall_median_ms"), "views_per_second": timing.get("views_per_second"),
                     "torch_peak_allocated_mib": memory.get("peak_allocated_bytes", 0) / 2**20,
                     "torch_peak_reserved_mib": memory.get("peak_reserved_bytes", 0) / 2**20,
                     "gl_tracked_mib": gl_bytes / 2**20 if gl_bytes is not None else None,
                     "nvml_peak_device_mib": r.get("nvml", {}).get("peak_bytes", 0) / 2**20,
                     "nvml_increment_mib": r.get("nvml", {}).get("peak_increment_bytes", 0) / 2**20,
                     "base_prepare_ms": r.get("base_prepare_ms"), "batch_prepare_ms": r.get("batch_prepare_ms"),
                     "first_forward_ms": r.get("first_forward_ms"),
                     "raster_ms": r.get("raster_stage", {}).get("wall_median_ms"),
                     "postprocess_ms": r.get("batched_postprocess_stage", {}).get("wall_median_ms"),
                     "error": r.get("error", "")})
    return rows


def make_plots(output, rows, validation_text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenes = sorted({r["pillars"] for r in rows})
    backends = sorted({r["backend"] for r in rows})
    fig, axes = plt.subplots(2, len(scenes), figsize=(5 * len(scenes), 7), squeeze=False)
    for col, scene in enumerate(scenes):
        for row, mode in enumerate(("tof", "camera")):
            ax = axes[row, col]
            for backend in backends:
                data = sorted([r for r in rows if r["pillars"] == scene and r["mode"] == mode
                               and r["backend"] == backend and r["status"] == "ok"], key=lambda r: r["batch_views"])
                ax.plot([r["batch_views"] for r in data], [r["views_per_second"] for r in data], marker="o", label=backend)
            ax.set(title=f"{scene} pillars / {mode}", xlabel="Batch viewpoints B", ylabel="views / second", xscale="log")
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle(f"Same-run render + required outputs on CUDA; correctness: {validation_text}")
    fig.tight_layout()
    fig.savefig(output / "throughput.png", dpi=150)
    plt.close(fig)
    # Keep unlike accounting systems on separate panels. Baseline GL explicit
    # allocation is unavailable rather than spuriously reported as zero.
    metrics = (("torch_peak_allocated_mib", "Torch allocator peak MiB"),
               ("gl_tracked_mib", "Tracked GL buffers + attachments MiB"),
               ("nvml_peak_device_mib", "Sampled whole-device NVML peak MiB"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    for row, mode in enumerate(("tof", "camera")):
        for col, (key, title) in enumerate(metrics):
            ax = axes[row, col]
            for scene in scenes:
                for backend in backends:
                    data = sorted([r for r in rows if r["pillars"] == scene and r["mode"] == mode
                                   and r["backend"] == backend and r["status"] == "ok" and r[key] is not None],
                                  key=lambda r: r["batch_views"])
                    if data:
                        ax.plot([r["batch_views"] for r in data], [r[key] for r in data], marker="o",
                                linestyle="--" if backend == "current_gpu" else "-", label=f"{scene} / {backend}")
            ax.set(title=f"{mode}: {title}", xlabel="Batch viewpoints B", ylabel="MiB", xscale="log")
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
    fig.suptitle("Separate memory scopes; Torch peak excludes GL, NVML includes driver and other processes")
    fig.tight_layout()
    fig.savefig(output / "memory.png", dpi=150)
    plt.close(fig)


def summarize(output, plots=False):
    records = [json.loads(p.read_text()) for p in sorted((output / "cells").glob("*.json"))]
    validations = validations_in(output)
    validation_text = "; ".join(f"{k}: {v['status']}" for k, v in sorted(validations.items())) or "not_run"
    rows = flatten(records, validations)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    if rows:
        with (output / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# OpenGL multi-view batch experiment", "", f"Correctness validation: **{validation_text}**.", "",
             "Timing status `ok` only means execution completed. Read validation separately; throughput does not establish equivalence.", "",
             "`current_gpu` calls the unchanged production sensor sequentially, including per-view GL readback and CPU postprocessing, then stacks/uploads required results. `gl_batch` accepts prepared CUDA poses, renders a shared scene using instancing, transfers GL outputs to Torch through CUDA interop, and batches range/coverage/mask/fraction/bbox postprocessing. Camera includes both full-scene and target-only rasterization.", "",
             "B counts sensor viewpoints, not rigs. Main timing is synchronized wall time, including GL/CUDA interop, required copies and postprocessing. Input poses and static scene are prepared; initialization, batch preparation and first call are recorded separately. Stage timings are separate probes and must not be summed.", "",
             "Torch allocated/reserved peaks exclude GL. Explicitly tracked GL storage excludes driver overhead; the baseline does not expose this storage (unavailable, not zero). NVML is whole-device sampling every 5 ms, includes other applications and CUDA/GL drivers, and may miss brief peaks. Its increment is relative to the process's pre-context baseline. These memory columns cannot be added or treated as interchangeable.", "",
             "The Torch allocator cap is 80% of device memory; GL is not constrained by that allocator cap. OOM results refer to the tested policy, not an independently measured physical GPU limit.", "",
             "| Pillars | Workload | Backend | B | Status | ms/batch | views/s | Torch peak MiB | GL tracked MiB | NVML device peak MiB |",
             "|---:|---|---|---:|---|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    for r in sorted(rows, key=lambda v: (v["pillars"], v["mode"], v["batch_views"], v["backend"])):
        lines.append(f'| {r["pillars"]} | {r["mode"]} | {r["backend"]} | {r["batch_views"]} | {r["status"]} | {fmt(r["median_ms"])} | {fmt(r["views_per_second"])} | {fmt(r["torch_peak_allocated_mib"])} | {fmt(r["gl_tracked_mib"])} | {fmt(r["nvml_peak_device_mib"])} |')
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if plots and rows:
        make_plots(output, rows, validation_text)
    return rows


def run_child(invocation, log_path, timeout):
    with log_path.open("w") as log:
        try:
            return subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return -999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pillars", type=int, nargs="+", default=[20, 500, 5000])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256])
    parser.add_argument("--backends", choices=("current_gpu", "gl_batch"), nargs="+", default=["current_gpu", "gl_batch"])
    parser.add_argument("--modes", choices=("tof", "camera"), nargs="+", default=["tof", "camera"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cell-timeout", type=int, default=240)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "outputs/opengl_batch_eval"):
        raise ValueError("run artifacts must stay below outputs/opengl_batch_eval/")
    if any(n < 1 for n in args.batches) or any(n < 0 for n in args.pillars):
        raise ValueError("batch sizes must be positive and pillar counts nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("cells", "logs"):
        (output / name).mkdir(exist_ok=True)
    ensure_maps(args.pillars, output)
    environment = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "docker_image_id": os.environ.get("OPENGL_BATCH_EVAL_IMAGE_ID"), "python": sys.version,
                   "platform": platform.platform(), "arguments": vars(args),
                   "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"]),
                   "gpu_processes_before": command(["nvidia-smi"]),
                   "source_sha256": source_hashes(), "packages": command([sys.executable, "-m", "pip", "freeze"]),
                   "gpu_identity": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])}
    fingerprint_input = {"source_sha256": environment["source_sha256"], "docker_image_id": environment["docker_image_id"],
                         "scene_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output / "maps").glob("*.obj"))},
                         "arguments": {k: v for k, v in vars(args).items() if k not in ("output", "resume")},
                         "packages": environment["packages"]["stdout"], "gpu_identity": environment["gpu_identity"]["stdout"]}
    environment["fingerprint"] = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    environment["input_fingerprint"] = fingerprint_input
    previous = output / "environment.json"
    if any((output / "cells").glob("*.json")):
        if not args.resume:
            raise ValueError("output already has cells; use a fresh directory or strict --resume")
        if not previous.exists() or json.loads(previous.read_text()).get("fingerprint") != environment["fingerprint"]:
            raise ValueError("cannot resume: source, image, inputs, packages, or measurement arguments changed")
    previous.write_text(json.dumps(environment, indent=2, allow_nan=False) + "\n")
    if not args.skip_validation:
        validation_path = output / "validation.json"
        # A crashed/timeout child must never inherit an older successful result.
        validation_path.unlink(missing_ok=True)
        invocation = [sys.executable, str(HERE / "validate.py"), "--output", str(output / "validation.json"),
                      "--pillars", *map(str, args.pillars)]
        code = run_child(invocation, output / "validation.log", timeout=1200)
        if not validation_path.exists():
            result = {"backend": "gl_batch", "status": "process_failed", "cases": [],
                      "error": f"validation exit={code}; see validation.log"}
        else:
            result = json.loads(validation_path.read_text())
            if code != 0 and result.get("status") == "passed":
                result.update(status="invalid_exit", status_before_invalid_exit="passed",
                              error=f"validation wrote passed but exited with code {code}; see validation.log")
        result["process_exit_code"] = code
        validation_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"validation exit={code}; semantic status in validation.json", flush=True)
    for count in args.pillars:
        for mode in args.modes:
            for backend in args.backends:
                oom_seen = False
                for batch in sorted(args.batches):
                    token = f"pillars{count}-{mode}-{backend}-b{batch}"
                    result_path = output / "cells" / f"{token}.json"
                    if args.resume and result_path.exists():
                        oom_seen |= json.loads(result_path.read_text()).get("status") == "oom"
                        continue
                    if oom_seen:
                        result_path.write_text(json.dumps({"pillars": count, "mode": mode, "backend": backend,
                                                          "batch_views": batch, "status": "skipped_after_oom"}, indent=2) + "\n")
                        continue
                    invocation = [sys.executable, str(HERE / "worker.py"), "--backend", backend, "--mode", mode,
                                  "--batch", str(batch), "--pillars", str(count), "--output", str(result_path),
                                  "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
                    print(f"running {token}", flush=True)
                    code = run_child(invocation, output / "logs" / f"{token}.log", args.cell_timeout)
                    if not result_path.exists():
                        result_path.write_text(json.dumps({"pillars": count, "mode": mode, "backend": backend,
                                                          "batch_views": batch, "status": "process_failed",
                                                          "error": f"worker exit={code}; see logs/{token}.log"}, indent=2) + "\n")
                    cell = json.loads(result_path.read_text())
                    cell["process_exit_code"] = code
                    if code != 0 and cell.get("status") == "ok":
                        cell.update(status="invalid_exit", status_before_invalid_exit="ok",
                                    error=f"worker wrote ok but exited with code {code}; see logs/{token}.log")
                    result_path.write_text(json.dumps(cell, indent=2, allow_nan=False) + "\n")
                    oom_seen = cell.get("status") == "oom"
                    print(f"  {cell['status']}: {cell.get('end_to_end', {}).get('wall_median_ms')} ms", flush=True)
                    summarize(output)
    environment["gpu_processes_after"] = command(["nvidia-smi"])
    previous.write_text(json.dumps(environment, indent=2, allow_nan=False) + "\n")
    summarize(output, plots=True)


if __name__ == "__main__":
    main()
