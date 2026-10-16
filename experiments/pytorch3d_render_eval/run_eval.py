"""Reproducible validation and per-process throughput/memory sweep."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def command(args):
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def ensure_maps(counts, output):
    folder = output / "maps"
    folder.mkdir(exist_ok=True)
    for count in counts:
        source = ROOT / "outputs/sensor_perf/maps" / f"pillars-{count}.obj"
        destination = folder / source.name
        if not destination.exists() and source.exists():
            shutil.copyfile(source, destination)
    missing = [n for n in counts if not (folder / f"pillars-{n}.obj").is_file()]
    if missing:
        build = output / "mapgen-build"
        for invocation in (["cmake", "-S", str(ROOT / "tools/mapgen"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                           ["cmake", "--build", str(build), "--parallel", "2"]):
            subprocess.run(invocation, check=True)
        for count in missing:
            subprocess.run([str(build / "offline_mapgen"), "--type", "pillars", "--seed", "3",
                            "--size-x", "60", "--size-y", "60", "--size-z", "15", "--count", str(count),
                            "--width-min", ".3", "--width-max", "1", "--height-min", "1.5", "--height-max", "15",
                            "--output", str(folder / f"pillars-{count}.obj")], check=True)
    os.environ["PYTORCH3D_EVAL_MAP_DIR"] = str(folder)


def summarize(output, plots=False):
    records = [json.loads(path.read_text()) for path in sorted((output / "cells").glob("*.json"))]
    validations = {}
    for path in sorted(output.glob("validation*.json")):
        result = json.loads(path.read_text())
        if "backend" in result and "cases" in result:
            validations[result["backend"]] = {"status": result.get("status", "unknown"),
                                               "file": path.name}
    validation_status = "; ".join(f"{key}: {value['status']}" for key, value in sorted(validations.items())) or "not_run"
    flattened = []
    for r in records:
        timing, memory = r.get("end_to_end", {}), r.get("warm_memory", {})
        flattened.append({"pillars": r["pillars"], "mode": r["mode"], "backend": r["backend"],
                          "batch_views": r["batch_views"], "status": r["status"],
                          "validation_status": validations.get(r["backend"], {}).get("status", "not_validated"),
                          "validation_file": validations.get(r["backend"], {}).get("file"),
                          "median_ms": timing.get("wall_median_ms"), "views_per_second": timing.get("views_per_second"),
                          "torch_peak_allocated_mib": memory.get("peak_allocated_bytes", 0) / 2**20,
                          "torch_peak_reserved_mib": memory.get("peak_reserved_bytes", 0) / 2**20,
                          "nvml_peak_device_mib": r.get("nvml", {}).get("peak_bytes", 0) / 2**20,
                          "nvml_increment_mib": r.get("nvml", {}).get("peak_increment_bytes", 0) / 2**20,
                          "prepare_ms": r.get("base_prepare_ms", 0) + r.get("batch_prepare_ms", 0),
                          "first_forward_ms": r.get("first_forward_ms"),
                          "postprocess_ms": r.get("batched_postprocess_stage", {}).get("wall_median_ms"),
                          "error": r.get("error", "")})
    (output / "summary.json").write_text(json.dumps(flattened, indent=2) + "\n")
    if flattened:
        with (output / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flattened)
    lines = ["# PyTorch3D batch rendering experiment", "", "Raw evidence: `cells/`, `validation.json`, `environment.json`.", "",
             f"**Correctness validation by backend: {validation_status}.** Timing cell status `ok` means execution completed, not semantic equivalence. A validation result applies only to its named backend. Retained timings do not establish correctness when validation failed or was not run.", "",
             "B counts sensor viewpoints, not rigs. Camera includes full + target-only rasterization and batched mask/fraction/bbox. ToF includes coverage/range postprocessing.", "",
             "`current` returns original CPU frames. `current_gpu` runs that same implementation then stacks/copies required outputs to CUDA. `p3d` returns CUDA tensors throughout. Main timing is synchronized wall time with scene, batch mesh, and input poses prepared.", "",
             "Safe bin capacity is 2F unless explicitly labeled. PyTorch process allocation is capped at 80% of device memory to leave space for desktop/other GPU work. OOM is a tested cap failure, not a physical GPU maximum.", "",
             "NVML is sampled device-wide, includes other processes and CUDA/GL driver allocations; its 5 ms samples may miss brief peaks. Torch allocated/reserved peaks cover its own allocator only. Stage timings are separate runs and must not be summed.", "",
             "| Pillars | Workload | Backend | B views | Status | ms/batch | views/s | Torch peak MiB | NVML device peak MiB |", "|---:|---|---|---:|---|---:|---:|---:|---:|"]
    policies = sorted({str(r.get("geometry", {}).get("max_faces_per_bin_policy"))
                       for r in records if r["backend"].startswith("p3d") and "geometry" in r})
    lines.insert(4, f"Measured PyTorch3D bin capacity policies: {', '.join(policies)}.")
    for r in sorted(flattened, key=lambda v: (v["pillars"], v["mode"], v["batch_views"], v["backend"])):
        fmt = lambda v: "—" if v is None else f"{v:.2f}"
        lines.append(f'| {r["pillars"]} | {r["mode"]} | {r["backend"]} | {r["batch_views"]} | {r["status"]} | {fmt(r["median_ms"])} | {fmt(r["views_per_second"])} | {fmt(r["torch_peak_allocated_mib"])} | {fmt(r["nvml_peak_device_mib"])} |')
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if not plots:
        return flattened
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scenes = sorted({r["pillars"] for r in flattened})
        fig, axes = plt.subplots(2, len(scenes), figsize=(5 * len(scenes), 7), squeeze=False)
        for col, scene in enumerate(scenes):
            for row, mode in enumerate(("tof", "camera")):
                ax = axes[row, col]
                if not any(v["pillars"] == scene and v["mode"] == mode and v["status"] == "ok" for v in flattened):
                    ax.set_visible(False)
                    continue
                for backend in sorted({v["backend"] for v in flattened}):
                    data = sorted([v for v in flattened if v["pillars"] == scene and v["mode"] == mode
                                   and v["backend"] == backend and v["status"] == "ok"], key=lambda v: v["batch_views"])
                    ax.plot([v["batch_views"] for v in data], [v["views_per_second"] for v in data], marker="o", label=backend)
                ax.set(title=f"{scene} pillars / {mode}", xlabel="Batch viewpoints B", ylabel="views / second", xscale="log", yscale="log")
                ax.grid(alpha=.25)
                ax.legend()
        fig.suptitle(f"Render + postprocessing, synchronized wall time; correctness: {validation_status}")
        fig.tight_layout()
        fig.savefig(output / "throughput.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, mode in zip(axes, ("tof", "camera")):
            if not any(v["mode"] == mode and v["backend"].startswith("p3d") and v["status"] == "ok" for v in flattened):
                ax.set_visible(False)
                continue
            for scene in scenes:
                for backend in sorted({v["backend"] for v in flattened if v["backend"].startswith("p3d")}):
                    data = sorted([v for v in flattened if v["pillars"] == scene and v["mode"] == mode
                                   and v["backend"] == backend and v["status"] == "ok"], key=lambda v: v["batch_views"])
                    ax.plot([v["batch_views"] for v in data], [v["torch_peak_allocated_mib"] for v in data],
                            marker="o", linestyle="--" if backend == "p3d" else "-",
                            label=f"{scene} pillars / {backend}")
            ax.set(title=f"PyTorch3D / {mode}", xlabel="Batch viewpoints B", ylabel="Torch peak allocated MiB", xscale="log")
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "memory.png", dpi=150)
        plt.close(fig)
    except ImportError as error:
        (output / "plot-unavailable.txt").write_text(str(error))
    return flattened


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pillars", type=int, nargs="+", default=[20, 500, 5000])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256])
    parser.add_argument("--backends", nargs="+", default=["current", "current_gpu", "p3d"])
    parser.add_argument("--modes", nargs="+", default=["tof", "camera"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validation-backend", choices=("p3d", "p3d_conditioned", "all"), default="p3d")
    parser.add_argument("--bin-capacity", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("all run artifacts must stay below outputs/")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cells").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    ensure_maps(args.pillars, output)
    environment = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "docker_image_id": os.environ.get("PYTORCH3D_EVAL_IMAGE_ID"),
                   "python": sys.version, "platform": platform.platform(), "arguments": vars(args),
                   "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"]),
                   "gpu_processes_before": command(["nvidia-smi"]),
                   "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [*HERE.glob("*.py"), *(ROOT / "devtools").glob("*.py"),
                                               ROOT / "src/tof_tracker/geometry.py"]},
                   "packages": command([sys.executable, "-m", "pip", "freeze"])}
    rev = Path("/opt/pytorch3d-source-revision")
    environment["pytorch3d_revision"] = rev.read_text().strip() if rev.exists() else None
    fingerprint_input = {
        "source_sha256": environment["source_sha256"],
        "configuration_sha256": hashlib.sha256((ROOT / "configs/devtools.json").read_bytes()).hexdigest(),
        "scene_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output / "maps").glob("*.obj")},
        "arguments": {k: v for k, v in vars(args).items() if k not in ("output", "resume", "skip_validation")},
        "packages": environment["packages"]["stdout"],
        "pytorch3d_revision": environment["pytorch3d_revision"],
    }
    environment["fingerprint"] = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    prior_environment = output / "environment.json"
    if any((output / "cells").glob("*.json")):
        if not args.resume:
            raise ValueError("Output contains prior cells; choose a fresh output or use strict --resume")
        if not prior_environment.exists() or json.loads(prior_environment.read_text()).get("fingerprint") != environment["fingerprint"]:
            raise ValueError("Cannot resume: source, inputs, dependencies, or measurement arguments changed")
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if not args.skip_validation:
        validation_backends = ("p3d", "p3d_conditioned") if args.validation_backend == "all" else (args.validation_backend,)
        for backend in validation_backends:
            token = f"validation-{backend}" if args.validation_backend == "all" else "validation"
            with (output / f"{token}.log").open("w") as log:
                validation_command = [sys.executable, str(HERE / "validate.py"), "--output", str(output / f"{token}.json"),
                                      "--backend", backend, "--pillars", *map(str, args.pillars)]
                if args.bin_capacity is not None:
                    validation_command += ["--bin-capacity", args.bin_capacity]
                result = subprocess.run(validation_command, stdout=log, stderr=subprocess.STDOUT)
            print(f"{backend} validation exit={result.returncode}; detailed status in {token}.json", flush=True)
    for count in args.pillars:
        for mode in args.modes:
            for backend in args.backends:
                oom_seen = False
                for batch in sorted(args.batches):
                    token = f"pillars{count}-{mode}-{backend}-b{batch}"
                    result_path = output / "cells" / f"{token}.json"
                    if args.resume and result_path.exists():
                        prior = json.loads(result_path.read_text())
                        oom_seen |= prior.get("status") == "oom"
                        continue
                    if oom_seen:
                        result_path.write_text(json.dumps({"pillars": count, "mode": mode, "backend": backend,
                                                          "batch_views": batch, "status": "skipped_after_oom"}, indent=2))
                        continue
                    command_args = [sys.executable, str(HERE / "worker.py"), "--backend", backend, "--mode", mode,
                                    "--batch", str(batch), "--pillars", str(count), "--output", str(result_path),
                                    "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
                    if args.bin_capacity is not None:
                        command_args += ["--bin-capacity", args.bin_capacity]
                    print(f"running {token}", flush=True)
                    with (output / "logs" / f"{token}.log").open("w") as log:
                        try:
                            completed = subprocess.run(command_args, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                            code = completed.returncode
                        except subprocess.TimeoutExpired:
                            code = -999
                    if not result_path.exists():
                        result_path.write_text(json.dumps({"pillars": count, "mode": mode, "backend": backend,
                                                          "batch_views": batch, "status": "process_failed",
                                                          "error": f"worker exit={code}; see logs/{token}.log"}, indent=2))
                    cell = json.loads(result_path.read_text())
                    log_text = (output / "logs" / f"{token}.log").read_text(errors="replace").lower()
                    if "overflow" in log_text and ("coarse raster" in log_text or "max_faces_per_bin" in log_text):
                        cell["status"] = "invalid_overflow"
                        cell["error"] = "PyTorch3D coarse rasterizer overflow: image may be incomplete; excluded from valid throughput"
                        result_path.write_text(json.dumps(cell, indent=2) + "\n")
                    oom_seen = cell.get("status") == "oom"
                    print(f"  {cell['status']}: {cell.get('end_to_end', {}).get('wall_median_ms')} ms", flush=True)
                    summarize(output)
    environment["gpu_processes_after"] = command(["nvidia-smi"])
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    summarize(output, plots=True)


if __name__ == "__main__":
    main()
