"""Build and run the read-only reference inside its existing CUDA Docker image."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("/reference"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    include = args.reference / "Simulator/src/include"
    source = args.reference / "Simulator/src/src/sensor_simulator.cu"
    harness = ROOT / "experiments/reference_tof_perf/benchmark.cu"
    binary = args.output / "benchmark"
    pkg = subprocess.check_output(["pkg-config", "--cflags", "--libs", "opencv4"], text=True)
    build = ["nvcc", "-O3", "-std=c++17", "-arch=sm_120", "--expt-relaxed-constexpr",
             "-I"+str(include), "-I/usr/include/pcl-1.10", "-I/usr/include/eigen3",
             str(harness), str(source), "-o", str(binary), "-lpcl_common", *shlex.split(pkg)]
    ref_files = [source, include / "sensor_simulator.cuh",
                 args.reference / "Simulator/src/src/test_simulator_cuda.cpp",
                 args.reference / "Simulator/src/config/config.yaml", args.reference / "LICENSE",
                 *sorted((include / "cuda_toolkit").glob("*"))]
    provenance = {"source_commit_verified_on_host": args.source_commit,
                  "reference_files_sha256": {str(p.relative_to(args.reference)): digest(p) for p in ref_files if p.is_file()},
                  "experiment_files_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (harness, Path(__file__), harness.parent / "prepare_fixture.py")},
                  "build_command": build,
                  "compiler": subprocess.check_output(["nvcc", "--version"], text=True),
                  "image": "yopo-local/c5-tracking:dev",
                  "image_id_verified_on_host": "sha256:1a029c5f6d21cb12b74780dc9453bf91ff6c56f269c6f94d7b5ff81a5306a5a6",
                  "reference_source_mount": "read-only",
                  "gpu_sharing": "Shared desktop GPU with existing online review service; no simultaneous agent GPU benchmark. User/desktop activity is not controlled.",
                  "scope": "Native reference implementation/CPU output; serial API at B>1. Scene is occupancy-discretized from the existing public geometry and includes the target sphere. No rendering/geometry equivalence to current continuous-zone renderer is claimed.",
                  "fixtures": json.loads((ROOT / "outputs/reference_tof_perf/fixtures/fixtures.json").read_text())}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("BUILD", flush=True)
    with (args.output / "build.log").open("w") as log:
        subprocess.run(build, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    provenance["binary_sha256"] = digest(binary)
    records = []
    for count in (20, 5000):
        for label, fov, distance in (("reference-default", 45, 4), ("matched-fov-range", 60, 12)):
            for batch in (1, 64):
                name = f"pillars{count}-{label}-b{batch}"
                path = args.output / (name + ".json")
                command = [str(binary), "--cloud", f"outputs/reference_tof_perf/fixtures/pillars-{count}-occupancy-f32-xyz.bin",
                           "--poses", "outputs/reference_tof_perf/fixtures/poses-f32-xyz-xyzw.bin",
                           "--output", str(path), "--batch", str(batch), "--fov", str(fov),
                           "--max-range", str(distance), "--resolution", "0.1", "--warmup", "5",
                           "--repeats", str(200 if batch == 1 else 30)]
                print("START " + name, flush=True)
                start = time.perf_counter()
                with (args.output / (name + ".log")).open("w") as log:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                result = json.loads(path.read_text()) if path.exists() else {}
                records.append({"name": name, "command": command, "returncode": completed.returncode,
                                "elapsed_s": time.perf_counter()-start, "result": str(path),
                                "status": result.get("status", "missing_result")})
                (args.output / "manifest.json").write_text(json.dumps(records, indent=2)+"\n")
                if completed.returncode != 0 or result.get("status") != "ok":
                    raise RuntimeError("cell failed: " + name)
                print("END " + name + " full_ms=" + str(result["stages"]["end_to_end_wall"]["median_ms"]), flush=True)
    after = {str(p.relative_to(args.reference)): digest(p) for p in ref_files if p.is_file()}
    assert after == provenance["reference_files_sha256"], "reference sources changed"
    provenance["reference_hashes_unchanged"] = True
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"cells": len(records), "status": "ok"}), flush=True)


if __name__ == "__main__":
    main()
