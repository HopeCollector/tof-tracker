"""Docker-only actual-map evaluation; build once, then reuse for optimization."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.nn.functional import softplus

from devtools.trajectory import TrajectoryState, fit_trajectory
from .evaluate import save_json, stats as finite_stats, values_gradients


CUBE = np.array([[-51.2, -51.2, -43.7], [51.2, 51.2, 58.7]])
ROI = np.array([[-3.2, -3.2, 0.], [3.2, 3.2, 6.4]])
CONFIGURATIONS = {
    "full": {"bounds": CUBE, "start_depth": 3, "max_depth": 12},
    "roi-coarse": {"bounds": ROI, "start_depth": 5, "max_depth": 5},
    "roi-adaptive": {"bounds": ROI, "start_depth": 5, "max_depth": 8},
}


def stats(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    return {**finite_stats(values[finite]), "total_count": int(values.size),
            "nonfinite_count": int((~finite).sum())}


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def source_hashes():
    paths = sorted(Path("experiments/adaptive_distance_eval").glob("*.py"))
    paths += [Path("experiments/adaptive_distance_eval/export_sdflib.cpp"),
              Path("experiments/adaptive_distance_eval/mesh_value_probe.cpp"),
              Path("experiments/adaptive_distance_eval/run_real_map.sh"),
              Path("experiments/adaptive_distance_eval/Dockerfile"),
              Path("experiments/adaptive_distance_eval/Dockerfile.real_map"),
              Path("experiments/adaptive_distance_eval/CMakeLists.txt"),
              Path("devtools/trajectory.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def geometry_from_output(output):
    from .real_map_geometry import load_real_map
    return load_real_map(output / "source-map.obj", cube_bounds=CUBE)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.map.resolve()
    shutil.copyfile(source, args.output / "source-map.obj")
    start = time.perf_counter()
    geometry = geometry_from_output(args.output)
    geometry.write_union_obj(args.output / "union.obj")
    meta = {
        "source_map_path": str(source), "source_map_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "geometry": geometry.metadata, "geometry_preparation_seconds": time.perf_counter() - start,
        "configurations": {name: {**config, "bounds": config["bounds"].tolist(),
                                  "max_depth": args.full_max_depth if name == "full" else config["max_depth"],
                                  "termination_error_m": args.full_termination_error_m if name == "full" else .001}
                           for name, config in CONFIGURATIONS.items()},
        "source_hashes": source_hashes(),
        "source_adaptation": json.loads(args.adaptation_manifest.read_text()) if args.adaptation_manifest else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "image_id": os.environ.get("ADAPTIVE_SDF_IMAGE_ID"),
                        "git_revision": os.environ.get("ADAPTIVE_SDF_SOURCE_REVISION"),
                        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                        "python_packages": json.loads(subprocess.check_output(["python", "-m", "pip", "list", "--format=json"], text=True)),
                        "system_dependencies": subprocess.check_output(["dpkg-query", "-W", "-f=${Package} ${Version}\n", "libglm-dev", "libspdlog-dev", "libcereal-dev", "g++", "cmake", "python3-dev"], text=True).splitlines()},
    }
    save_json(args.output / "manifest.json", meta)
    print(json.dumps({"stage": "prepared", "source_sha256": meta["source_map_sha256"],
                      "geometry_seconds": meta["geometry_preparation_seconds"],
                      "geometry": {k: geometry.metadata[k] for k in ("original_pillar_count", "overlap_pair_count", "rectangle_count", "triangle_count", "closed_oriented_edge_check")}}), flush=True)


def build(args):
    if not (args.output / "manifest.json").exists():
        raise ValueError("prepare the immutable geometry first")
    manifest = json.loads((args.output / "manifest.json").read_text())
    names = list(CONFIGURATIONS) if args.field == "all" else [args.field]
    for name in names:
        config = manifest["configurations"][name]
        prefix = args.output / name
        if prefix.with_suffix(".json").exists():
            raise FileExistsError(f"refusing to replace a completed field: {prefix}")
        command = [str(args.binary), str(args.output / "union.obj"), str(prefix),
                   str(config["start_depth"]), str(config["max_depth"]), str(config["termination_error_m"]), "1", "--bounds",
                   *(str(v) for v in np.asarray(config["bounds"]).ravel()), "--query-count", "4096", "--threads", "1", "--morton-index"]
        start = time.perf_counter()
        record = {"command": command, "source_hashes": source_hashes(),
                  "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  "union_obj_sha256": hashlib.sha256((args.output / "union.obj").read_bytes()).hexdigest(),
                  "runtime_image": os.environ.get("ADAPTIVE_SDF_IMAGE_ID")}
        with (args.output / f"{name}.build.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=completed.returncode, process_wall_seconds=time.perf_counter() - start)
        save_json(args.output / f"{name}.build-process.json", record)
        print(json.dumps({"stage": "built", "field": name, "returncode": completed.returncode,
                          "process_wall_seconds": record["process_wall_seconds"]}), flush=True)
        if completed.returncode:
            raise RuntimeError(f"field build failed; retained {name}.build.log")
        adaptation = manifest.get("source_adaptation")
        if adaptation:
            exported = json.loads(prefix.with_suffix(".json").read_text())
            if exported["interpolation_source_sha256"] != adaptation["patched_header_sha256"]:
                raise RuntimeError("built field does not match the declared source adaptation")


def accuracy_probes(geometry, bounds, seed=2317):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(bounds)
    uniform = rng.uniform(lo + 1e-4, hi - 1e-4, (32768, 3))
    lower, upper = geometry.box_lower, geometry.box_upper
    surface = rng.uniform(lower[:, None, :], upper[:, None, :], (len(lower), 24, 3))
    axis = rng.integers(0, 3, (len(lower), 24))
    side = rng.integers(0, 2, (len(lower), 24))
    ii, jj = np.indices(axis.shape)
    surface[ii, jj, axis] = np.where(side, upper[ii, axis], lower[ii, axis])
    surface[ii, jj, axis] += (side * 2 - 1) * rng.uniform(-.04, .04, side.shape)
    interior = rng.uniform(lower[:, None, :], upper[:, None, :], (len(lower), 12, 3))
    overlap = []
    for a, b in geometry.overlap_pairs:
        overlap.append(rng.uniform(np.maximum(lower[a], lower[b]), np.minimum(upper[a], upper[b]), (16, 3)))
    ground = rng.uniform(lo + 1e-4, hi - 1e-4, (2048, 3))
    ground[:, 2] = rng.uniform(max(lo[2], 0.) + 1e-5, min(hi[2], .05), len(ground))
    groups = [("uniform", uniform), ("near_original_faces", surface.reshape(-1, 3)),
              ("inside_original_boxes", interior.reshape(-1, 3)), ("near_ground", ground)]
    if overlap:
        groups.append(("overlap", np.concatenate(overlap)))
    points, labels = [], []
    for label, values in groups:
        values = values[((values >= lo).all(1) & (values <= hi).all(1))]
        points.append(values)
        labels.extend([label] * len(values))
    return np.concatenate(points), np.asarray(labels)


def check_float32_accuracy(field, geometry, points, destination):
    """Check the throughput dtype against geometry at its actual stored inputs."""
    actual_points = np.asarray(points, dtype=np.float32)
    predicted, gradient = values_gradients(field, actual_points, dtype=torch.float32, chunk=4096)
    with torch.no_grad():
        expected = geometry.distance_torch(torch.tensor(actual_points, dtype=torch.float64, device=field.device)).cpu().numpy()
    error = predicted - expected
    finite = bool(np.isfinite(predicted).all() and np.isfinite(gradient).all())
    np.savez_compressed(destination, points=actual_points, expected=expected, predicted=predicted, gradient=gradient)
    return {"dtype": "float32 query/evaluation; float64 geometry at the identical float32-represented input",
            "distance_error_m": stats(error), "maximum_overestimate_m": finite_number(error.max()),
            "sign_misses_beyond_1mm": int(np.count_nonzero((np.abs(expected) > .001) & ((predicted < 0) != (expected < 0)))),
            "maximum_field_gradient_norm": finite_number(np.linalg.norm(gradient, axis=1).max()),
            "finite": finite, "source_hashes": source_hashes()}


def check_accuracy(field, geometry, bounds, destination):
    points, labels = accuracy_probes(geometry, bounds)
    predicted, gradient = values_gradients(field, points, dtype=torch.float64, chunk=4096)
    with torch.no_grad():
        expected = geometry.distance_torch(torch.tensor(points, dtype=torch.float64, device=field.device)).cpu().numpy()
    error = predicted - expected
    def summarize(mask):
        if not mask.any():
            return {"count": 0}
        return {"distance_error_m": stats(error[mask]), "maximum_overestimate_m": finite_number(error[mask].max()),
                "sign_misses_beyond_1mm": int(np.count_nonzero((np.abs(expected[mask]) > .001) & ((predicted[mask] < 0) != (expected[mask] < 0)))),
                "maximum_field_gradient_norm": finite_number(np.linalg.norm(gradient[mask], axis=1).max())}
    result = {"count": len(points), "all": summarize(np.ones(len(points), bool)),
              "by_probe_group": {label: summarize(labels == label) for label in np.unique(labels)},
              "by_geometry": {"free_near_1m": summarize((expected >= 0) & (expected < 1)),
                              "free_far": summarize(expected >= 1), "inside": summarize(expected < 0)},
              "finite": bool(np.isfinite(predicted).all() and np.isfinite(gradient).all()),
              "strict_global_error_certificate": False,
              "oracle": "float64 analytic exposed-rectangle distance and original-box/ground union occupancy"}
    worst = np.argsort(np.abs(error))[-12:][::-1]
    result["worst_probes"] = [{"point": points[i].tolist(), "expected_m": finite_number(expected[i]),
                                "field_m": finite_number(predicted[i]), "error_m": finite_number(error[i]), "group": str(labels[i])} for i in worst]
    np.savez_compressed(destination, points=points, expected=expected, predicted=predicted,
                        gradient=gradient, labels=labels)
    result["evaluation_dtype"] = "float64 query/evaluation, fixed float32 exported coefficients/origins"
    result["float32_query_path"] = check_float32_accuracy(field, geometry, points, destination.with_name(destination.stem + "-float32.npz"))
    result["finite"] = result["finite"] and result["float32_query_path"]["finite"]
    return result


def check_native(field, prefix):
    rows = np.fromfile(str(prefix) + ".queries.bin", dtype="<f4").reshape(-1, 11)
    value, gradient = values_gradients(field, rows[:, :3], chunk=2048)
    selected = field.leaf_ids(torch.tensor(rows[:, :3], dtype=torch.float64, device=field.device)).cpu().numpy()
    integer_ids_path = Path(str(prefix) + ".query_leaf_ids.bin")
    native_ids = np.fromfile(integer_ids_path, dtype="<u4") if integer_ids_path.exists() else rows[:, 10].astype(np.int64)
    result = {"count": len(rows), "value_max_error_m": float(np.max(np.abs(value - rows[:, 3]))),
              "raw_gradient_max_error": float(np.max(np.abs(gradient - rows[:, 7:10]))),
              "leaf_id_mismatches": int(np.count_nonzero(selected != native_ids)),
              "tolerances": {"value_m": .0001, "raw_gradient_linf": .01, "leaf_id_mismatches": 0}}
    result["passed"] = bool(result["value_max_error_m"] < .0001 and result["raw_gradient_max_error"] < .01 and result["leaf_id_mismatches"] == 0)
    return result


def sampled_seams(field, destination, count=8192):
    """Finite same-point two-polynomial probes; no dense topology allocation."""
    rng = np.random.default_rng(2319)
    lo, hi = (value.cpu().numpy() for value in field.logical_bounds)
    candidates = rng.integers(0, len(field.rows), size=count * 16)
    rows = field.rows[candidates]
    centres = rows[:, :3].astype(np.float64) + rows[:, 3:4] / 2
    keep = ((centres > lo) & (centres < hi)).all(1)
    candidates, rows = candidates[keep][:count], rows[keep][:count]
    domain_lo, domain_hi = (value.cpu().numpy() for value in field.domain_bounds)
    lattice_step = (domain_hi-domain_lo) / field.n
    integer_lo = np.rint((rows[:, :3].astype(np.float64)-domain_lo)/lattice_step)
    widths = np.exp2(field.max_depth-rows[:, 4].astype(int))
    canonical_lo = domain_lo + integer_lo*lattice_step
    sizes = widths[:, None]*lattice_step
    axes = rng.integers(0, 3, len(rows))
    signs = rng.integers(0, 2, len(rows))
    points = canonical_lo + rng.uniform(.1, .9, (len(rows), 3))*sizes
    indices = np.arange(len(rows))
    points[indices, axes] = canonical_lo[indices, axes] + signs*sizes[indices, axes]
    epsilon = np.minimum(sizes[:, 0]*1e-4, field.finest_size*1e-3)
    offset = np.zeros_like(points)
    offset[indices, axes] = epsilon
    pt = torch.tensor(points, dtype=torch.float64, device=field.device)
    left = field.leaf_ids(pt-torch.tensor(offset, device=field.device))
    right = field.leaf_ids(pt+torch.tensor(offset, device=field.device))
    valid = (left >= 0) & (right >= 0) & (left != right)
    pt, left, right = pt[valid].requires_grad_(True), left[valid], right[valid]
    values, gradients = [], []
    for ids in (left, right):
        value = field.evaluate_leaf(pt, ids)
        gradient, = torch.autograd.grad(value.sum(), pt)
        values.append(value.detach().cpu().numpy())
        gradients.append(gradient.cpu().numpy())
    v0, v1 = values
    g0, g1 = gradients
    jump = np.linalg.norm(g0-g1, axis=1)
    norms = np.linalg.norm(np.stack(gradients), axis=-1)
    denominator = norms[0]*norms[1]
    good_angle = denominator > 1e-8
    angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", g0[good_angle], g1[good_angle])/denominator[good_angle], -1, 1)))
    np.savez_compressed(destination, points=pt.detach().cpu().numpy(), values=np.stack(values),
                        gradients=np.stack(gradients), leaf_ids=np.stack((left.cpu().numpy(), right.cpu().numpy())))
    worst = int(np.argmax(jump))
    return {"sample_count": len(pt), "value_jump_m": stats(v0-v1), "gradient_l2_jump": stats(jump),
            "angle_degrees_nonzero_gradients": stats(angle), "same_level": int((field.rows[left.cpu().numpy(), 4] == field.rows[right.cpu().numpy(), 4]).sum()),
            "worst_gradient_jump": {"point": pt[worst].detach().cpu().tolist(), "gradient_left": g0[worst].tolist(),
                                     "gradient_right": g1[worst].tolist(), "distance_left_m": float(v0[worst]), "distance_right_m": float(v1[worst])},
            "scope": "seeded subset of leaf-face interior points in logical domain, both polynomials at identical points; not exhaustive or a continuity certificate"}


def measure(call, *, repetitions=10, warmup=3):
    torch.cuda.synchronize()
    start = time.perf_counter()
    call()
    torch.cuda.synchronize()
    first = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"first_call_ms": first, "samples_ms": samples, "median_ms": float(np.median(samples)),
            "p95_ms": float(np.quantile(samples, .95))}


def point_benchmarks(field):
    lo, hi = (x.float() for x in field.logical_bounds)
    result = {"dtype": "float32", "device": str(field.device), "query_only": {}, "locate_only": {}, "distance_cost_backward": {},
              "scope": "resident field and query tensors; synchronized wall time; no upload/build; cost=mean(softplus((.3-d)/.1)^2)"}
    generator = torch.Generator(device=field.device).manual_seed(2318)
    for count in (1, 1024, 65536, 100000, 262144):
        points = (lo + (hi-lo)*torch.rand((count, 3), generator=generator, device=field.device)).requires_grad_(True)
        def forward():
            with torch.no_grad():
                return field.query(points)
        def backward():
            points.grad = None
            d = field.query(points)
            softplus((.3-d)/.1).square().mean().backward()
        result["locate_only"][str(count)] = measure(lambda: field.leaf_ids(points))
        result["query_only"][str(count)] = measure(forward)
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
        result["distance_cost_backward"][str(count)] = measure(backward)
        result["distance_cost_backward"][str(count)]["peak_extra_allocated_bytes"] = torch.cuda.max_memory_allocated() - memory_before
        del points
    return result


def trajectory_benchmarks(field):
    result = {"dtype": "float32", "samples_per_trajectory": 96, "batches": {},
              "scope": "synthetic repeated map-contained boundary states for throughput; not extra independent optimization cases"}
    device = field.device
    fractions = torch.linspace(0, 1, 96, device=device)
    for count in (1, 64, 256, 1024):
        q = torch.zeros((count, 10), device=device, requires_grad=True)
        initial = TrajectoryState(torch.tensor([-2., 0., 3.], device=device).expand(count, 3),
                                  torch.tensor([2., .1, .05], device=device).expand(count, 3),
                                  torch.zeros((count, 3), device=device), torch.zeros(count, device=device), torch.zeros(count, device=device))
        def decode():
            duration = 2. * (1 + .2*q[:, 9])
            terminal = TrajectoryState(torch.tensor([2., 0., 3.], device=device) + .5*q[:, :3],
                                       torch.tensor([2., .1, .05], device=device) + 2*q[:, 3:6], 2*q[:, 6:9],
                                       torch.zeros(count, device=device), torch.zeros(count, device=device))
            return fit_trajectory(initial, terminal, duration).sample(duration[:, None] * fractions).position
        def forward_decode():
            with torch.no_grad():
                return decode()
        def chain():
            q.grad = None
            d = field.query(decode().reshape(-1, 3))
            softplus((.3-d)/.1).square().mean().backward()
        result["batches"][str(count)] = {"decode_only": measure(forward_decode), "decode_query_cost_backward": measure(chain)}
    return result


def compiled_point_benchmarks(field):
    """Measure optional query fusion separately; the optimizer uses eager query."""
    result = {"mode": "torch.compile(fullgraph=True,dynamic=False), query polynomial and VJP only",
              "optimizer_uses_this_mode": False, "batches": {},
              "scope": "same resident float32 map; same obstacle penalty as eager point benchmark; first call includes compilation"}
    compiled = torch.compile(field.query, fullgraph=True, dynamic=False)
    lo, hi = (value.float() for value in field.logical_bounds)
    generator = torch.Generator(device=field.device).manual_seed(2320)
    try:
        for count in (65536, 100000):
            points = (lo + (hi-lo)*torch.rand((count, 3), generator=generator, device=field.device)).requires_grad_(True)
            expected = field.query(points)
            expected_gradient, = torch.autograd.grad(expected.sum(), points)
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual = compiled(points)
            actual_gradient, = torch.autograd.grad(actual.sum(), points)
            torch.cuda.synchronize()
            first_seconds = time.perf_counter()-start
            distance_error = float((actual-expected).abs().max().detach())
            gradient_error = float((actual_gradient-expected_gradient).abs().max().detach())
            parity = {"first_forward_backward_seconds": first_seconds,
                      "distance_max_abs_m": finite_number(distance_error),
                      "world_gradient_max_abs": finite_number(gradient_error),
                      "finite": bool(np.isfinite(distance_error) and np.isfinite(gradient_error))}
            parity["passed"] = bool(parity["finite"] and distance_error <= 1e-4 and gradient_error <= .002)
            if not parity["passed"]:
                result["batches"][str(count)] = {"eager_parity": parity, "timing_skipped": "compiled value/gradient parity failed"}
                continue
            def run():
                points.grad = None
                distance = compiled(points)
                softplus((.3-distance)/.1).square().mean().backward()
            result["batches"][str(count)] = {"eager_parity": parity, "distance_cost_backward": measure(run)}
        result["completed"] = True
    except Exception as error:
        result["completed"] = False
        result["error"] = repr(error)
    finally:
        torch._dynamo.reset()
    return result


def evaluate(args):
    from .batch_optimization import run_batch_optimization
    from .sparse_field import SparseAdaptiveField
    if not torch.cuda.is_available():
        raise RuntimeError("this throughput evaluation requires CUDA; no silent CPU fallback")
    torch.set_num_threads(2)
    geometry = geometry_from_output(args.output)
    manifest = json.loads((args.output / "manifest.json").read_text())
    requested = list(CONFIGURATIONS) if args.field == "all" else [args.field]
    results_path = args.output / ("results.json" if args.field == "all" else f"{args.field}-evaluation.json")
    result = {"experiment": "shared-node SdfLib actual-map batch optimization", "manifest": manifest,
              "requested_fields": requested, "fields": {}, "environment": {"python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "image_id": os.environ.get("ADAPTIVE_SDF_IMAGE_ID"), "source_hashes": source_hashes()},
              "semantics": {"geometry": "actual 500-pillar map; exposed union boundary with ground half-space equivalence on logical domain",
                            "full_map_start_cell_m": 12.8,
                            "full_map_min_cell_m": float(np.ptp(np.asarray(manifest["configurations"]["full"]["bounds"]), axis=0)[0] / 2**manifest["configurations"]["full"]["max_depth"]),
                            "full_map_termination_error_m": manifest["configurations"]["full"]["termination_error_m"],
                            "roi_start_cell_m": .2, "roi_min_cell_m": .025,
                            "trajectory": "finite sampled cost; dense independent probes are observations, not continuous clearance proof"}}
    for name in requested:
        prefix = args.output / name
        field = SparseAdaptiveField(prefix)
        native = check_native(field, prefix)
        metadata = field.metadata
        del field
        gc.collect()
        logical = geometry.logical_bounds if name == "full" else ROI
        torch.cuda.synchronize()
        start = time.perf_counter()
        field = SparseAdaptiveField(prefix, "cuda", logical_bounds=logical)
        torch.cuda.synchronize()
        loaded = time.perf_counter() - start
        entry = {"export_metadata": metadata, "native_parity": native, "host_load_and_upload_seconds": loaded,
                 "resident_allocated_bytes_after_load": torch.cuda.memory_allocated(),
                 "accuracy": check_accuracy(field, geometry, logical, args.output / f"{name}-accuracy.npz")}
        entry["sampled_seams"] = sampled_seams(field, args.output / f"{name}-sampled-seams.npz")
        result["fields"][name] = entry
        save_json(results_path, result)
        print(json.dumps({"stage": "accuracy", "field": name, "native": native, "distance": entry["accuracy"]["all"]}), flush=True)
        if name == "full":
            start = time.perf_counter()
            entry["optimization"] = run_batch_optimization(field, geometry.distance_torch, args.output / "optimization", count=args.trajectories, steps=args.steps, device="cuda")
            torch.cuda.synchronize()
            entry["optimization_total_wall_seconds"] = time.perf_counter() - start
            entry["point_benchmarks"] = point_benchmarks(field)
            entry["trajectory_benchmarks"] = trajectory_benchmarks(field)
            entry["compiled_point_benchmarks"] = compiled_point_benchmarks(field)
        del field
        gc.collect()
        torch.cuda.empty_cache()
        save_json(results_path, result)
    result["completed"] = True
    result["assessment"] = {"all_native_parity_passed": all(f["native_parity"]["passed"] for f in result["fields"].values()),
                             "all_accuracy_values_and_gradients_finite": all(f["accuracy"]["finite"] for f in result["fields"].values()),
                             "batch_optimization_pipeline_finite": result["fields"].get("full", {}).get("optimization", {}).get("pipeline_passed"),
                             "strict_global_error_or_continuous_safety_certificate": False}
    save_json(results_path, result)
    print(json.dumps({"stage": "complete", "assessment": result["assessment"]}), flush=True)
    gates = (result["assessment"]["all_native_parity_passed"], result["assessment"]["all_accuracy_values_and_gradients_finite"])
    if not all(gates) or result["assessment"]["batch_optimization_pipeline_finite"] is False:
        raise SystemExit("integration failed; complete numerical evidence retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "build", "evaluate"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--adaptation-manifest", type=Path)
    parser.add_argument("--full-termination-error-m", type=float, default=.001,
                        help="prepare only: persist the full-map error heuristic in the immutable manifest; ROI remains .001m")
    parser.add_argument("--full-max-depth", type=int, default=12,
                        help="prepare only: cap full-map refinement (9 gives 0.2m; 12 gives 0.025m); ROI unchanged")
    parser.add_argument("--field", choices=(*CONFIGURATIONS, "all"), default="all")
    parser.add_argument("--trajectories", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if not np.isfinite(args.full_termination_error_m) or args.full_termination_error_m <= 0:
        parser.error("full termination error must be finite and positive")
    if not 3 <= args.full_max_depth <= 12:
        parser.error("full max depth must lie in [3,12]")
    if not Path("/.dockerenv").exists():
        raise RuntimeError("run inside Docker")
    if args.stage == "prepare" and args.map is None:
        parser.error("prepare requires --map")
    if args.stage == "build" and args.binary is None:
        parser.error("build requires --binary")
    {"prepare": prepare, "build": build, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
