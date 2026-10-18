"""Docker-only end-to-end evidence for actual SdfLib leaves and Torch queries."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .field import AdaptiveField
from .fixtures import cases, exhaustive_triangle_distance, reference, write_case
from .seam_checks import run_seam_checks
from .trajectory_checks import run_trajectory_checks


def stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "max_abs": float(np.abs(values).max()),
            "p95_abs": float(np.quantile(np.abs(values), .95)),
            "rmse": float(np.sqrt(np.mean(values**2))), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max())}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def values_gradients(field, points, *, dtype=torch.float64, chunk=8192):
    values, gradients = [], []
    for block in np.array_split(points, max(1, int(np.ceil(len(points) / chunk)))):
        p = torch.tensor(block, dtype=dtype, device=field.device, requires_grad=True)
        d = field.query(p)
        g, = torch.autograd.grad(d.sum(), p)
        values.append(d.detach().cpu().numpy())
        gradients.append(g.detach().cpu().numpy())
    return np.concatenate(values), np.concatenate(gradients)


def check_geometry(field, boxes, geometry, output, name):
    rng = np.random.default_rng(20261017)
    points = rng.uniform(-.799, .799, size=(16384, 3))
    # Extra original-surface and interior probes prevent all-random tests from
    # overlooking signs or a thin solid between the 0.2 m start-grid vertices.
    extras = []
    for box in boxes:
        local = rng.uniform(-1., 1., size=(2048, 3)) * np.asarray(box.half)
        extras.append(local @ np.asarray(box.rotation).T + box.center)
    points = np.concatenate((points, *extras))
    expected, expected_gradient, unique = reference(points, boxes)
    values, gradient = values_gradients(field, points)
    tri = exhaustive_triangle_distance(points[:512], geometry)
    oracle_error = float(np.max(np.abs(tri - np.abs(expected[:512]))))
    difference = values - expected
    gerror = gradient[unique] - expected_gradient[unique]
    interior = expected < -1e-5
    strong_sign = np.abs(expected) > .005
    indices = np.argsort(np.abs(difference))[-16:][::-1]
    records = [{"point": points[i].tolist(), "expected_m": float(expected[i]),
                "field_m": float(values[i]), "error_m": float(difference[i])} for i in indices]
    result = {
        "point_count": len(points), "oracle_triangle_abs_distance_max_error_m": oracle_error,
        "distance_error_m": stats(difference), "maximum_overestimate_m": float(np.max(difference)),
        "interior_distance_error_m": stats(difference[interior]),
        "interior_sign_misses": int(np.count_nonzero(interior & (values >= 0))),
        "sign_misses_beyond_5mm": int(np.count_nonzero(strong_sign & ((values < 0) != (expected < 0)))),
        "gradient_component_error_unique_reference": stats(gerror),
        "gradient_reference_count": int(unique.sum()),
        "gradient_reference_rule": "exclude |d|<=1um and a 2mm interior/union medial tie band; no claim that this masks all interpolation influence from kinks",
        "worst_distance_probes": records,
        "distance_10mm_budget_observed_pass": bool(np.max(np.abs(difference)) <= .01),
        "finite": bool(np.isfinite(values).all() and np.isfinite(gradient).all()),
        "independent_oracle_pass": bool(oracle_error < 1e-12),
        "strict_global_error_certificate": False,
    }
    f32_value, f32_gradient = values_gradients(field, points[:2048], dtype=torch.float32)
    # Compare at exactly the same float32-representable coordinates; otherwise
    # input quantization can conceal an independent arithmetic discrepancy.
    q32 = points[:2048].astype(np.float32).astype(np.float64)
    f64_value, f64_gradient = values_gradients(field, q32)
    result["float32_vs_float64_at_identical_represented_points"] = {
        "distance_error_m": stats(f32_value-f64_value),
        "gradient_component_error": stats(f32_gradient-f64_gradient)}
    np.savez_compressed(output / f"{name}-geometry.npz", points=points, expected=expected,
                        values=values, gradient=gradient, expected_gradient=expected_gradient,
                        unique_gradient_mask=unique)
    return result


def check_native(field, prefix):
    rows = np.fromfile(Path(str(prefix) + ".queries.bin"), dtype="<f4").reshape(-1, 11)
    value, gradient = values_gradients(field, rows[:, :3])
    value_error = np.max(np.abs(value - rows[:, 3]))
    gradient_error = np.max(np.abs(gradient - rows[:, 7:10]))
    normalized = rows[:, 4:7]
    finite = np.isfinite(normalized).all(1)
    normalized_difference = stats((gradient[finite] - normalized[finite]).ravel())
    selected = field.leaf_ids(torch.tensor(rows[:, :3], dtype=torch.float64, device=field.device)).cpu().numpy()
    lookup_mismatches = int(np.count_nonzero(selected != rows[:, 10].astype(np.int64)))
    return {
        "count": len(rows), "value_max_error_m": float(value_error),
        "raw_world_gradient_max_error": float(gradient_error),
        "native_unit_gradient_nonfinite_count": int((~finite).sum()),
        "true_derivative_minus_native_unit_gradient": normalized_difference,
        "native_lookup_mismatch_count": lookup_mismatches,
        "passed": bool(value_error < 2e-5 and gradient_error < .002 and lookup_mismatches == 0),
        "note": "native normalized gradient is diagnostic, never used as the training VJP",
    }


def check_query_contract(field):
    lo, hi = field.bounds
    middle = (lo + hi) / 2
    points = torch.stack((middle, lo, hi, middle + torch.tensor([10.,0,0], device=field.device),
                          torch.tensor([float('nan'),0,0], device=field.device),
                          torch.tensor([float('inf'),0,0], device=field.device))).requires_grad_(True)
    d, ids = field.query(points, return_leaf_ids=True)
    grad, = torch.autograd.grad(torch.nan_to_num(d).sum(), points)
    passed = (bool(torch.isfinite(d[:3]).all()) and bool(torch.isnan(d[3:]).all())
              and bool((ids[3:] == -1).all()) and bool((grad[3:] == 0).all()))
    return {"passed": passed, "domain_policy": "inclusive declared bounds; outside/nonfinite -> NaN, leaf=-1, zero query gradient; no padding obstacles"}


def check_ridge(field, output):
    x = np.linspace(-.1, .1, 2001)
    points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    values, gradient = values_gradients(field, points)
    true = .54 - np.abs(x)
    # The previously discussed controlled C1 patch is only an analytic
    # reference here. Upstream SdfLib is not claimed to construct this patch.
    delta = .02
    controlled = np.where(np.abs(x) <= delta, .54-delta/2-x*x/(2*delta), true)
    controlled_g = np.where(np.abs(x) <= delta, -x/delta, -np.sign(x))
    np.savez_compressed(output / "ridge-section.npz", x=x, raw_mesh=true,
                        field=values, field_gradient_x=gradient[:,0],
                        controlled_reference=controlled, controlled_reference_gradient_x=controlled_g)
    return {"raw_mesh_error_m": stats(values-true),
            "controlled_reference_error_m": stats(values-controlled),
            "field_gradient_x_range": [float(gradient[:,0].min()), float(gradient[:,0].max())],
            "center": {"raw_distance_m": .54, "field_distance_m": float(values[len(x)//2]),
                       "field_gradient": gradient[len(x)//2].tolist()},
            "controlled_reference_half_width_m": delta,
            "controlled_reference_max_bias_m": delta/2,
            "controlled_smoothing_implemented_by_library": False}


def timing(field, count=65536):
    generator = torch.Generator(device=field.device).manual_seed(1710)
    points = (torch.rand((count,3), generator=generator, device=field.device) * 1.598 - .799).requires_grad_(True)
    def sync():
        if field.device.type == "cuda":
            torch.cuda.synchronize(field.device)
    samples = []
    for iteration in range(13):
        points.grad = None
        sync()
        start = time.perf_counter()
        value = field.query(points)
        torch.exp(-value).mean().backward()
        sync()
        if iteration >= 3:
            samples.append((time.perf_counter() - start) * 1000)
    return {"count":count, "dtype":"float32", "scope":"resident field/query; forward + mean(exp(-d)) + position backward, synchronized; lookup table construction excluded",
            "samples_ms":samples, "median_ms":float(np.median(samples)), "p95_ms":float(np.quantile(samples,.95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(cases()), default=list(cases()))
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--variant", choices=("original", "shared-nodes"), default="original")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("run inside Docker")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    summary = {"experiment":"SdfLib adaptive distance field reuse", "cases":{},
               "environment":{"python":platform.python_version(), "torch":torch.__version__,
                              "numpy":np.__version__, "cuda_available":torch.cuda.is_available(),
                              "cuda":torch.version.cuda, "image_id":os.environ.get("ADAPTIVE_SDF_IMAGE_ID"),
                              "source_revision":os.environ.get("ADAPTIVE_SDF_SOURCE_REVISION"),
                              "upstream_revision":"8db373ef71d65be24badf6ae10750a932bbc223b"},
               "semantics":{"geometry":"small disjoint closed box meshes; no overlap-union, open ground or dynamic object implementation",
                            "field":"actual SdfLib tricubic OctreeSdf; start cell 0.2m; native distance-only adaptive termination",
                            "smoothing":"library's CONTINUITY mode evaluated, with measured seam failures; no explicit narrow-band bias/width policy added",
                            "query":"Torch polynomial evaluation with fixed coefficients, dense finest-lattice integer leaf lookup",
                            "trajectory":"main 99b9ead fit_trajectory; normalized u coefficients; sampled cost for gradient validation only"}}
    summary["variant"] = args.variant
    summary["environment"]["python_packages"] = json.loads(subprocess.check_output(["python","-m","pip","list","--format=json"],text=True))
    summary["environment"]["system_dependencies"] = subprocess.check_output(
        ["dpkg-query","-W","-f=${Package} ${Version}\n","libglm-dev","libspdlog-dev","libcereal-dev","g++","cmake"],text=True).splitlines()
    if torch.cuda.is_available():
        summary["environment"]["gpu"] = torch.cuda.get_device_name(0)
        summary["environment"]["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    for name in args.cases:
        boxes = cases()[name]
        mesh_path, geometry = write_case(args.output, name, boxes)
        configurations = [("coarse",3,1e-3,1), ("adaptive",args.max_depth,1e-3,1)]
        if name == "box":
            configurations += [("tight",args.max_depth,2.5e-4,1), ("unconstrained",args.max_depth,1e-3,0)]
        case_result = {"boxes":[asdict(b) for b in boxes], "fields":{}}
        for label, maximum, error, continuity in configurations:
            prefix = args.output / f"{name}-{label}"
            command = [str(args.binary), str(mesh_path), str(prefix), "3", str(maximum), str(error), str(continuity), "--query-count", "2048", "--threads", str(args.threads)]
            start = time.perf_counter()
            with Path(str(prefix) + ".build.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            build_ms = (time.perf_counter() - start)*1000
            field = AdaptiveField(prefix)
            expected_header = {
                "original": "c258cab51c35f96116dd98d43a946fb474fbcbee318111c3cfa43fb0d83df265",
                "shared-nodes": "5cef4745a46ed0a44b47c3a85fc93fdf4e0e4b558a433fcc7227905e67641396",
            }[args.variant]
            if field.metadata["interpolation_source_sha256"] != expected_header:
                raise RuntimeError("exported source hash does not match the requested variant")
            result = {"configuration":{"start_cell_m":.2,"max_depth":maximum,"expected_rmse_m":error,"continuity":bool(continuity)},
                      "build_process_wall_ms":build_ms, "export_metadata":field.metadata,
                      "leaf_depth_histogram":{str(int(d)):int((field.rows[:,4] == d).sum()) for d in np.unique(field.rows[:,4])},
                      "leaf_payload_bytes":int(field.rows.nbytes), "lookup_table_bytes":int(field.lookup.numel()*field.lookup.element_size()),
                      "geometry":check_geometry(field,boxes,geometry,args.output,f"{name}-{label}"),
                      "native_parity":check_native(field,prefix), "query_contract":check_query_contract(field)}
            if label == "adaptive":
                result["seams"] = run_seam_checks(field,args.output / f"{name}-seams",max_faces=None)
            if name == "box" and label in ("adaptive","unconstrained"):
                if label == "unconstrained":
                    result["seams"] = run_seam_checks(field,args.output / "box-unconstrained-seams",max_faces=None)
                else:
                    result["trajectory_cpu"] = run_trajectory_checks(field,"cpu",args.output / "trajectory-cpu")
                    result["timing_cpu"] = timing(field)
                    if torch.cuda.is_available():
                        gpu = AdaptiveField(prefix,"cuda")
                        result["trajectory_cuda"] = run_trajectory_checks(gpu,"cuda",args.output / "trajectory-cuda")
                        result["timing_cuda"] = timing(gpu)
                        result["cuda_query_contract"] = check_query_contract(gpu)
                        gd, gg = values_gradients(gpu,np.random.default_rng(42).uniform(-.799,.799,(2048,3)))
                        cd, cg = values_gradients(field,np.random.default_rng(42).uniform(-.799,.799,(2048,3)))
                        result["cpu_cuda_float64_parity"] = {"distance_max_abs":float(np.max(np.abs(gd-cd))),"gradient_max_abs":float(np.max(np.abs(gg-cg)))}
                        g32d,g32g = values_gradients(gpu,np.random.default_rng(42).uniform(-.799,.799,(2048,3)),dtype=torch.float32)
                        result["cuda_float32_vs_float64"] = {"distance_max_abs":float(np.max(np.abs(g32d-gd))),"gradient_max_abs":float(np.max(np.abs(g32g-gg))),
                                                             "note":"includes float32 input quantization"}
            if name == "ridge" and label == "adaptive":
                result["ridge_section"] = check_ridge(field,args.output)
            case_result["fields"][label] = result
            summary["cases"][name] = case_result
            save_json(args.output / "results.json",summary)
            print(json.dumps({"case":name,"field":label,"leaves":len(field.rows),"distance_max_m":result["geometry"]["distance_error_m"]["max_abs"],"native_parity":result["native_parity"]["passed"]}),flush=True)
    paths = sorted(Path("docs/explorations/esdf/adaptive_distance_eval/code").glob("*")) + [Path("devtools/trajectory.py")]
    summary["source_hashes"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    summary["completed"] = True
    summary["assessment"] = {
        "integration_passed":all(v["native_parity"]["passed"] and v["query_contract"]["passed"] and v["geometry"]["independent_oracle_pass"] and v["geometry"]["finite"]
                                 and v.get("cuda_query_contract",{"passed":True})["passed"]
                                 and v.get("trajectory_cpu",{"passed":True})["passed"] and v.get("trajectory_cuda",{"passed":True})["passed"]
                                 for c in summary["cases"].values() for v in c["fields"].values()),
        "adaptive_seam_acceptance_passed":all(c["fields"]["adaptive"]["seams"]["passed"] for c in summary["cases"].values()),
        "adaptive_observed_distance_10mm_passed":all(c["fields"]["adaptive"]["geometry"]["distance_10mm_budget_observed_pass"] for c in summary["cases"].values()),
        "controlled_smoothing_policy_implemented":False,
    }
    save_json(args.output / "results.json",summary)
    if not summary["assessment"]["integration_passed"]:
        raise SystemExit("integration checks failed; retained all results")


if __name__ == "__main__":
    main()
