"""CPU raw-Hermite regression for this experiment's axis-aligned real map.

Run in Docker after building mesh_value_probe against the intended adaptation:
python -m experiments.adaptive_distance_eval.raw_value_checks \
    --binary BUILD/mesh_value_probe --union-mesh OUT/union.obj \
    --source-map OUT/source-map.obj --output OUT/raw-value-checks

Each height layer and known counterexample runs in a fresh process: arbitrary
nearby diagnostic points must not share the construction-only nodal cache.
The oracle uses exposed rectangles, independently of the mesh triangulation.
These pointwise checks do not certify global nearest-face selection or C1.
The two fixed counterexamples apply only to the pinned seed23 source map;
other source maps retain the clear-ground checks and explicitly skip them.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .real_map_geometry import load_real_map

SEED23_SOURCE_SHA256 = "da25463989072ef10ceaa5c6cbddd55697c9c16a8d9c2e3c5df3cb5f523ba6e5"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checks(binary, union_mesh, source_map, output):
    binary, union_mesh, source_map, output = map(Path, (binary, union_mesh, source_map, output))
    output.mkdir(parents=True, exist_ok=False)
    geometry = load_real_map(source_map)
    source_hash = sha256(source_map)
    header_hashes = set()

    def probe(points, name):
        points_path, values_path = output / f"{name}-points.txt", output / f"{name}-values.txt"
        np.savetxt(points_path, points, fmt="%.17g")
        command = [str(binary.resolve()), str(union_mesh.resolve()), str(points_path.resolve()), str(values_path.resolve())]
        completed = subprocess.run(command, capture_output=True, text=True)
        (output / f"{name}-process.json").write_text(json.dumps({
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(f"mesh_value_probe failed for {name}; see process JSON")
        with values_path.open() as stream:
            header = stream.readline().split()
        if len(header) != 3 or header[:2] != ["#", "interpolation_source_sha256"] or len(header[2]) != 64:
            raise ValueError("Probe must report the compiled interpolation header hash")
        header_hashes.add(header[2])
        rows = np.loadtxt(values_path, ndmin=2)
        if rows.shape != (len(points), 12):
            raise ValueError("Unexpected probe result shape")
        if not np.array_equal(rows[:, :3], np.asarray(points, dtype=np.float32).astype(np.float64)):
            raise ValueError("Probe changed the input point sequence")
        return rows

    # Fixed world-coordinate lattice that exposed the near-zero tmd failure.
    xy = np.random.default_rng(812).integers(-2399, 2400, (1024, 2)) * .0125
    delta = np.maximum(np.maximum(geometry.box_lower[None, :, :2] - xy[:, None],
                                  xy[:, None] - geometry.box_upper[None, :, :2]), 0)
    clear = np.linalg.norm(delta, axis=2).min(axis=1) > .05
    xy = xy[clear]
    if not len(xy):
        raise ValueError("This source map has no clear-ground regression points")
    result = {
        "binary": str(binary), "binary_sha256": sha256(binary),
        "union_mesh_sha256": sha256(union_mesh), "source_map_sha256": source_hash,
        "validation_source_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("raw_value_checks.py", "mesh_value_probe.cpp")
        },
        "seed": 812, "clear_ground_xy_count": len(xy), "ground_levels": [], "counterexamples": [],
        "skipped_fixed_examples": None if source_hash == SEED23_SOURCE_SHA256 else
            "Source SHA256 differs from the pinned seed23 map; the fixed examples' geometry is not assumed. Clear-ground regressions still run.",
        "scope": "Raw mesh values only; same float32 mesh/query representation as SdfLib. No octree, GPU, or global selection/continuity certificate.",
    }
    for level in (0., 1e-7, -1e-7, 1e-5, -1e-5, .0125, -.0125):
        rows = probe(np.column_stack((xy, np.full(len(xy), level))), f"ground-{level:g}")
        expected = geometry.distance_numpy(rows[:, :3])
        if np.max(np.abs(expected - rows[:, 2])) > 1e-12:
            raise ValueError("Clear-ground analytic expectation disagrees with exposed-rectangle oracle")
        value_error = np.abs(rows[:, 8] - expected)
        gradient_error = np.abs(rows[:, 9:12] - [0., 0., 1.])
        finite = bool(np.isfinite(rows[:, 8:12]).all())
        signs = np.signbit(rows[:, 8]) == np.signbit(expected)
        sign_pass = bool(signs[np.abs(expected) > 1e-12].all())
        result["ground_levels"].append({
            "z_m": level, "point_count": len(rows), "finite": finite,
            "distance_max_error_m": float(value_error.max()) if finite else None,
            "gradient_max_component_error": float(gradient_error.max()) if finite else None,
            "nonzero_signs_pass": sign_pass,
            "passed": bool(finite and value_error.max() <= 1e-10 and gradient_error.max() <= 1e-6 and sign_pass),
        })

    examples = (
        ("ground-cancellation", (-10.1875, 14.462499618530273, 0.)),
        ("obtuse-region", (2.325000047683716, 1.225000023841858, 0.)),
    ) if source_hash == SEED23_SOURCE_SHA256 else ()
    for name, point in examples:
        row = probe(np.array([point]), name)[0]
        expected = float(geometry.distance_numpy(row[None, :3])[0])
        if name == "ground-cancellation":
            if abs(expected) > 1e-12:
                raise ValueError("Known ground example does not lie on this map's exposed ground")
            expected_gradient = np.array([0., 0., 1.])
        else:
            closest = np.clip(row[:3], geometry.rectangle_lower, geometry.rectangle_upper)
            delta = row[:3] - closest
            lengths = np.linalg.norm(delta, axis=1)
            best = lengths.argmin()
            if lengths[best] <= 1e-5:
                raise ValueError("Known obtuse example no longer has a nonzero oracle distance")
            expected_gradient = delta[best] / lengths[best] * (-1 if expected < 0 else 1)
            tied = np.abs(lengths - lengths[best]) < 1e-7
            gradients = delta[tied] / lengths[tied, None] * (-1 if expected < 0 else 1)
            if np.max(np.linalg.norm(gradients - expected_gradient, axis=1)) > 1e-4:
                raise ValueError("Known example has an ambiguous oracle gradient on this source map")
        finite = bool(np.isfinite(row[8:12]).all())
        value_error = abs(row[8] - expected)
        gradient_error = np.max(np.abs(row[9:12] - expected_gradient))
        result["counterexamples"].append({
            "name": name, "point": row[:3].tolist(), "selected_face": int(row[3]),
            "tmd_distance_diagnostic_m": float(row[4]), "expected_m": expected,
            "raw_m": float(row[8]) if finite else None,
            "distance_error_m": float(value_error) if finite else None,
            "gradient_max_component_error": float(gradient_error) if finite else None,
            "passed": bool(finite and value_error <= 3e-6 and gradient_error <= 1e-4),
        })
    if len(header_hashes) != 1:
        raise ValueError("Probe binary/header changed during the regression")
    result["interpolation_source_sha256"] = header_hashes.pop()
    result["passed"] = all(item["passed"] for item in result["ground_levels"] + result["counterexamples"])
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--union-mesh", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks(args.binary, args.union_mesh, args.source_map, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit("Raw mesh-value regression failed; see summary.json")


if __name__ == "__main__":
    main()
