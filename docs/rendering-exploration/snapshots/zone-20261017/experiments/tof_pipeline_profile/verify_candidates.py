"""Docker-only checks for experimental conservative GS candidates.

This does not install a production optimization. The five existing zone-test
groups retain their independent analytic/QP checks. Additional boundary cases
compare exact raw outputs against the production shader, not a new oracle.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT), str(ROOT / "src")]

import verify_tof_zones as checks
from devtools.mesh import MeshObject
from devtools.rendering import BatchRenderer, RasterBatch
from devtools.scene import GeometrySnapshot, SceneObject, Pose
from devtools.sensors import poses_to_tensor
from experiments.tof_pipeline_profile.variants import get_variants


CANDIDATES = ("single_cell_tight_gs", "range_aabb_gs", "range_aabb_single_cell_gs")


@contextmanager
def candidate_pipeline(variant):
    """Swap only the experimental zone program; pixel camera stays unchanged."""
    original = BatchRenderer._pipeline

    def pipeline(renderer, sampling):
        if sampling != "zone":
            return original(renderer, sampling)
        if renderer._zone_program is None:
            renderer._gl.activate()
            renderer._zone_program = renderer._ctx.program(
                vertex_shader=variant.vertex_shader,
                geometry_shader=variant.geometry_shader,
                fragment_shader=variant.fragment_shader)
            renderer._zone_uniforms = {
                name: renderer._gl.glGetUniformLocation(renderer._zone_program.glo, name.encode())
                for name in ("tangent_fov", "image_size")}
        return renderer._zone_program, renderer._zone_uniforms

    BatchRenderer._pipeline = pipeline
    try:
        yield
    finally:
        BatchRenderer._pipeline = original


def exact_raw(actual, expected):
    report = {}
    for field in fields(RasterBatch):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            passed = isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            passed = passed and a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
            if passed:
                a, b = a.detach().cpu(), b.detach().cpu()
                passed = bool(torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)) if a.is_floating_point() else bool(torch.equal(a, b))
            report[field.name] = bool(passed)
        else:
            report[field.name] = a == b
    assert all(report.values()), report
    return report


def triangle(vertices):
    """A closed thin prism with the requested triangle as its front face."""
    front = np.asarray(vertices, dtype=np.float64)
    points = np.vstack((front, front + np.asarray((.001, 0., 0.))))
    center = points.mean(axis=0)
    faces = ((0, 1, 2), (3, 4, 5),
             (0, 1, 4), (0, 4, 3),
             (1, 2, 5), (1, 5, 4),
             (2, 0, 3), (2, 3, 5))
    outward = []
    for face in faces:
        a, b, c = points[list(face)]
        normal = np.cross(b - a, c - a)
        if normal @ ((a + b + c) / 3 - center) < 0:
            face = (face[0], face[2], face[1])
        outward.append(face)
    mesh = MeshObject("probe", tuple(tuple(float(v) for v in p) for p in points),
                      tuple(outward), "obstacle")
    return GeometrySnapshot((SceneObject("probe", "obstacle", mesh),), 7.5, None)


def boundary_cases():
    config = checks.CONFIG
    origin = checks.ORIGIN
    # At x=4, the y=1 side is an exact 90-degree/8-column cell boundary.
    for direction, upper in (("below", np.nextafter(np.float32(1), np.float32(-np.inf))),
                             ("touch", np.float32(1)),
                             ("above", np.nextafter(np.float32(1), np.float32(np.inf)))):
        yield "cell_side_" + direction, triangle(((4, .8, 10.2), (4, float(upper), 10.2),
                                                   (4, .8, 10.24))), config, None
    for direction, lower in (("below", np.nextafter(np.float32(0), np.float32(-np.inf))),
                             ("touch", np.float32(0)),
                             ("above", np.nextafter(np.float32(0), np.float32(np.inf)))):
        yield "central_side_" + direction, triangle(((4, float(lower), 10.2), (4, .1, 10.2),
                                                      (4, float(lower), 10.24))), config, None
    yield "cell_corner_touch", triangle(((4, 0, 10), (4, .1, 10), (4, 0, 10.1))), config, None
    # Match the formal render-radius guard for this float32 input pose. These
    # are raw-render checks, so the guard sphere, rather than sensor max alone,
    # is the relevant rejection boundary.
    maximum = max(1., config.max_range_m, *(abs(v) for v in origin))
    guard = max(32 * np.finfo(np.float32).eps * maximum * 4, config.max_range_m * .001)
    render_radius = np.float32(config.max_range_m + guard)
    for label, center in (("sensor_max", np.float32(config.max_range_m)),
                          ("render_guard", render_radius)):
        for direction, radius in (("below", np.nextafter(center, np.float32(-np.inf))),
                                   ("equal", center),
                                   ("above", np.nextafter(center, np.float32(np.inf)))):
            x = float(radius)
            yield label + "_" + direction, triangle(((x, -.05, 9.95), (x, .05, 9.95),
                                                      (x, 0, 10.05))), config, None
    # Every vertex is farther than max range; the face crosses the range ball.
    # Vertex-distance-only culling would incorrectly remove its visible middle.
    yield "face_crosses_range_ball", triangle(((4, -20, -10), (4, 20, -10),
                                               (4, 0, 30))), config, "must_hit"
    odd = replace(config, width=7, height=9, hfov_deg=73, vfov_deg=61)
    source = checks.fixture(("front", ((4, 4.01), (-10, 10), (0, 20))),
                            ("small", ((2, 2.002), (.2, .202), (10.2, 10.202))))
    yield "non_power_of_two_7x9", source, odd, "must_hit"


def verify_boundary_parity(variant):
    results = []
    for name, snapshot, config, expectation in boundary_cases():
        with BatchRenderer(snapshot) as renderer:
            poses = poses_to_tensor((Pose(checks.ORIGIN),), device=renderer.device)
            baseline = renderer.render(poses, config, sampling="zone")
            original_program, original_uniforms = renderer._zone_program, renderer._zone_uniforms
            renderer._zone_program = None
            try:
                with candidate_pipeline(variant):
                    candidate = renderer.render(poses, config, sampling="zone")
                comparison = exact_raw(candidate, baseline)
                if expectation == "must_hit":
                    assert bool((baseline.object_index >= 0).any()), name
                results.append({"case": name, "status": "passed", "exact_fields": comparison,
                                "resolution": [config.width, config.height],
                                "baseline_hit_cells": int((baseline.object_index >= 0).sum()),
                                "oracle": "exact production raw-output parity; no independent boundary oracle"})
            finally:
                experimental_program = renderer._zone_program
                renderer._zone_program, renderer._zone_uniforms = original_program, original_uniforms
                if experimental_program is not None:
                    experimental_program.release()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CANDIDATES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    variant = get_variants()[args.variant]
    from devtools import _zone_shaders
    assert variant.fragment_shader == _zone_shaders.FRAGMENT_SHADER
    assert variant.vertex_shader == _zone_shaders.VERTEX_SHADER
    record = {"status": "running", "variant": variant.metadata(), "checks": [],
              "scope": "Experimental GS candidates only; production files and running review unchanged.",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "experiments/tof_pipeline_profile/variants.py",
                                          ROOT / "devtools/_zone_shaders.py", ROOT / "devtools/rendering.py",
                                          ROOT / "tests/verify_tof_zones.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(name, action, *, patched=True):
        print(name, flush=True)
        try:
            if patched:
                with candidate_pipeline(variant):
                    detail = action()
            else:
                detail = action()
            result = {"name": name, "status": "passed", "detail": detail}
        except Exception as error:
            result = {"name": name, "status": "failed", "error_type": type(error).__name__,
                      "error": str(error), "traceback": traceback.format_exc()}
        record["checks"].append(result)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(name + ": " + result["status"], flush=True)

    for name, action in (("existing_small_objects_and_boundaries", checks.verify_small_objects_and_boundaries),
                         ("existing_plane_and_occlusion", checks.verify_plane_and_occlusion),
                         ("existing_near_range_and_coverage", checks.verify_near_and_coverage),
                         ("existing_batch_ownership_and_updates", checks.verify_batch_ownership_and_updates),
                         ("existing_exact_review_screenshot", checks.verify_screenshot_case)):
        run(name, action)
    run("extra_boundary_raw_parity", lambda: verify_boundary_parity(variant), patched=False)
    record["status"] = "passed" if all(c["status"] == "passed" for c in record["checks"]) else "failed"
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if record["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
