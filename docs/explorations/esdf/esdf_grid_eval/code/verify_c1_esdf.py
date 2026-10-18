"""Archived C1 contract checks, frozen from d7073fb; run only in Docker.

These checks target ``c1_esdf`` here, not the formal trilinear ESDF component.
Only this notice and relocated import/provenance paths differ from the snapshot.

The random-node oracle builds tensor-product Hermite cubics from nodal values
and centred-difference jets. It does not import the implementation's weights or
cell selection. These checks establish an interpolant contract, not geometric
SDF accuracy, unit gradient, conservative clearance, or continuous collision.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np
import torch

from docs.explorations.esdf.esdf_grid_eval.code.c1_esdf import ESDFGrid, build_esdf


BITS = tuple(itertools.product((0, 1), repeat=3))


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def rejected(action):
    try:
        action()
    except (ValueError, TypeError, RuntimeError, OSError, KeyError):
        return
    raise AssertionError("invalid input was accepted")


def close(actual, expected, *, atol=2e-11, rtol=2e-11):
    np.testing.assert_allclose(array(actual), array(expected), atol=atol, rtol=rtol)


def nodes(shape, origin, spacing):
    indices = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1)
    return np.asarray(origin) + indices * spacing


def fixture(device, *, random=True):
    shape, origin, spacing = (7, 6, 5), np.array([-.5, .25, -1.]), .25
    points = nodes(shape, origin, spacing)
    samples = (np.random.default_rng(1823).normal(size=shape) if random else points[..., 2] - .125).astype(np.float32)
    bounds = np.stack((origin + spacing, origin + (np.asarray(shape)-2)*spacing))
    field = ESDFGrid(samples, sample_origin_m=origin, spacing_m=spacing, bounds_m=bounds, device=device)
    return field, samples, origin, spacing, bounds


def value_gradient(field, points, device, dtype=torch.float64):
    p = torch.tensor(points, device=device, dtype=dtype, requires_grad=True)
    distance, valid = field.query(p, return_valid=True)
    gradient, = torch.autograd.grad(distance.sum(), p)
    return distance.detach().cpu().numpy(), gradient.detach().cpu().numpy(), valid.detach().cpu().numpy()


class HermiteOracle:
    """Endpoint values/slopes, including mixed central-difference nodal jets."""

    def __init__(self, samples, origin, spacing):
        self.samples = np.asarray(samples, dtype=np.float64)
        self.origin, self.spacing = np.asarray(origin), spacing
        self.jets = {}

    def jet(self, index, derivatives):
        key = (*index, *derivatives)
        if key not in self.jets:
            choices = [((-1, -.5), (1, .5)) if derivative else ((0, 1.),)
                       for derivative in derivatives]
            self.jets[key] = sum(np.prod([item[1] for item in combination]) *
                                 self.samples[tuple(i + item[0] for i, item in zip(index, combination))]
                                 for combination in itertools.product(*choices))
        return self.jets[key]

    def evaluate(self, point, cell=None):
        coordinate = (np.asarray(point) - self.origin) / self.spacing
        if cell is None:
            cell = np.clip(np.floor(coordinate).astype(int), 1, np.asarray(self.samples.shape)-3)
        cell = np.asarray(cell, dtype=int)
        u = coordinate - cell
        # Indexed as [axis, endpoint, value-or-slope].
        basis = np.stack((np.stack((2*u**3-3*u**2+1, u**3-2*u**2+u), axis=-1),
                          np.stack((-2*u**3+3*u**2, u**3-u**2), axis=-1)), axis=1)
        derivative = np.stack((np.stack((6*u**2-6*u, 3*u**2-4*u+1), axis=-1),
                               np.stack((-6*u**2+6*u, 3*u**2-2*u), axis=-1)), axis=1)
        value, gradient = 0., np.zeros(3)
        for corner in BITS:
            index = tuple(cell + corner)
            for orders in BITS:
                coefficient = self.jet(index, orders)
                weights = [basis[axis, corner[axis], orders[axis]] for axis in range(3)]
                value += coefficient * np.prod(weights)
                for axis in range(3):
                    factors = weights.copy()
                    factors[axis] = derivative[axis, corner[axis], orders[axis]] / self.spacing
                    gradient[axis] += coefficient * np.prod(factors)
        return value, gradient

    def batch(self, points):
        pairs = [self.evaluate(point) for point in points]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def analytic_reproduction(device):
    shape, origin, spacing = (8, 7, 6), np.array([-.75, -.5, -.25]), .25
    grid_points = nodes(shape, origin, spacing)
    bounds = np.stack((origin + spacing, origin + (np.asarray(shape)-2)*spacing))
    points = np.random.default_rng(721).uniform(bounds[0], bounds[1], (127, 3))
    points = np.concatenate((points, np.array(list(itertools.product(*zip(*bounds))))))

    def quadratic(p):
        x, y, z = np.moveaxis(p, -1, 0)
        value = .25*x*x - .5*y*y + .125*z*z + .25*x*y - .125*x*z + .375*y*z + .5*x - .25*y + .125*z + .75
        gradient = np.stack((.5*x+.25*y-.125*z+.5, -y+.25*x+.375*z-.25,
                             .25*z-.125*x+.375*y+.125), axis=-1)
        return value, gradient

    measurements = {}
    for name, function in (("plane", lambda p: (p[..., 2]-.125, np.broadcast_to([0., 0., 1.], p.shape))),
                           ("quadratic_with_cross_terms", quadratic)):
        values, _ = function(grid_points)
        samples = values.astype(np.float32)
        assert np.array_equal(samples.astype(np.float64), values), "analytic fixture nodes must be exactly representable"
        field = ESDFGrid(samples, sample_origin_m=origin, spacing_m=spacing, bounds_m=bounds, device=device)
        actual, gradient, valid = value_gradient(field, points, device)
        expected, expected_gradient = function(points)
        assert valid.all()
        close(actual, expected)
        close(gradient, expected_gradient)
        measurements[name] = {"point_count": len(points), "value_max_abs": float(abs(actual-expected).max()),
                              "gradient_max_abs": float(abs(gradient-expected_gradient).max())}
    return measurements


def all_internal_faces(device):
    field, samples, origin, spacing, _ = fixture(device)
    oracle = HermiteOracle(samples, origin, spacing)
    points, left_cells, right_cells, axes = [], [], [], []
    face_counts = []
    for axis in range(3):
        count = 0
        ranges = [range(2, size-2) if a == axis else range(1, size-2)
                  for a, size in enumerate(samples.shape)]
        other = [a for a in range(3) if a != axis]
        for anchor in itertools.product(*ranges):
            count += 1
            for fractions in itertools.product((.17, .5, .83), repeat=2):
                coordinate = np.array(anchor, dtype=float)
                coordinate[other] += fractions
                left, right = np.array(anchor), np.array(anchor)
                left[axis] -= 1
                points.append(origin + coordinate*spacing)
                left_cells.append(left)
                right_cells.append(right)
                axes.append(axis)
        face_counts.append(count)
    points, axes = np.array(points), np.array(axes)
    left = [oracle.evaluate(p, cell) for p, cell in zip(points, left_cells)]
    right = [oracle.evaluate(p, cell) for p, cell in zip(points, right_cells)]
    vl, vr = np.array([item[0] for item in left]), np.array([item[0] for item in right])
    gl, gr = np.array([item[1] for item in left]), np.array([item[1] for item in right])
    close(vl, vr, atol=3e-13, rtol=0)
    close(gl, gr, atol=3e-12, rtol=0)
    actual, gradient, valid = value_gradient(field, points, device)
    assert valid.all()
    close(actual, vl)
    close(gradient, gl, atol=3e-10, rtol=2e-11)
    offset = np.zeros_like(points)
    offset[np.arange(len(points)), axes] = spacing*1e-6
    limits = []
    oracle_errors = []
    for sign in (-1, 1):
        v1, g1, _ = value_gradient(field, points+sign*offset, device)
        v2, g2, _ = value_gradient(field, points+2*sign*offset, device)
        expected, expected_gradient = oracle.batch(points+sign*offset)
        close(v1, expected)
        close(g1, expected_gradient, atol=3e-10, rtol=2e-11)
        limits.append((2*v1-v2, 2*g1-g2))
        oracle_errors.append(float(abs(g1-expected_gradient).max()))
    value_jump = float(abs(limits[0][0]-limits[1][0]).max())
    gradient_jump = float(abs(limits[0][1]-limits[1][1]).max())
    assert value_jump < 1e-10 and gradient_jump < 1e-8
    return {"fixture_shape": list(samples.shape), "faces_by_axis": face_counts,
            "all_internal_faces": sum(face_counts), "face_points": len(points),
            "independent_same_point_value_jump": float(abs(vl-vr).max()),
            "independent_same_point_world_gradient_jump": float(abs(gl-gr).max()),
            "query_face_value_error": float(abs(actual-vl).max()),
            "query_face_world_gradient_error": float(abs(gradient-gl).max()),
            "query_two_sided_linear_limit_value_jump": value_jump,
            "query_two_sided_linear_limit_gradient_jump": gradient_jump,
            "query_one_sided_oracle_gradient_errors": oracle_errors,
            "scope": "every internal face of a fixed random fixture, 3x3 interior points; numerical evidence plus shared Hermite jets, not geometric-SDF C1"}


def float32_and_finite_differences(device):
    field, samples, origin, spacing, bounds = fixture(device)
    oracle = HermiteOracle(samples, origin, spacing)
    rng = np.random.default_rng(992)
    cells = np.column_stack([rng.integers(1, size-2, 47) for size in samples.shape])
    points = origin + (cells+rng.uniform(.2, .8, (47, 3)))*spacing
    actual32, gradient32, valid = value_gradient(field, points, device, torch.float32)
    represented = points.astype(np.float32).astype(np.float64)
    expected32, expected_gradient32 = oracle.batch(represented)
    assert valid.all() and actual32.dtype == np.float32 and gradient32.dtype == np.float32
    close(actual32, expected32, atol=2e-6, rtol=5e-6)
    close(gradient32, expected_gradient32, atol=6e-5, rtol=5e-6)
    _, gradient64, _ = value_gradient(field, points, device)
    fd_errors = []
    for step, tolerance in ((1e-5, 5e-7), (1e-6, 2e-8)):
        finite_difference = np.empty_like(points)
        for axis in range(3):
            delta = np.zeros_like(points)
            delta[:, axis] = step
            plus = field.query(torch.tensor(points+delta, dtype=torch.float64, device=device)).detach().cpu().numpy()
            minus = field.query(torch.tensor(points-delta, dtype=torch.float64, device=device)).detach().cpu().numpy()
            finite_difference[:, axis] = (plus-minus)/(2*step)
        close(gradient64, finite_difference, atol=tolerance, rtol=2e-7)
        fd_errors.append({"step_m": step, "gradient_max_abs": float(abs(gradient64-finite_difference).max())})
    return {"float32_value_error": float(abs(actual32-expected32).max()),
            "float32_world_gradient_error": float(abs(gradient32-expected_gradient32).max()),
            "float64_central_differences": fd_errors, "points_away_from_seams": len(points)}


def batch_shape_and_ownership(device):
    field, samples, origin, spacing, bounds = fixture(device)
    p = torch.tensor(np.random.default_rng(19).uniform(bounds[0], bounds[1], (2, 3, 4, 3)),
                     device=device, dtype=torch.float64).transpose(1, 2).requires_grad_()
    assert not p.is_contiguous()
    distance, valid = field.query(p, return_valid=True)
    assert distance.shape == p.shape[:-1] and valid.shape == distance.shape and valid.dtype == torch.bool
    assert distance.dtype == p.dtype and distance.device == p.device and valid.device == p.device and bool(valid.all())
    close(distance, field.query(p.reshape(-1, 3)).reshape(distance.shape))
    gradient, = torch.autograd.grad(distance.sum(), p)
    assert gradient.shape == p.shape and bool(torch.isfinite(gradient).all())
    scalar = field.query(p.detach()[0, 0, 0])
    assert scalar.shape == ()
    for shape in ((0, 3), (2, 0, 3), (0, 2, 3)):
        empty = torch.empty(shape, device=device, dtype=torch.float64, requires_grad=True)
        values, mask = field.query(empty, return_valid=True)
        assert values.shape == mask.shape == shape[:-1]
        empty_gradient, = torch.autograd.grad(values.sum(), empty)
        assert empty_gradient.shape == empty.shape
    supplied = torch.tensor(samples, device=device, requires_grad=True)
    detached = ESDFGrid(supplied, sample_origin_m=origin, spacing_m=spacing, bounds_m=bounds, device=device)
    point = p.detach()[0, 0, 0].clone().requires_grad_()
    before = detached.query(point)
    gp, gs = torch.autograd.grad(before, (point, supplied), allow_unused=True)
    assert gp is not None and gs is None
    with torch.no_grad():
        supplied.add_(100.)
        detached.samples.add_(200.)
    close(detached.query(point), before)
    samples.fill(500.)
    assert not np.all(array(field.samples) == 500.)
    return {"noncontiguous_batch_shape": list(p.shape), "empty_shapes": [[0, 3], [2, 0, 3], [0, 2, 3]],
            "sample_input_detached_and_cloned": True, "samples_property_isolated": True,
            "map_sample_gradient": None, "query_gradient_preserved": True}


def closed_bounds_and_invalid_queries(device):
    # Decimal bounds force the explicit float32 represented-boundary policy.
    origin, spacing, shape = np.array([-.7, -.6, -.5]), .2, (8, 8, 8)
    coordinates = nodes(shape, origin, spacing)
    samples = (coordinates[..., 0] + 2*coordinates[..., 1] - .5*coordinates[..., 2]).astype(np.float32)
    bounds = np.array([[-.37, -.28, -.19], [.43, .51, .59]])
    field = ESDFGrid(samples, sample_origin_m=origin, spacing_m=spacing, bounds_m=bounds, device=device)
    oracle = HermiteOracle(samples, origin, spacing)
    centre = bounds.mean(axis=0)
    results = {}
    for dtype in (torch.float64, torch.float32):
        corners = np.array(list(itertools.product(*zip(*bounds))))
        points = torch.tensor(corners, device=device, dtype=dtype, requires_grad=True)
        values, valid = field.query(points, return_valid=True)
        gradients, = torch.autograd.grad(values.sum(), points)
        assert bool(valid.all())
        expected, expected_gradient = oracle.batch(corners)
        close(values, expected, atol=2e-6 if dtype == torch.float32 else 2e-11, rtol=2e-6)
        close(gradients, expected_gradient, atol=1e-5 if dtype == torch.float32 else 2e-10, rtol=2e-6)
        bad = []
        for axis in range(3):
            for side, direction in ((0, -float("inf")), (1, float("inf"))):
                point = torch.tensor(centre, dtype=dtype, device=device)
                edge = torch.tensor(bounds[side, axis], dtype=dtype, device=device)
                point[axis] = torch.nextafter(edge, torch.tensor(direction, dtype=dtype, device=device))
                bad.append(point)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            point = torch.tensor(centre, dtype=dtype, device=device)
            point[1] = invalid
            bad.append(point)
        points = torch.stack([torch.tensor(centre, dtype=dtype, device=device), *bad]).requires_grad_()
        values, mask = field.query(points, return_valid=True)
        assert bool(mask[0]) and not bool(mask[1:].any()) and bool(torch.isnan(values[1:]).all())
        # Backpropagate through NaN outputs themselves: invalid positional VJPs
        # must already be zero, not depend on a downstream mask hiding them.
        gradients, = torch.autograd.grad(values.sum(), points)
        assert bool(torch.isfinite(gradients).all()) and bool((gradients[1:] == 0).all())
        results[str(dtype)] = {"legal_corners": 8, "rejected_nextafter_or_nonfinite": len(bad),
                               "invalid_gradient_exactly_zero": True}
    # Uppermost interior node must use the left cell and still have a derivative.
    edge_field, _, _, _, edge_bounds = fixture(device, random=False)
    value, gradient, valid = value_gradient(edge_field, edge_bounds, device)
    assert valid.all()
    close(value, edge_bounds[:, 2]-.125)
    close(gradient, np.broadcast_to([0., 0., 1.], gradient.shape))
    return results


def construction_and_query_rejections(device):
    field, samples, origin, spacing, bounds = fixture(device)
    count = 0

    def reject(action):
        nonlocal count
        rejected(action)
        count += 1

    def construct(value=samples, **changes):
        arguments = {"sample_origin_m": origin, "spacing_m": spacing, "bounds_m": bounds, "device": device}
        arguments.update(changes)
        return ESDFGrid(value, **arguments)

    for bad in (samples.astype(np.float64), samples.astype(np.int32), samples[0], samples[:, :, :3]):
        reject(lambda bad=bad: construct(bad))
    for invalid in (np.nan, np.inf, -np.inf):
        bad = samples.copy()
        bad.flat[0] = invalid
        reject(lambda bad=bad: construct(bad))
    for bad in (True, False, 0., -.1, np.nan, np.inf, [.2, .2, .2]):
        reject(lambda bad=bad: construct(spacing_m=bad))
    for bad in ([0, 0], [0, np.nan, 0], [0, np.inf, 0]):
        reject(lambda bad=bad: construct(sample_origin_m=bad))
    for bad in (bounds[::-1], np.zeros((2, 3)), [[0, 0], [1, 1]],
                np.stack((origin, bounds[1])), np.stack((bounds[0], origin+np.asarray(samples.shape)*spacing))):
        reject(lambda bad=bad: construct(bounds_m=bad))
    for bad in ([], {"invalid": np.nan}, {"invalid": {1, 2}}):
        reject(lambda bad=bad: construct(metadata=bad))
    for bad in (torch.zeros((3, 2), device=device), torch.zeros((), device=device),
                torch.zeros((2, 3), device=device, dtype=torch.int64),
                torch.zeros((2, 3), device=device, dtype=torch.float16), [[0., 0., 0.]]):
        reject(lambda bad=bad: field.query(bad))
    if device.type == "cuda":
        reject(lambda: field.query(torch.zeros((1, 3), dtype=torch.float64, device="cpu")))
    return {"rejected_cases": count, "wrong_device_case": device.type == "cuda"}


def builder_resolution_and_padding(device):
    bounds = np.array([[-.35, -.28, .07], [.43, .36, .79]])
    measurements = []
    for spacing in (.2, .1, .05):
        chunks = []

        def oracle(points):
            assert isinstance(points, np.ndarray) and points.dtype == np.float64
            assert points.ndim == 2 and points.shape[1] == 3 and 0 < len(points) <= 37
            chunks.append(points.copy())
            return points[:, 0] - .5*points[:, 1] + .25*points[:, 2]

        field = build_esdf(oracle, bounds_m=bounds, voxel_size_m=spacing, chunk_size=37,
                           metadata={"fixture": "builder-plane", "spacing": spacing})
        shape = np.ceil((bounds[1]-bounds[0])/spacing).astype(int)+3
        assert tuple(field.shape) == tuple(shape)
        assert field.samples.dtype == torch.float32 and field.samples.device.type == "cpu"
        close(field.sample_origin_m, bounds[0]-spacing)
        close(field.bounds_m, bounds)
        coordinates = nodes(shape, bounds[0]-spacing, spacing)
        observed = np.concatenate(chunks)
        assert len(observed) == int(np.prod(shape))
        expected_keys = set(map(tuple, coordinates.reshape(-1, 3)))
        assert len(set(map(tuple, observed))) == len(expected_keys)
        close(np.array(sorted(map(tuple, observed))), np.array(sorted(expected_keys)), atol=1e-14, rtol=0)
        expected = (coordinates[..., 0]-.5*coordinates[..., 1]+.25*coordinates[..., 2]).astype(np.float32)
        assert np.array_equal(array(field.samples), expected)
        moved = field.to(device)
        p = np.vstack((bounds, bounds.mean(axis=0)))
        value, gradient, valid = value_gradient(moved, p, device)
        assert valid.all()
        close(value, p[:, 0]-.5*p[:, 1]+.25*p[:, 2], atol=2e-7, rtol=0)
        close(gradient, np.broadcast_to([1., -.5, .25], gradient.shape), atol=6e-6, rtol=0)
        padded_hi = bounds[0]+(shape-3)*spacing
        for axis in np.flatnonzero(padded_hi > bounds[1]+1e-12):
            outside = bounds.mean(axis=0)
            outside[axis] = (padded_hi[axis]+bounds[1, axis])/2
            distance, valid = moved.query(torch.tensor(outside, device=device, dtype=torch.float64), return_valid=True)
            assert not bool(valid) and bool(torch.isnan(distance))
        measurements.append({"spacing_m": spacing, "shape": shape.tolist(), "callback_count": len(chunks),
                             "sample_count": len(observed), "ceil_padding_does_not_expand_bounds": True})
    for bad in (0, -1, True):
        rejected(lambda bad=bad: build_esdf(lambda p: p[:, 0], bounds_m=bounds, chunk_size=bad))
    rejected(lambda: build_esdf(lambda p: np.zeros((len(p), 1)), bounds_m=bounds))
    rejected(lambda: build_esdf(lambda p: np.full(len(p), np.nan), bounds_m=bounds))
    return {"resolutions": measurements, "builder_invalid_cases": 5}


def storage_roundtrip_and_corruption(device):
    _, samples, origin, spacing, bounds = fixture(device)
    metadata = {"fixture": "random", "units": "metre", "nested": {"items": [1, True, None, "测试"]}}
    field = ESDFGrid(samples, sample_origin_m=origin, spacing_m=spacing, bounds_m=bounds,
                     metadata=metadata, device=device)
    metadata["nested"]["items"][0] = 999
    assert field.metadata["nested"]["items"][0] == 1
    returned = field.metadata
    returned["nested"]["items"][0] = 998
    assert field.metadata["nested"]["items"][0] == 1
    points = np.random.default_rng(58).uniform(bounds[0], bounds[1], (23, 3))
    expected, expected_gradient, _ = value_gradient(field, points, device)
    corruptions = 0
    with tempfile.TemporaryDirectory(prefix="verify-esdf-") as temporary:
        root = Path(temporary)
        saved = root / "saved"
        field.save(saved)
        manifest = json.loads((saved / "metadata.json").read_text())
        stored = np.load(saved / "samples.npy", allow_pickle=False)
        assert stored.dtype == np.float32 and np.array_equal(stored, samples)
        assert manifest["samples_sha256"] == hashlib.sha256((saved / "samples.npy").read_bytes()).hexdigest()
        loaded = ESDFGrid.load(saved, device=device)
        assert loaded.metadata == field.metadata and tuple(loaded.shape) == tuple(samples.shape)
        close(loaded.sample_origin_m, origin)
        close(loaded.spacing_m, spacing)
        close(loaded.bounds_m, bounds)
        actual, gradient, _ = value_gradient(loaded, points, device)
        close(actual, expected, atol=0, rtol=0)
        close(gradient, expected_gradient, atol=0, rtol=0)
        for key, invalid in (("format", "other"), ("version", 999), ("axis_order", "zyx"),
                             ("dtype", "float64"), ("halo", 0), ("interpolation", "trilinear"),
                             ("shape", [7, 6, 6]), ("spacing_m", -spacing),
                             ("bounds_m", [[0, 0, 0], [100, 100, 100]]),
                             ("samples_sha256", "0"*64), ("metadata", [])):
            bad_dir = root / f"bad-{key}"
            shutil.copytree(saved, bad_dir)
            altered = dict(manifest)
            altered[key] = invalid
            (bad_dir / "metadata.json").write_text(json.dumps(altered))
            rejected(lambda bad_dir=bad_dir: ESDFGrid.load(bad_dir, device=device))
            corruptions += 1
        # Even a matching hash cannot relabel a float64 sample file as float32.
        bad_dir = root / "bad-array-dtype"
        shutil.copytree(saved, bad_dir)
        np.save(bad_dir / "samples.npy", samples.astype(np.float64), allow_pickle=False)
        altered = dict(manifest)
        altered["samples_sha256"] = hashlib.sha256((bad_dir / "samples.npy").read_bytes()).hexdigest()
        (bad_dir / "metadata.json").write_text(json.dumps(altered))
        rejected(lambda: ESDFGrid.load(bad_dir, device=device))
        corruptions += 1
    return {"sample_dtype": "float32", "metadata_deep_copy": True,
            "value_and_gradient_bitwise_roundtrip": True, "rejected_corruptions": corruptions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("run these checks inside Docker")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    verifiers = (analytic_reproduction, all_internal_faces, float32_and_finite_differences,
                 batch_shape_and_ownership, closed_bounds_and_invalid_queries,
                 construction_and_query_rejections, builder_resolution_and_padding,
                 storage_roundtrip_and_corruption)
    groups = {}
    started = time.perf_counter()
    for verify in verifiers:
        try:
            groups[verify.__name__] = {"status": "passed", **verify(device)}
        except Exception:
            groups[verify.__name__] = {"status": "failed", "traceback": traceback.format_exc()}
    passed = all(group["status"] == "passed" for group in groups.values())
    result = {"status": "passed" if passed else "failed", "device": str(device),
              "torch_version": torch.__version__, "elapsed_correctness_seconds": time.perf_counter()-started,
              "scope": "fixed-node interpolation C1, position AD and serialization contract; not geometric gradient, clearance or performance validation",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), Path("docs/explorations/esdf/esdf_grid_eval/code/c1_esdf.py"))}, "groups": groups}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"esdf ({device}): {result['status']}; {len(groups)} groups; result: {args.results}")
    if not passed:
        for name, group in groups.items():
            if group["status"] == "failed":
                print(name, group["traceback"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
