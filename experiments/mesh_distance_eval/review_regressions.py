"""Independent review regressions; execute inside the experiment image.

Checks are separate from verify.py. The near-surface Tape diagnostics are saved
without claiming that a finite native gradient equals the analytic derivative.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .backend import MeshDistance, triangle_distance
from .fixtures import make_box, make_scene


def run(output, device='cuda'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks, diagnostics, arrays = {}, {}, {}

    def evaluate(query, values):
        p = torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)
        d = query(p)
        g, = torch.autograd.grad(torch.nan_to_num(d).sum(), p)
        return d.detach().cpu().numpy(), g.detach().cpu().numpy()

    # This valid triangle used to have a finite distance and all-NaN gradient
    # because its float32 Gram determinant rounded to zero.
    p = np.array([[.5, .000025, .1]], dtype=np.float32)
    tri = torch.tensor([[[0., 0., 0.], [1., .0001, 0.], [1., 0., 0.]]], device=device)
    d, g = evaluate(lambda q: triangle_distance(q, tri), p)
    checks['sliver_recompute_finite_analytic_gradient'] = bool(
        np.isfinite(g).all() and np.allclose(d, .1, atol=1e-7, rtol=0)
        and np.allclose(g, [[0, 0, 1]], atol=1e-7, rtol=0))
    arrays.update(sliver_points=p, sliver_distance=d, sliver_gradient=g)

    # A closed thin box also defeated native nearest-face selection. A stable
    # selected-face formula cannot repair a wrong selection, so this class must
    # now be rejected explicitly at backend preparation.
    rejection = {}
    for width in (.001, .0001):
        snapshot = make_scene((make_box('thin', (-1, -width/2, 1), (1, width/2, 3)),))\
            .get_geometry(exclude_ids={'vehicle'})
        try:
            MeshDistance(snapshot, device=device)
            rejection[str(width)] = None
        except ValueError as exc:
            rejection[str(width)] = str(exc)
    checks['native_thin_face_failure_class_rejected'] = all(
        text is not None and 'quality < 1e-3' in text for text in rejection.values())
    diagnostics['native_thin_face_rejections'] = rejection

    # The explicit domain admits this point, but the native search radius does
    # not. Misses must not become an invented zero-distance hit in A_torch.
    snapshot = make_scene((make_box('box', (-1, -1, 1), (1, 1, 3)),))\
        .get_geometry(exclude_ids={'vehicle'})
    backend = MeshDistance(snapshot, device=device, bounds=((-2e10, 2e10), (-5, 5), (0, 6)))
    p = np.array([[1.5e10, 0, 2]], dtype=np.float32)
    for method in ('torch', 'tape'):
        d, g = evaluate(lambda q: backend.mesh_query(q, method=method), p)
        checks[f'search_miss_{method}_nan_zero_gradient'] = bool(
            np.isnan(d).all() and np.array_equal(g, np.zeros_like(g)))
        arrays.update({f'search_miss_{method}_distance': d, f'search_miss_{method}_gradient': g})

    # Reproduce loss of small displacement precision at ordinary map world
    # offsets. Axis-aligned box face gives an independent, exact gradient (+X).
    # Four scales separate near-surface behavior from the smooth far regime.
    obstacle = make_box('offset_box', (-27., 17., 1.), (-26., 18., 14.))
    snapshot = make_scene((obstacle,), size=(60., 60., 15.))\
        .get_geometry(exclude_ids={'vehicle'})
    backend = MeshDistance(snapshot, device=device)
    rng = np.random.default_rng(2026101507)
    points = []
    for separation in (2e-5, 1e-4, 1e-3, .1):
        yz = rng.uniform((17.1, 1.5), (17.9, 13.5), size=(128, 2))
        points.append(np.column_stack((np.full(len(yz), -26. + separation), yz)))
    p = np.concatenate(points).astype(np.float32)
    expected = p[:, 0].astype(np.float64) + 26.
    expected_g = np.tile([1., 0., 0.], (len(p), 1))
    arrays.update(offset_points=p, offset_expected_distance=expected,
                  offset_expected_gradient=expected_g)
    for method in ('torch', 'tape'):
        d, g = evaluate(lambda q: backend.mesh_query(q, method=method), p)
        arrays.update({f'offset_{method}_distance': d, f'offset_{method}_gradient': g})
        checks[f'offset_{method}_outputs_finite'] = bool(np.isfinite(d).all() and np.isfinite(g).all())
        if method == 'torch':
            checks['offset_torch_exact_face_distance_and_gradient'] = bool(
                np.allclose(d, expected, atol=2e-7, rtol=0)
                and np.allclose(g, expected_g, atol=2e-6, rtol=0))
        rows = []
        for i, separation in enumerate((2e-5, 1e-4, 1e-3, .1)):
            sl = slice(i*128, (i+1)*128)
            rows.append({'requested_separation_m': separation,
                         'max_distance_error_m': float(np.max(np.abs(d[sl]-expected[sl]))),
                         'max_gradient_component_error': float(np.max(np.abs(g[sl]-expected_g[sl])))})
        diagnostics[f'offset_{method}'] = rows
    diagnostics['offset_tape_claim'] = (
        'Values and gradients are retained as numeric diagnostics. Finite native output '
        'does not assert accurate near-surface derivatives; no near-surface pass is claimed.')

    report = {'passed': all(checks.values()), 'device': device, 'checks': checks,
              'diagnostics': diagnostics, 'raw_arrays': 'review-regression-arrays.npz',
              'backend_sha256': hashlib.sha256(Path(__file__).with_name('backend.py').read_bytes()).hexdigest()}
    np.savez_compressed(output / 'review-regression-arrays.npz', **arrays)
    (output / 'review-regressions.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    report = run(args.output, args.device)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
