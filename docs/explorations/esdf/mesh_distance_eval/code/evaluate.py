"""Reproducible correctness + synchronized wall-clock route evaluation."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import warp as wp

from devtools.config import load_config
from devtools.scene import Pose, Scene
from docs.explorations.esdf.mesh_distance_eval.code.backend import MeshDistance, SampledField
from docs.explorations.esdf.mesh_distance_eval.code.fixtures import make_box, write_obj, snapshot_oracle


CASES = {
    'analytic': dict(size=(8., 6., 4.), shape=(40, 30, 20), count=None),
    'pillar_medium': dict(size=(32., 32., 8.8), shape=(160, 160, 44), count=500),
    'pillar_omni': dict(size=(60., 60., 15.), shape=(300, 300, 75), count=500),
}


def sync():
    torch.cuda.synchronize()


def timed_once(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, (time.perf_counter() - start) * 1000


def timing(fn, repeats=15, warmup=3):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        _, duration = timed_once(fn)
        samples.append(duration)
    return dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                min_ms=min(samples), samples_ms=samples, warmup=warmup)


def gpu_status():
    result = subprocess.run(['nvidia-smi'], text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else result.stderr


def prepare_scene(case, directory, binary):
    spec = CASES[case]
    path = directory / (case + '.obj')
    if spec['count'] is None:
        write_obj(path, [make_box('box', (-1., -.8, .4), (1., .8, 2.4))], size=spec['size'])
    else:
        subprocess.run([str(binary), '--output', str(path), '--seed', '23',
                        '--size-x', str(spec['size'][0]), '--size-y', str(spec['size'][1]),
                        '--size-z', str(spec['size'][2]), '--count', str(spec['count']),
                        '--width-min', '.3', '--width-max', '1.0'], check=True)
    return path


def load_scene(path):
    return Scene.from_obj(path, load_config('configs/devtools.json'),
                          vehicle_pose=Pose((-2, 0, 1)), target_position_m=(2, .6, 1.4), t_ref=0.)


def field_args(case):
    spec = CASES[case]
    return ((-spec['size'][0]/2 + .1, -spec['size'][1]/2 + .1, .1), spec['shape'], .2)


def error_stats(delta):
    delta = np.asarray(delta)
    return dict(max_abs=float(np.max(np.abs(delta))), mean_abs=float(np.mean(np.abs(delta))),
                p95_abs=float(np.percentile(np.abs(delta), 95)),
                max_overestimate=float(np.max(delta)), min_difference=float(np.min(delta)))


def query_backward(query, points):
    p = points.detach().requires_grad_(True)
    d = query(p)
    loss = torch.exp(-d).mean()
    return torch.autograd.grad(loss, p)[0]


def evaluate_case(case, path, output, repeats):
    directory = output / case
    directory.mkdir()
    spec = CASES[case]
    scene, load_ms = timed_once(lambda: load_scene(path))
    snapshot, snapshot_ms = timed_once(lambda: scene.get_geometry(exclude_ids={'vehicle'}))
    free_before = torch.cuda.mem_get_info()[0]
    torch_before = torch.cuda.memory_allocated()
    backend, prep_ms = timed_once(lambda: MeshDistance(snapshot))
    # The first case also includes JIT in these first calls; subsequent cases reuse it.
    origin, shape, spacing = field_args(case)
    lo = np.asarray(origin)
    hi = lo + (np.asarray(shape) - 1) * spacing
    rng = np.random.default_rng(20261015)
    host_points = rng.uniform(lo, hi, (65536, 3)).astype(np.float32)
    points, upload_ms = timed_once(lambda: torch.tensor(host_points, device='cuda'))
    _, first_torch_ms = timed_once(lambda: backend.query(points, method='torch'))
    _, first_tape_ms = timed_once(lambda: query_backward(lambda p: backend.query(p, method='tape'), points))
    # Native sampling uses no_grad, retaining no Tape or position graph.
    field, first_field_ms = timed_once(lambda: SampledField(backend, origin, shape, spacing))
    torch.cuda.reset_peak_memory_stats()
    memory_before_bench = torch.cuda.memory_allocated()
    profiles = {}
    for batch in (1024, 65536):
        p = points[:batch]
        profiles[str(batch)] = {}
        for route, query in (
            ('A_torch', lambda q: backend.query(q, method='torch')),
            ('A_tape', lambda q: backend.query(q, method='tape')),
            ('B_trilinear', field.query),
        ):
            with torch.no_grad():
                distance = timing(lambda: query(p), repeats)
            backward = timing(lambda: query_backward(query, p), repeats)
            profiles[str(batch)][route] = dict(distance=distance, distance_cost_backward=backward)
    # Match the declared sampled function, then independently check continuous faces.
    with torch.no_grad():
        direct = backend.query(points, method='tape').cpu().numpy()
        stable_direct = backend.query(points, method='torch').cpu().numpy()
        interpolated = field.query(points).cpu().numpy()
    expected = snapshot_oracle(host_points[:256], snapshot)
    oracle_delta = direct[:256] - expected
    checks = {'raw_surface_oracle': error_stats(oracle_delta),
              'A_torch_raw_surface_oracle': error_stats(stable_direct[:256] - expected),
              'A_torch_minus_A_tape_distance': error_stats(stable_direct - direct),
              'B_minus_A': error_stats(interpolated - direct),
              'B_minus_A_torch': error_stats(interpolated - stable_direct),
              'oracle_points': len(expected),
              'all_finite': bool(np.isfinite(direct).all() and np.isfinite(interpolated).all())}
    gradients = {}
    for name, query in (('A_torch', lambda p: backend.query(p, method='torch')),
                        ('A_tape', lambda p: backend.query(p, method='tape')),
                        ('B', field.query)):
        p = points.detach().requires_grad_(True)
        gradients[name] = torch.autograd.grad(query(p).sum(), p)[0].detach().cpu().numpy()
    checks['all_gradients_finite'] = all(np.isfinite(g).all() for g in gradients.values())
    checks['A_torch_minus_tape_gradient'] = error_stats(gradients['A_torch'] - gradients['A_tape'])
    separated = np.abs(direct) > .05
    checks['gradient_separated_distance_m'] = .05
    checks['gradient_near_surface_points'] = int((~separated).sum())
    checks['A_torch_minus_tape_gradient_separated'] = error_stats(
        gradients['A_torch'][separated] - gradients['A_tape'][separated])
    checks['B_minus_A_gradient'] = error_stats(gradients['B'] - gradients['A_tape'])
    np.savez_compressed(directory / 'query-evidence.npz', points=host_points,
                        A=direct, A_torch=stable_direct, B=interpolated, oracle=expected,
                        **{f'gradient_{name}': g for name, g in gradients.items()})
    (directory / 'correctness.json').write_text(json.dumps(checks, indent=2) + '\n')
    if (checks['raw_surface_oracle']['max_abs'] > 1e-4
            or checks['A_torch_raw_surface_oracle']['max_abs'] > 1e-4 or not checks['all_finite']
            or not checks['all_gradients_finite']
            or checks['A_torch_minus_tape_gradient_separated']['max_abs'] > 2e-3):
        raise AssertionError(f'{case}: independent face-distance oracle failed: {checks}')
    np.save(directory / 'field-initial.npy', field.values.cpu().numpy())
    # Isolate optional disk/CPU field paths from GPU sampling.
    cpu_values, download_ms = timed_once(lambda: field.values.cpu().contiguous())
    uploaded, field_upload_ms = timed_once(lambda: cpu_values.to('cuda'))
    del uploaded
    sampling = timing(lambda: field.rebuild(backend), repeats)
    snapshots = [scene.with_target_position(p).get_geometry(exclude_ids={'vehicle'})
                 for p in ((-2, 1, 1.5), (2, .6, 1.4))]
    counter = [0]

    def update():
        counter[0] += 1
        backend.update(snapshots[counter[0] % 2])

    snapshot_update = timing(lambda: scene.with_target_position((-2, 1, 1.5)).get_geometry(
        exclude_ids={'vehicle'}), repeats)
    a_update = timing(update, repeats)

    def rebuild():
        update()
        field.rebuild(backend)

    b_update = timing(rebuild, repeats)
    memory = dict(torch_baseline_bytes=torch_before,
                  torch_current_increment_bytes=torch.cuda.memory_allocated() - torch_before,
                  torch_benchmark_peak_above_resident_bytes=torch.cuda.max_memory_allocated() - memory_before_bench,
                  global_free_delta_bytes=free_before - torch.cuda.mem_get_info()[0],
                  field_bytes=field.values.numel() * field.values.element_size(),
                  field_volume_copy_bytes=field.volume.numel() * field.volume.element_size(),
                  vertex_bytes=backend.vertices.numel() * backend.vertices.element_size(),
                  index_bytes=backend.indices.numel() * backend.indices.element_size(),
                  note='Torch excludes Warp BVH allocator; global free delta includes caches and desktop activity. See isolated memory probes.')
    result = dict(case=case, size_xyz_m=spec['size'], shape_xyz=shape, origin_xyz=origin,
                  spacing_m=spacing, triangles=len(backend.indices), points=65536,
                  min_triangle_quality=backend.min_triangle_quality,
                  geometry_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  first=dict(scene_load_cpu_ms=load_ms, public_selection_cpu_ms=snapshot_ms,
                             static_pack_upload_bvh_ms=prep_ms, points_upload_ms=upload_ms,
                             torch_distance_ms=first_torch_ms, tape_distance_cost_backward_ms=first_tape_ms,
                             GPU_sampling_field_ms=first_field_ms, field_download_ms=download_ms,
                             optional_field_upload_ms=field_upload_ms),
                  profiles=profiles, field_sampling=sampling, snapshot_update=snapshot_update,
                  A_target_update=a_update, B_target_update_fresh_sampling=b_update,
                  correctness=checks, memory=memory)
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def memory_probe(args):
    """Each call is a new process: context baseline, resident and Torch peak."""
    torch.cuda.init()
    wp.init()
    sync()
    gc.collect()
    torch.cuda.empty_cache()
    free0 = torch.cuda.mem_get_info()[0]
    torch0 = torch.cuda.memory_allocated()
    scene = load_scene(args.map)
    b = MeshDistance(scene.get_geometry(exclude_ids={'vehicle'}))
    route = args.route
    field = SampledField(b, *field_args(args.memory_case)) if route == 'B' else None
    sync()
    resident_free = torch.cuda.mem_get_info()[0]
    resident_torch = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    origin, shape, spacing = field_args(args.memory_case)
    p = torch.rand((65536, 3), device='cuda')
    p = p * (torch.tensor(shape, device='cuda') - 1) * spacing + torch.tensor(origin, device='cuda')
    query = field.query if field else lambda q: b.query(q, method='tape' if route == 'A_tape' else 'torch')
    for _ in range(3):
        query_backward(query, p)
    sync()
    result = dict(route=route, case=args.memory_case,
                  context_baseline_free_bytes=free0,
                  post_prepare_global_used_delta_bytes=free0 - resident_free,
                  post_backward_global_used_delta_bytes=free0 - torch.cuda.mem_get_info()[0],
                  torch_resident_delta_bytes=resident_torch - torch0,
                  torch_peak_delta_bytes=torch.cuda.max_memory_allocated() - torch0,
                  limitation='Fresh process but global CUDA free deltas include other GPU activity; Warp peak not instrumented.',
                  gpu_status=gpu_status())
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('outputs/mesh_distance_eval/build/offline_mapgen'))
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--memory-case', choices=CASES)
    parser.add_argument('--route', choices=['A_torch', 'A_tape', 'B'])
    parser.add_argument('--map', type=Path)
    args = parser.parse_args()
    if args.memory_case:
        memory_probe(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.cuda.init()
    wp.init()
    start_status = gpu_status()
    from docs.explorations.esdf.mesh_distance_eval.code.verify import run_verification
    from docs.explorations.esdf.mesh_distance_eval.code.review_regressions import run as run_review
    # Keep first-call / JIT evidence separate, ahead of full verification.
    results = []
    maps = {}
    for case in args.cases:
        maps[case] = prepare_scene(case, args.output, args.binary)
        print(f'evaluate {case}', flush=True)
        results.append(evaluate_case(case, maps[case], args.output, args.repeats))
    verification = run_verification(args.output / 'verification')
    review = run_review(args.output / 'review-regressions')
    probes = []
    for case in args.cases:
        for route in ('A_torch', 'A_tape', 'B'):
            path = args.output / case / f'memory-{route}.json'
            subprocess.run([sys.executable, '-m', 'docs.explorations.esdf.mesh_distance_eval.code.evaluate',
                            '--memory-case', case, '--map', str(maps[case]), '--route', route,
                            '--output', str(path)], check=True)
            probes.append(json.loads(path.read_text()))
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       torch=torch.__version__, torch_cuda=torch.version.cuda,
                       warp=wp.__version__, numpy=np.__version__,
                       trimesh=importlib.metadata.version('trimesh'),
                       GPU=torch.cuda.get_device_name(), torch_cpu_threads=torch.get_num_threads(),
                       gpu_status_start=start_status, gpu_status_end=gpu_status(),
                       nvcc=(subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout
                             if shutil.which('nvcc') else None),
                       compiler=subprocess.run(['g++', '--version'], capture_output=True, text=True).stdout,
                       pip_freeze=subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                                 capture_output=True, text=True).stdout,
                       git_base_revision=os.environ.get('MESH_EVAL_SOURCE_REVISION'),
                       image_id=os.environ.get('MESH_EVAL_IMAGE_ID'),
                       host_gpu_status_before=os.environ.get('MESH_EVAL_HOST_GPU_STATUS'))
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path('docs/explorations/esdf/mesh_distance_eval/code').glob('*.py')}
    report = dict(environment=environment, timing='synchronized wall-clock; 3 warmup; raw samples, median and p95; no CUDA graphs',
                  results=results, verification=verification, review_regressions=review,
                  memory_probes=probes, source_sha256=source_hashes)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({r['case']: r['correctness'] for r in results}, indent=2), flush=True)
    if not verification['passed'] or not review['passed']:
        raise SystemExit('Verification failed; all reports and raw evidence were retained.')


if __name__ == '__main__':
    main()
