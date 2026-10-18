"""Docker-only throughput and memory evidence for unified, budgeted rays.

Run cells sequentially. Pose generation retains warp_ray_perf's seed/prefix
formula without importing its frozen, older sensor API. No Graph or fused
postprocessing kernel is used. All timed inputs/outputs stay on CUDA.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from devtools.config import SamplingRuntimeConfig, load_config
from devtools.scene import Pose, Scene
from devtools.ray_rendering import RayRenderer
from devtools.sensors import SensorSampler, ToFSensor
from devtools.rig import poses_to_tensor, sample_rig
from devtools.perception import sample_measurements

SEED = 20261016
FIRST_POSITION = (.00400000019, -17.60400009, 1.5)
FIRST_YAW_DEG = 78.6839981


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quaternion(roll, pitch, yaw):
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def make_poses(batch):
    random = np.random.default_rng(SEED)
    first = Pose(FIRST_POSITION, quaternion(0., 0., math.radians(FIRST_YAW_DEG)))
    result = [first]
    for _ in range(batch - 1):
        delta = random.uniform(-1., 1., 6)
        position = np.asarray(FIRST_POSITION) + delta[:3] * (.25, .25, .15)
        rpy = np.radians(delta[3:] * (8., 6., 12.) + (0., 0., FIRST_YAW_DEG))
        result.append(Pose(tuple(float(x) for x in position), quaternion(*rpy)))
    return result


def tensor_fields(value, prefix=''):
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if is_dataclass(value):
        result = {}
        for field in fields(value):
            result.update(tensor_fields(getattr(value, field.name), f'{prefix}.{field.name}'.strip('.')))
        return result
    return {}


def memory():
    free, total = torch.cuda.mem_get_info()
    return {'torch_allocated_bytes': torch.cuda.memory_allocated(),
            'torch_reserved_bytes': torch.cuda.memory_reserved(),
            'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'shared_device_free_bytes': free, 'shared_device_total_bytes': total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--mode', choices=('tof', 'rig', 'measurements'), required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--memory-fraction', type=float, default=.5)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--event-repeats', type=int, default=3)
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 1 or args.warmup < 0 or args.event_repeats < 1:
        parser.error('invalid batch/repeats')
    destination = Path(args.output)
    if destination.exists():
        parser.error('output must be new; preserve prior results')
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    renderer = None
    record = {'status': 'running', 'mode': args.mode, 'batch': args.batch,
              'batch_unit': 'sensor_views' if args.mode == 'tof' else 'body_poses',
              'scope': 'warm formal API; CUDA Pose input to CUDA final observations, including budget planning/chunk assembly/rays/query/postprocessing; excludes scene setup, CPU Pose upload, GUI and host export',
              'memory_scope': 'Torch allocator peak includes fixed inputs/scene plus live final outputs and temporaries; excludes native Warp BVH/driver/other processes. Global free is a shared-device snapshot, not whole-run peak.',
              'same_scene': 'one renderer and static BVH reused for all chunks and calls',
              'optimizations': 'unified rays, shared origins, outer perception, conservative automatic chunks; no RGB, Graph or fused postprocess kernel'}
    try:
        source_paths = ['devtools/ray_rendering.py', 'devtools/_ray_budget.py', 'devtools/sensors.py',
                        'devtools/_sensor_ops.py', 'devtools/rig.py',
                        'devtools/_ray_processing.py', 'devtools/perception.py', 'devtools/config.py',
                        'devtools/scene.py', 'devtools/mesh.py', 'configs/devtools.json',
                        'experiments/unified_ray_perf/benchmark.py']
        record['source_sha256'] = {name: digest(ROOT / name) for name in source_paths}
        snapshot_folder = destination.parent / (destination.stem + '-source')
        for name in source_paths:
            target = snapshot_folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        record['source_snapshot'] = str(snapshot_folder)
        record['map'] = {'path': args.map, 'sha256': digest(args.map)}
        config = load_config(ROOT / 'configs/devtools.json')
        runtime = SamplingRuntimeConfig(args.memory_fraction)
        record['config'] = asdict(config)
        record['sampling_runtime'] = asdict(runtime)
        record['environment'] = {'torch': torch.__version__, 'warp': wp.__version__,
                                 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                                 'python': sys.version, 'cpu_threads': torch.get_num_threads()}
        poses_cpu = make_poses(args.batch)
        pose_array = np.stack([p.as_transform().matrix for p in poses_cpu]).astype(np.float32)
        record['input'] = {'seed': SEED, 'pose_sha256': hashlib.sha256(pose_array.tobytes()).hexdigest(),
                           'policy': 'warp_ray_perf prefix-stable full RPY distribution',
                           'target_position_m': list(poses_cpu[0].body_to_world((3., 0., 0.)))}
        scene = Scene.from_obj(args.map, config, poses_cpu[0], record['input']['target_position_m'], t_ref=12.5)
        snapshot = scene.get_geometry(exclude_ids={'vehicle'})
        poses = poses_to_tensor(poses_cpu)
        torch.cuda.synchronize()
        started = time.perf_counter()
        renderer = RayRenderer(snapshot)
        sampler = SensorSampler(renderer, runtime=runtime)
        torch.cuda.synchronize()
        record['scene_setup_ms'] = (time.perf_counter() - started) * 1000
        record['geometry_before'] = renderer.geometry_info
        if args.mode == 'tof':
            sensor = ToFSensor(sampler, config.sensors.tof,
                               quantile=config.sensors.tof_depth_quantile,
                               signal_floor=config.sensors.tof_signal_floor)
            action = lambda: sensor.sample(poses)
            rays_per_item = 1024
        elif args.mode == 'rig':
            action = lambda: sample_rig(sampler, config.sensors, poses)
            rays_per_item = 4 * 1024 + 160 * 120 * 4
        else:
            action = lambda: sample_measurements(sampler, config.sensors, poses, snapshot,
                                                 config.perception)
            rays_per_item = 4 * 1024 + 160 * 120 * 4
        def check_contract(before):
            after = renderer.geometry_info
            if after['rays_traced'] - before['rays_traced'] != args.batch * rays_per_item:
                raise AssertionError('ray count differs from requested full-scene samples')
            if before['geometry_uploads'] != after['geometry_uploads'] or after['geometry_uploads'] != 1:
                raise AssertionError('scene reuploaded')
            return after['trace_calls'] - before['trace_calls']

        before = renderer.geometry_info
        started = time.perf_counter()
        output = action()
        torch.cuda.synchronize()
        record['first_forward_ms'] = (time.perf_counter() - started) * 1000
        check_contract(before)
        del output
        for _ in range(args.warmup):
            before = renderer.geometry_info
            output = action()
            torch.cuda.synchronize()
            check_contract(before)
            del output
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        record['memory_before'] = memory()
        wall, calls, executions = [], [], []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            before = renderer.geometry_info
            started = time.perf_counter()
            output = action()
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - started) * 1000)
            # Copying the O(scene object count) diagnostic dictionary is a
            # verification boundary, not part of the sampling API timer.
            calls.append(check_contract(before))
            executions.append(output.execution_info)
            del output
        event_spans = []
        for _ in range(args.event_repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            before = renderer.geometry_info
            start.record()
            output = action()
            end.record()
            end.synchronize()
            event_spans.append(start.elapsed_time(end))
            check_contract(before)
            del output
        record['memory_after'] = memory()
        # Verification-only reductions/downloads occur after the timed samples.
        before = renderer.geometry_info
        output = action()
        torch.cuda.synchronize()
        check_contract(before)
        outputs = tensor_fields(output)
        if not outputs or any(not t.is_cuda for t in outputs.values()):
            raise AssertionError('final observation is not entirely CUDA')
        record['outputs'] = {name: {'shape': list(t.shape), 'dtype': str(t.dtype),
                                   'bytes': t.numel() * t.element_size()}
                             for name, t in outputs.items()}
        record['execution_info'] = output.execution_info
        record['wall_samples_ms'] = wall
        record['wall_median_ms'] = statistics.median(wall)
        record['wall_p95_ms'] = float(np.quantile(wall, .95))
        record['items_per_second'] = args.batch * 1000 / record['wall_median_ms']
        record['trace_calls_per_action'] = calls
        record['execution_info_per_action'] = executions
        record['cuda_event_span_ms'] = event_spans
        record['event_scope'] = 'complete stream interval including host submission gaps; not pure kernel time'
        record['geometry_after'] = renderer.geometry_info
        record['status'] = 'ok'
        del output
    except Exception as error:
        record.update(status='error', error=str(error), traceback=traceback.format_exc())
    finally:
        if renderer is not None:
            renderer.close()
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'mode': args.mode, 'batch': args.batch,
                      'output': args.output, 'median_ms': record.get('wall_median_ms'),
                      'error': record.get('error')}), flush=True)
    return 0 if record['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
