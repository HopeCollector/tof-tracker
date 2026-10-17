"""Profile fixed-raw GPU sensor postprocessing inside Docker, one cell per run.

Use the same pillars-20 map as warp_ray_perf. Recommended cells: ToF B1/64/10000
and camera B1/64. No query is executed in any measured postprocessing block.
The formal function is timed/profiled without inserting synchronization inside
it. A separately labelled, exact-output decomposition provides annotated traces
and optional per-block synchronization diagnostics; these are not additive
costs of the formal function. --cuda-graph is a diagnostic replay of the formal
function with fixed inputs and shared output storage, not a production change.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from devtools.config import DepthConfig, _finite_number, load_config
from devtools.ray_rendering import RayRenderer
from devtools.scene import Scene
from devtools.sensors import (CameraBatch, DepthBatch, SampleBatch, _check_poses,
                              generate_rays, poses_to_tensor, process_samples)
from devtools._ray_processing import (_bounds, _check_samples,
                                      camera_measurements, gather_subray,
                                      range_by_sample, select_subray)
from experiments.warp_ray_perf.benchmark import (allocator, command, describe,
                                                 make_poses, sha256, verify_equal)


class Markers:
    """Annotation-only by default; explicit-sync timings are a separate run."""

    def __init__(self, samples=None):
        self.samples = samples

    @contextmanager
    def block(self, name):
        with record_function(name):
            if self.samples is None:
                yield
                return
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            started = time.perf_counter()
            yield
            end.record()
            end.synchronize()
            self.samples.setdefault(name, []).append({
                'wall_ms': (time.perf_counter() - started) * 1000,
                'cuda_event_span_ms': start.elapsed_time(end)})


def split_select(distance, identity, config, reduction, quantile, signal_floor, mark, prefix):
    """Same expressions and evaluation order as select_subray, with labels."""
    with mark.block(prefix + '.select.validation_effective'):
        _check_samples(distance, identity)
        maximum = _finite_number(config.max_range_m, 'max_distance')
        minimum = _finite_number(config.min_range_m, 'min_distance')
        if not 0 <= minimum < maximum:
            raise ValueError('ranges must satisfy 0 <= min_distance < max_distance')
        effective = torch.where(identity >= 0, distance, maximum)
    if reduction == 'nearest':
        with mark.block(prefix + '.select.argmin'):
            return effective.argmin(dim=-1)
    with mark.block(prefix + '.select.weighted_validation_clamp'):
        if reduction != 'weighted_quantile':
            raise ValueError('unsupported reduction')
        quantile = _finite_number(quantile, 'quantile')
        floor = _finite_number(signal_floor, 'signal_floor')
        if not 0 <= quantile <= 1 or floor <= 0:
            raise ValueError('invalid weighted quantile configuration')
        effective = effective.clamp(min=minimum, max=maximum)
    with mark.block(prefix + '.select.stable_sort'):
        ordered, order = torch.sort(effective, dim=-1, stable=True)
    with mark.block(prefix + '.select.weights'):
        inverse_square = (1 / (ordered.square() + 0.05)).clamp_min(floor)
        signal = torch.where(ordered >= maximum - 1e-4, floor, inverse_square)
        signal = signal * (1 - 0.35 * (ordered / maximum).square()).clamp_min(0.05)
    with mark.block(prefix + '.select.cdf'):
        cumulative = signal.cumsum(dim=-1)
    with mark.block(prefix + '.select.quantile_index'):
        selected = (cumulative < quantile * cumulative[..., -1:]).sum(dim=-1)
        selected = selected.clamp_max(distance.shape[-1] - 1)
    with mark.block(prefix + '.select.gather_selection'):
        return order.gather(-1, selected[..., None]).squeeze(-1)


def split_ranges(distance, identity, origins, directions, config, bounds, mark, prefix):
    """Same range_by_sample(return_known=True), including current availability."""
    with mark.block(prefix + '.range.validation'):
        _check_samples(distance, identity)
        if not isinstance(config, DepthConfig):
            raise TypeError('config must be DepthConfig')
        for values in (origins, directions):
            if (not isinstance(values, torch.Tensor) or values.device != distance.device
                    or values.dtype != torch.float32 or values.shape != (*distance.shape, 3)):
                raise TypeError('invalid ray records')
    with mark.block(prefix + '.range.hit_valid_ray'):
        hit = identity >= 0
        valid_ray = (torch.isfinite(origins).all(dim=-1) & torch.isfinite(directions).all(dim=-1)
                     & (directions.abs().amax(dim=-1) > 0))
    if bounds is None:
        with mark.block(prefix + '.range.missing_bounds'):
            known = valid_ray
            full = torch.zeros_like(hit)
    else:
        with mark.block(prefix + '.range.bounds_origin'):
            limits = _bounds(bounds, distance.device)
            origin_inside = (valid_ray & ((origins >= limits[:, 0])
                                         & (origins <= limits[:, 1])).all(dim=-1))
        with mark.block(prefix + '.range.max_endpoint'):
            end = origins + config.max_range_m * directions
            full = origin_inside & ((end >= limits[:, 0]) & (end <= limits[:, 1])).all(dim=-1)
        with mark.block(prefix + '.range.hit_point_tolerance'):
            point = origins + distance[..., None] * directions
            tolerance = (32 * torch.finfo(torch.float32).eps
                         * origins.abs().amax(dim=-1).clamp_min(max(1.0, config.max_range_m)))[..., None]
        with mark.block(prefix + '.range.hit_box'):
            known = origin_inside & ((point >= limits[:, 0] - tolerance)
                                     & (point <= limits[:, 1] + tolerance)).all(dim=-1)
    with mark.block(prefix + '.range.classification'):
        valid = hit & known & (distance >= config.min_range_m) & (distance <= config.max_range_m)
        ranges = torch.where(valid, distance, math.nan)
        ranges = torch.where(~hit & full, math.inf, ranges)
    with mark.block(prefix + '.range.available'):
        available = torch.where(hit, known & torch.isfinite(distance) & (distance >= 0), full)
    return ranges, available


def split_process(hits, origins, directions, poses, config, mark, prefix,
                  reduction='nearest', quantile=.2, signal_floor=.08):
    """Literal process_samples computation with separately named blocks."""
    with mark.block(prefix + '.validate_reshape'):
        _check_poses(poses)
        if not isinstance(config, DepthConfig):
            raise TypeError('config must be DepthConfig')
        shape = (poses.shape[0], config.height, config.width, config.samples_per_axis ** 2)
        rays = math.prod(shape)
        for values in (origins, directions):
            if (not isinstance(values, torch.Tensor) or values.device != poses.device
                    or values.dtype != torch.float32 or values.shape != (rays, 3)):
                raise TypeError('invalid origins/directions')
        if hits.distance_m.shape != (rays,) or hits.distance_m.device != poses.device:
            raise ValueError('hits must match generated rays')
        distance = hits.distance_m.reshape(shape)
        identity = hits.object_index.reshape(shape)
        points = hits.hit_point_world_m.reshape(*shape, 3)
        triangles = hits.triangle_index.reshape(shape)
        origins = origins.reshape(*shape, 3)
    with mark.block(prefix + '.normalize_directions'):
        scale = directions.abs().amax(dim=-1, keepdim=True)
        scaled = directions / scale.clamp_min(torch.finfo(torch.float32).tiny)
        directions = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        directions = directions.reshape(*shape, 3)
    selected = split_select(distance, identity, config, reduction, quantile,
                            signal_floor, mark, prefix)
    ranges, known = split_ranges(distance, identity, origins, directions, config,
                                 hits.map_bounds_m, mark, prefix)
    with mark.block(prefix + '.gather_representative_records'):
        selected_distance = gather_subray(distance, selected)
        selected_identity = gather_subray(identity, selected)
        selected_triangles = gather_subray(triangles, selected)
        selected_world = gather_subray(points, selected)
    with mark.block(prefix + '.world_to_sensor_point'):
        relative = selected_world - poses[:, None, None, :3, 3]
        sensor_points = torch.matmul(relative.reshape(len(poses), -1, 3),
                                     poses[:, :3, :3]).reshape(*shape[:3], 3)
        sensor_points = torch.where((selected_identity >= 0)[..., None], sensor_points, 0.0)
    with mark.block(prefix + '.raw_dataclass'):
        raw = SampleBatch(selected_distance, selected_identity, hits.object_ids,
                          hits.scene_version, hits.t_ref, hits.map_bounds_m,
                          sensor_points, selected_triangles, distance, identity, points,
                          triangles, origins, directions, selected)
    with mark.block(prefix + '.unknown_cell_representative'):
        representative = gather_subray(ranges, selected)
        representative = torch.where(known.all(dim=-1), representative, math.nan)
    with mark.block(prefix + '.depth_dataclass'):
        result = DepthBatch(representative, selected_identity, hits.object_ids, hits.scene_version,
                            hits.t_ref, sensor_points, selected_triangles, raw, ranges)
    return result


def split_camera_measurements(full_ranges, full_identity, target_ranges, target_identity,
                              config, target_index, threshold, mark):
    prefix = 'camera.measurements'
    with mark.block(prefix + '.validation'):
        _check_samples(full_ranges, full_identity)
        _check_samples(target_ranges, target_identity)
        if full_ranges.shape != target_ranges.shape or full_ranges.device != target_ranges.device:
            raise ValueError('full and target shapes/devices differ')
        threshold = _finite_number(threshold, 'threshold')
        if not 0 <= threshold <= 1 or isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0:
            raise ValueError('invalid target index/threshold')
    with mark.block(prefix + '.visible_masks'):
        visible = (full_identity == target_index) & torch.isfinite(full_ranges)
        unoccluded = (target_identity == target_index) & torch.isfinite(target_ranges)
    with mark.block(prefix + '.coverage_counts'):
        coverage = visible.to(torch.float32).mean(dim=-1)
        mask = visible.any(dim=-1)
        count = unoccluded.sum(dim=(-3, -2, -1), dtype=torch.int64)
        visible_count = visible.sum(dim=(-3, -2, -1), dtype=torch.int64)
    with mark.block(prefix + '.fraction_detection'):
        fraction = visible_count.to(torch.float64) / count.clamp_min(1).to(torch.float64)
        fraction = torch.where(count > 0, fraction, 0.0)
        detected = (count > 0) & (visible_count > 0) & (fraction >= threshold)
    with mark.block(prefix + '.hint_box'):
        rows = torch.arange(config.height, device=full_ranges.device)
        cols = torch.arange(config.width, device=full_ranges.device)
        row_occupied, col_occupied = mask.any(dim=-1), mask.any(dim=-2)
        first_row = torch.where(row_occupied, rows, config.height).amin(dim=-1)
        last_row = torch.where(row_occupied, rows, -1).amax(dim=-1)
        first_col = torch.where(col_occupied, cols, config.width).amin(dim=-1)
        last_col = torch.where(col_occupied, cols, -1).amax(dim=-1)
        box = torch.stack((first_col / config.width, first_row / config.height,
                           (last_col + 1) / config.width, (last_row + 1) / config.height), dim=-1)
        hint = torch.cat((detected[:, None].to(torch.float32), box), dim=-1)
        hint = torch.where(detected[:, None], hint, 0.0)
    with mark.block(prefix + '.unoccluded_pixels_result'):
        result = {'target_mask': mask, 'target_coverage': coverage, 'target_hint': hint,
                  'visible_fraction': fraction,
                  'unoccluded_pixels': unoccluded.any(dim=-1).sum(dim=(-2, -1), dtype=torch.int64),
                  'unoccluded_samples': count}
    return result


def summarize(samples):
    return {'median_ms': statistics.median(samples), 'p95_ms': float(np.quantile(samples, .95)),
            'min_ms': min(samples), 'samples_ms': samples}


def measure(action, warmup, repeats, event_repeats):
    for _ in range(warmup):
        output = action()
        torch.cuda.synchronize()
        del output
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = allocator()
    wall = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = action()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - started) * 1000)
        del output
    events = []
    for _ in range(event_repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output = action()
        end.record()
        end.synchronize()
        events.append(start.elapsed_time(end))
        del output
    memory_after = allocator()
    return {'wall': summarize(wall), 'cuda_event_span': summarize(events) if events else None,
            'event_scope': 'complete stream span including CPU submission gaps; not kernel sum',
            'memory_before': memory_before, 'memory_after': memory_after,
            'peak_increment_bytes': memory_after['peak_allocated_bytes'] - memory_before['allocated_bytes']}


def device_time(event, self_time=False):
    field = 'self_device_time_total' if self_time else 'device_time_total'
    return float(getattr(event, field, 0.0))


def profiler_probe(action, destination, steps):
    """No intra-function synchronization; read raw exported kernel durations."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True, with_stack=False) as prof:
        for _ in range(steps):
            with record_function('profiled_postprocess'):
                output = action()
            del output
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(destination))
    exported = json.loads(destination.read_text())
    events = exported.get('traceEvents', [])
    kernels = [item for item in events if item.get('cat') == 'kernel' and item.get('ph') == 'X']
    copies = [item for item in events if 'memcpy' in item.get('cat', '').lower()]
    averages = prof.key_averages()

    def row(event):
        return {'name': event.key, 'count': event.count,
                'self_cpu_us': float(event.self_cpu_time_total), 'cpu_total_us': float(event.cpu_time_total),
                'self_device_us': device_time(event, True), 'device_total_us': device_time(event),
                'self_cpu_memory_bytes': int(event.self_cpu_memory_usage),
                'self_device_memory_bytes': int(getattr(event, 'self_device_memory_usage', 0))}

    cpu = sorted(averages, key=lambda item: item.self_cpu_time_total, reverse=True)
    cuda = sorted(averages, key=lambda item: device_time(item, True), reverse=True)
    annotations = [row(item) for item in averages if item.key.startswith(('tof.', 'camera.'))]
    durations = sum(float(item.get('dur', 0)) for item in kernels)
    device_events = []
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            name = event.name
            kind = ('memcpy' if 'memcpy' in name.lower() else
                    'memset' if 'memset' in name.lower() else 'kernel')
            device_events.append({'name': name, 'kind': kind,
                                  'duration_us': float(event.time_range.elapsed_us())})
    device_kernels = [event for event in device_events if event['kind'] == 'kernel']
    driver_allocation = {event.key: event.count for event in averages
                         if event.key.startswith(('cudaMalloc', 'cudaFree', 'cuMemAlloc', 'cuMemFree'))}
    tensor_allocation = {event.key: event.count for event in averages
                         if event.key.startswith(('aten::empty', 'aten::new_empty'))}
    positive_memory_events = [event for event in events
                              if event.get('name') == '[memory]'
                              and event.get('args', {}).get('Bytes', 0) > 0]
    return {'steps': steps, 'trace': str(destination),
            'scope': 'profiler diagnostic, no intra-function barriers; profiler overhead excluded from warm wall measurements',
            'cuda_kernel_count_total': len(kernels), 'cuda_kernel_count_per_call': len(kernels) / steps,
            'raw_cuda_kernel_duration_sum_us_total': durations,
            'raw_cuda_kernel_duration_sum_us_per_call': durations / steps,
            'raw_duration_scope': 'sum of Chrome trace cat=kernel durations; excludes launch gaps/runtime/memcpy and may overlap on multiple streams',
            'device_type_cuda_kernel_count': len(device_kernels),
            'device_type_cuda_kernel_duration_sum_us': sum(event['duration_us'] for event in device_kernels),
            'raw_device_events': device_events,
            'allocation_observations': {
                'driver_runtime_allocation_call_counts': driver_allocation,
                'tensor_empty_operator_counts': tensor_allocation,
                'positive_profiler_memory_event_count': len(positive_memory_events),
                'scope': 'aten::empty and positive memory events may reuse the Torch cache; they do not imply cudaMalloc or a fresh driver allocation'},
            'memcpy_event_count': len(copies), 'cpu_top_ops': [row(item) for item in cpu[:40]],
            'cuda_top_ops': [row(item) for item in cuda[:40]],
            'annotations': annotations,
            'all_key_averages': [row(item) for item in averages]}


def graph_probe(action, reference, changed_input, args):
    """Capture only the formal postprocess, not rays/query or split functions."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = action()
            del output
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = action()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    parity = verify_equal(captured, reference)

    def replay():
        graph.replay()
        return captured

    result = measure(replay, args.warmup, args.repeats, args.event_repeats)
    # A view of raw distance aliases the supplied input; use a cloned final
    # range as the baseline to prove that replay consumes changed input data.
    depth = reference.depth if isinstance(reference, CameraBatch) else reference
    original_range = depth.range_m.clone()
    with changed_input():
        changed_reference = action()
        graph.replay()
        torch.cuda.synchronize()
        changed_parity = verify_equal(captured, changed_reference)
        changed_depth = (changed_reference.depth if isinstance(changed_reference, CameraBatch)
                         else changed_reference)
        observable_change = not bool(torch.allclose(changed_depth.range_m, original_range,
                                                   rtol=0., atol=0., equal_nan=True))
        if not observable_change:
            raise AssertionError('changed-input graph probe did not alter a final range')
        del changed_reference
    graph.replay()
    torch.cuda.synchronize()
    restored_parity = verify_equal(captured, reference)
    result.update(status='ok', parity=parity,
                  changed_input_parity=changed_parity, changed_input_observable=observable_change,
                  restored_input_parity=restored_parity,
                  scope='diagnostic only: CUDA Graph replay of unmodified formal postprocess with fixed inputs and shared output buffers; not production owned-output API',
                  conclusion_limit='Replay jointly removes/reduces Python, dispatcher, cached allocator and driver submission work and changes allocation/lifetimes. A faster replay cannot be assigned solely to CUDA kernel launch cost or one exclusive CPU category.')
    return result


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.set_grad_enabled(False)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    record = {'status': 'running', 'mode': args.mode, 'batch': args.batch,
              'scope': 'fixed raw inputs only; no scene query or ray generation in timed/profiled postprocessing',
              'gpu_shared': 'desktop/review may share the device; no clocks or host settings changed',
              'nonadditive': 'unsynchronized formal profile, annotated profile, synchronized sub-block probes and graph replay are separate experiments; do not add their medians'}

    def progress(stage):
        print(json.dumps({'stage': stage, 'mode': args.mode, 'batch': args.batch}), file=sys.stderr, flush=True)

    try:
        files = ('devtools/sensors.py', 'devtools/_ray_processing.py', 'devtools/ray_rendering.py',
                 'devtools/config.py', 'devtools/scene.py', 'devtools/mesh.py',
                 'experiments/warp_ray_perf/benchmark.py', 'experiments/warp_postprocess_profile/analyze.py')
        record['source'] = {'head': command(['git', 'rev-parse', 'HEAD']),
                            'sha256': {name: sha256(ROOT / name) for name in files},
                            'function_sha256': {function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                                                for function in (process_samples, select_subray, range_by_sample, camera_measurements)}}
        snapshot_root = destination.parent / (destination.stem + '-source')
        for name in files:
            target = snapshot_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        record['source']['snapshot'] = str(snapshot_root)
        config = load_config(args.config)
        sensor = config.sensors
        if sensor is None:
            raise ValueError('configuration has no sensors')
        record['config'] = {'path': args.config, 'sha256': sha256(args.config), 'values': asdict(config)}
        record['map'] = {'path': args.map, 'sha256': sha256(args.map)}
        record['environment'] = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                                 'python': sys.version, 'cpu_threads': args.cpu_threads,
                                 'gpu': torch.cuda.get_device_name(),
                                 'nvidia_smi': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])}
        progress('prepare_fixed_raw')
        poses_cpu = make_poses(args.batch)
        first = make_poses(1)[0]
        target_position = first.body_to_world((3., 0., 0.))
        scene = Scene.from_obj(args.map, config, first, target_position, t_ref=12.5)
        snapshot = scene.get_geometry(exclude_ids={'vehicle'})
        renderer = RayRenderer(snapshot)
        poses = poses_to_tensor(poses_cpu)
        pose_array = np.stack([pose.as_transform().matrix for pose in poses_cpu]).astype(np.float32)
        record['input'] = {'pose_sha256': hashlib.sha256(pose_array.tobytes()).hexdigest(),
                           'target_position_m': list(target_position),
                           'pose_policy': 'same make_poses default seed/prefix/full RPY as warp_ray_perf; target 3m forward'}
        depth_config = sensor.tof if args.mode == 'tof' else sensor.camera
        origins, directions = generate_rays(poses, depth_config)
        full_hits = renderer.trace(origins, directions, depth_config.max_range_m)
        target_hits = None if args.mode == 'tof' else renderer.trace(
            origins, directions, depth_config.max_range_m, only_object='target')
        torch.cuda.synchronize()
        record['fixed_raw'] = describe((poses, origins, directions, full_hits, target_hits))
        record['geometry'] = renderer.geometry_info

        def formal():
            if args.mode == 'tof':
                return process_samples(full_hits, origins, directions, poses, depth_config,
                                       reduction='weighted_quantile', quantile=sensor.tof_depth_quantile,
                                       signal_floor=sensor.tof_signal_floor)
            depth = process_samples(full_hits, origins, directions, poses, depth_config)
            target = process_samples(target_hits, origins, directions, poses, depth_config)
            measurements = camera_measurements(depth.subray_range_m, depth.raw.subray_object_index,
                                               target.subray_range_m, target.raw.subray_object_index,
                                               depth_config, full_hits.object_ids.index('target'),
                                               sensor.min_visible_fraction)
            return CameraBatch(depth, **measurements)

        def decomposed(mark):
            if args.mode == 'tof':
                return split_process(full_hits, origins, directions, poses, depth_config, mark, 'tof',
                                     reduction='weighted_quantile', quantile=sensor.tof_depth_quantile,
                                     signal_floor=sensor.tof_signal_floor)
            with mark.block('camera.full.process_samples'):
                depth = split_process(full_hits, origins, directions, poses, depth_config, mark, 'camera.full')
            with mark.block('camera.target.process_samples'):
                target = split_process(target_hits, origins, directions, poses, depth_config, mark, 'camera.target')
            measurements = split_camera_measurements(
                depth.subray_range_m, depth.raw.subray_object_index,
                target.subray_range_m, target.raw.subray_object_index,
                depth_config, full_hits.object_ids.index('target'), sensor.min_visible_fraction, mark)
            with mark.block('camera.result_dataclass'):
                return CameraBatch(depth, **measurements)

        progress('strict_parity')
        reference, split = formal(), decomposed(Markers())
        torch.cuda.synchronize()
        record['decomposed_parity'] = verify_equal(split, reference)
        record['output'] = describe(reference)
        del split, reference
        gc.collect()
        traces_before = renderer.geometry_info['trace_calls']
        progress('formal_warm_measure')
        record['formal_warm'] = measure(formal, args.warmup, args.repeats, args.event_repeats)
        progress('profiler_prime_outside_timers')
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True, with_stack=False):
            output = formal()
            torch.cuda.synchronize()
            del output
        record['profiler_prime'] = 'one discarded formal invocation initializes CUPTI/activity buffers before recorded profiles; not included in formal warm timings'
        progress('formal_profiler_no_internal_barriers')
        record['formal_profiler'] = profiler_probe(formal, destination.with_suffix('.formal.trace.json'), args.profile_steps)
        progress('annotated_profiler_no_internal_barriers')
        record['annotated_profiler'] = profiler_probe(lambda: decomposed(Markers()),
                                                      destination.with_suffix('.annotated.trace.json'), args.profile_steps)
        progress('synchronized_subblocks_diagnostic')
        # Warm the exact decomposed expressions without per-block synchronization.
        for _ in range(args.warmup):
            output = decomposed(Markers())
            torch.cuda.synchronize()
            del output
        blocks = {}
        for _ in range(args.block_repeats):
            output = decomposed(Markers(blocks))
            torch.cuda.synchronize()
            del output
        record['synchronized_subblocks'] = {
            'scope': 'diagnostic with a barrier and CUDA events around each block; scheduler/allocator altered; nested camera parent blocks inclusive; not original-function exclusive/additive costs',
            'blocks': {name: {'wall': summarize([x['wall_ms'] for x in samples]),
                              'cuda_event_span': summarize([x['cuda_event_span_ms'] for x in samples])}
                       for name, samples in blocks.items()}}
        if args.cuda_graph:
            progress('cuda_graph_diagnostic')
            reference = formal()
            torch.cuda.synchronize()

            @contextmanager
            def changed_input():
                saved = full_hits.distance_m.clone()
                full_hits.distance_m.mul_(0.5)
                try:
                    yield
                finally:
                    full_hits.distance_m.copy_(saved)

            try:
                record['cuda_graph'] = graph_probe(formal, reference, changed_input, args)
            except Exception as error:
                record['cuda_graph'] = {'status': 'unavailable', 'error': str(error),
                                        'traceback': traceback.format_exc(), 'scope': 'optional diagnostic; no production changes'}
            del reference
        record['postprocess_query_contract'] = {'trace_calls_before': traces_before,
                                                'trace_calls_after': renderer.geometry_info['trace_calls'],
                                                'passed': traces_before == renderer.geometry_info['trace_calls']}
        if not record['postprocess_query_contract']['passed']:
            raise AssertionError('postprocess ran an extra scene query')
        record['status'] = 'ok'
        progress('complete')
    except Exception as error:
        record.update(status='error', error=str(error), traceback=traceback.format_exc(), memory_at_failure=allocator())
    finally:
        if renderer is not None:
            renderer.close()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/devtools.json'))
    parser.add_argument('--mode', choices=('tof', 'camera'), required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--event-repeats', type=int, default=5)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--block-repeats', type=int, default=5)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--cuda-graph', action='store_true')
    args = parser.parse_args()
    if (args.batch < 1 or args.warmup < 0 or min(args.repeats, args.profile_steps, args.block_repeats,
                                               args.cpu_threads) < 1 or args.event_repeats < 0):
        parser.error('invalid batch/repeat/thread count')
    result = run(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'mode': args.mode, 'batch': args.batch,
                      'output': args.output, 'error': result.get('error')}), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
