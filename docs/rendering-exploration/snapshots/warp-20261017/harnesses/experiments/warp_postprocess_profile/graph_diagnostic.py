"""Prepared-constant scheduling wrapper diagnostic; original API capture fails.

Only in this experiment, a strong-held, already-synchronized bounds tensor
replaces the Python cache/event/record_stream wrapper during eager/capture/
replay. All numeric operations still use formal process_samples and camera
measurement functions. This is not direct graph support or a production change.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import torch
import devtools._ray_processing as processing
from experiments.warp_postprocess_profile import analyze as base


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
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    record = {'status': 'running', 'mode': args.mode, 'batch': args.batch,
              'scope': __doc__, 'original_direct_graph_capture': 'unsupported: cached bounds ready-event dependency'}
    renderer, original_bounds = None, processing._bounds
    paths = ('devtools/sensors.py', 'devtools/_ray_processing.py', 'devtools/ray_rendering.py',
             'experiments/warp_postprocess_profile/analyze.py',
             'experiments/warp_postprocess_profile/graph_diagnostic.py')
    record['source_sha256'] = {name: base.sha256(ROOT / name) for name in paths}
    record['inputs'] = {'map': args.map, 'map_sha256': base.sha256(args.map),
                        'config': args.config, 'config_sha256': base.sha256(args.config),
                        'pose_policy': 'warp_ray_perf make_poses default prefix; target 3m forward'}
    try:
        config = base.load_config(args.config)
        sensor = config.sensors
        views = base.make_poses(args.batch)
        first = base.make_poses(1)[0]
        scene = base.Scene.from_obj(args.map, config, first, first.body_to_world((3., 0., 0.)), t_ref=12.5)
        snapshot = scene.get_geometry(exclude_ids={'vehicle'})
        renderer = base.RayRenderer(snapshot)
        poses = base.poses_to_tensor(views)
        cfg = sensor.tof if args.mode == 'tof' else sensor.camera
        origins, directions = base.generate_rays(poses, cfg)
        full = renderer.trace(origins, directions, cfg.max_range_m)
        target = None if args.mode == 'tof' else renderer.trace(origins, directions, cfg.max_range_m, only_object='target')

        def formal():
            if args.mode == 'tof':
                return base.process_samples(full, origins, directions, poses, cfg,
                    reduction='weighted_quantile', quantile=sensor.tof_depth_quantile, signal_floor=sensor.tof_signal_floor)
            depth = base.process_samples(full, origins, directions, poses, cfg)
            target_depth = base.process_samples(target, origins, directions, poses, cfg)
            values = base.camera_measurements(depth.subray_range_m, depth.raw.subray_object_index,
                target_depth.subray_range_m, target_depth.raw.subray_object_index,
                cfg, full.object_ids.index('target'), sensor.min_visible_fraction)
            return base.CameraBatch(depth, **values)

        reference = formal()
        prepared_bounds = original_bounds(snapshot.map_bounds_m, poses.device)
        torch.cuda.synchronize()
        record['formal_eager_control'] = base.measure(formal, args.warmup, args.repeats, args.event_repeats)

        def bounds_wrapper(bounds, device):
            if bounds != snapshot.map_bounds_m or torch.device(device) != prepared_bounds.device:
                raise AssertionError('prepared wrapper used with different bounds/device')
            return prepared_bounds

        processing._bounds = bounds_wrapper
        prepared = formal()
        torch.cuda.synchronize()
        record['prepared_constant_eager_parity'] = base.verify_equal(prepared, reference)
        del prepared
        record['prepared_constant_eager'] = base.measure(formal, args.warmup, args.repeats, args.event_repeats)

        @contextmanager
        def changed_input():
            saved = full.distance_m.clone()
            full.distance_m.mul_(0.5)
            try:
                yield
            finally:
                full.distance_m.copy_(saved)

        record['graph'] = base.graph_probe(formal, reference, changed_input, args)
        record['graph']['scope'] = __doc__
        record['graph']['initialization'] = 'pre-synchronized bounds tensor held strongly; only Python bounds cache/event/record_stream scheduling wrapper bypassed in this diagnostic; restored finally'
        record['status'] = 'ok'
    except Exception as error:
        record.update(status='error', error=str(error), traceback=traceback.format_exc())
    finally:
        torch.cuda.synchronize()
        processing._bounds = original_bounds
        if renderer is not None:
            renderer.close()
        record['formal_source_unchanged'] = all(base.sha256(ROOT / name) == record['source_sha256'][name] for name in paths)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'mode': args.mode, 'batch': args.batch, 'error': record.get('error')}))
    return 0 if record['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
