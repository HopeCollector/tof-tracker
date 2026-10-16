"""One isolated GPU timing cell. Run through run_eval.py, inside Docker only."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sensor_reference import MeshSensor
from experiments.pytorch3d_render_eval.fixtures import CONFIG, load_scene, make_poses, poses_tensor
from experiments.pytorch3d_render_eval.postprocess import postprocess_depth, postprocess_camera


class MemoryMonitor:
    """Device-wide NVML sampling; includes other processes, never called inside a timer."""
    def __init__(self):
        import pynvml
        self.api = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.before = self.read()
        self.peak = self.before
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def read(self):
        return int(self.api.nvmlDeviceGetMemoryInfo(self.handle).used)

    def run(self):
        while not self.stop.wait(.005):
            self.peak = max(self.peak, self.read())
            self.samples += 1

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        final = self.read()
        return {"scope": "whole device; sampled 5ms; includes other processes and CUDA/GL drivers",
                "before_bytes": self.before, "peak_bytes": max(self.peak, final),
                "peak_increment_bytes": max(self.peak, final) - self.before,
                "end_bytes": final, "samples": self.samples}


def allocator():
    if not torch.cuda.is_initialized():
        return {"allocated_bytes": 0, "reserved_bytes": 0,
                "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def measure(action, *, cuda, repeats=7, warmup=2):
    sync = torch.cuda.synchronize if cuda else lambda: None
    for _ in range(warmup):
        value = action()
        sync()
        del value
    walls, events = [], []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        value = action()
        sync()
        walls.append((time.perf_counter() - start) * 1000)
        del value
    # Separate probe: an event span includes stream idle while the CPU submits
    # work / upstream scalar synchronizations, and is not pure kernel time.
    if cuda:
        for _ in range(min(repeats, 3)):
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            sync()
            start_event.record()
            value = action()
            end_event.record()
            sync()
            events.append(start_event.elapsed_time(end_event))
            del value
    return {"wall_ms": walls, "wall_median_ms": statistics.median(walls),
            "wall_p95_ms": float(np.quantile(walls, .95)), "repeats": repeats,
            "cuda_event_span_ms": events,
            "cuda_event_span_median_ms": statistics.median(events) if events else None}


def main(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(7)
    monitor = MemoryMonitor()
    monitor.start()
    cuda = args.backend != "current"
    if cuda:
        torch.cuda.set_per_process_memory_fraction(.8)
        torch.cuda.init()
    record = {"status": "running", "backend": args.backend, "mode": args.mode,
              "batch_views": args.batch, "pillars": args.pillars,
              "torch_memory_fraction_cap": .8 if cuda else None,
              "output_location": "cuda:0" if cuda else "cpu",
              "timing_scope": "prepared input poses; warm scene; render + required postprocessing; synchronized wall",
              "grad_enabled": False}
    try:
        scene, scene_info = load_scene(args.pillars)
        config = getattr(CONFIG.sensors, args.mode)
        snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        bounds = snapshot.map_bounds_m
        poses = make_poses(args.batch, args.mode)
        is_p3d = args.backend.startswith("p3d")
        matrices = poses_tensor(poses) if is_p3d else None
        record.update(scene=scene_info, config=asdict(config))
        started = time.perf_counter()
        if is_p3d:
            import_started = time.perf_counter()
            if args.backend == "p3d_conditioned":
                from experiments.pytorch3d_render_eval.conditioned_backend import P3DRenderer
            else:
                from experiments.pytorch3d_render_eval.backend import P3DRenderer
            record["backend_import_ms"] = (time.perf_counter() - import_started) * 1000
            capacity = args.bin_capacity
            if capacity not in (None, "pytorch3d_default"):
                capacity = int(capacity)
            started = time.perf_counter()
            renderer = P3DRenderer(scene, max_faces_per_bin=capacity)
            torch.cuda.synchronize()
            record["base_prepare_ms"] = (time.perf_counter() - started) * 1000
            record["geometry"] = renderer.geometry_info
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            preparation = renderer.prepare(args.batch, include_target=args.mode == "camera")
            torch.cuda.synchronize()
            record["batch_prepare_ms"] = (time.perf_counter() - started) * 1000
            record["batch_prepare"] = preparation
            record["memory_after_prepare"] = allocator()

            def raw():
                full = renderer.render(matrices, config)
                if args.mode == "camera":
                    return full, renderer.render(matrices, config, target_only=True)
                return full

            def post(data):
                full = data[0] if args.mode == "camera" else data
                ranges = postprocess_depth(*full, matrices, config, bounds)
                output = {"range_m": ranges, "object_index": full[1]}
                if args.mode == "camera":
                    output.update(postprocess_camera(ranges, full[1], *data[1], matrices, config,
                                                     bounds, renderer.object_ids.index("target"),
                                                     CONFIG.sensors.min_visible_fraction))
                return output

            action = lambda: post(raw())
        else:
            renderer = MeshSensor(scene)
            record["base_prepare_ms"] = (time.perf_counter() - started) * 1000
            record["batch_prepare_ms"] = 0
            record["geometry"] = {"triangle_count": renderer.triangle_count,
                                  "geometry_uploads": renderer.geometry_uploads,
                                  "renderer": renderer.renderer_info}

            def action():
                if args.mode == "tof":
                    frames = renderer.sample_many(config, poses)
                    if args.backend == "current":
                        return frames
                    data = {"range_m": np.stack([v.range_m for v in frames]),
                            "object_index": np.stack([v.object_index for v in frames])}
                else:
                    frames = tuple(renderer.camera(config, pose, CONFIG.sensors.min_visible_fraction)
                                   for pose in poses)
                    if args.backend == "current":
                        return frames
                    data = {"range_m": np.stack([v.depth.range_m for v in frames]),
                            "object_index": np.stack([v.depth.object_index for v in frames]),
                            "target_mask": np.stack([v.target_mask for v in frames]),
                            "target_hint": np.stack([v.target_hint for v in frames]),
                            "visible_fraction": np.array([v.visible_fraction for v in frames]),
                            "unoccluded_pixels": np.array([v.unoccluded_pixels for v in frames])}
                return {key: torch.as_tensor(value, device="cuda") for key, value in data.items()}

        # First forward includes lazy shader/raster allocations, but not asset load/extend.
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        first = action()
        if cuda:
            torch.cuda.synchronize()
        record["first_forward_ms"] = (time.perf_counter() - started) * 1000
        record["first_forward_memory"] = allocator()
        if cuda:
            record["output_tensors"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                               "device": str(value.device)} for key, value in first.items()}
            record["first_output_summary"] = {
                "finite": int(torch.isfinite(first["range_m"]).sum().item()),
                "inf": int(torch.isposinf(first["range_m"]).sum().item()),
                "nan": int(torch.isnan(first["range_m"]).sum().item())}
        del first
        gc.collect()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        repeats = max(3, min(args.repeats, int(2000 / max(record["first_forward_ms"], .1))))
        record["end_to_end"] = measure(action, cuda=cuda, repeats=repeats, warmup=args.warmup)
        record["end_to_end"]["views_per_second"] = args.batch * 1000 / record["end_to_end"]["wall_median_ms"]
        record["warm_memory"] = allocator()
        if is_p3d:
            # Stage timings are separate runs, not numbers to add together.
            record["raster_stage"] = measure(raw, cuda=True, repeats=min(repeats, 5), warmup=1)
            raw_result = raw()
            torch.cuda.synchronize()
            record["batched_postprocess_stage"] = measure(lambda: post(raw_result), cuda=True,
                                                          repeats=min(repeats, 5), warmup=1)
            del raw_result
            renderer.release_batch()
        else:
            renderer.close()
        record["status"] = "ok"
    except torch.cuda.OutOfMemoryError as error:
        record.update(status="oom", error=str(error), memory_at_failure=allocator())
    except Exception as error:
        record.update(status="error", error=str(error), traceback=traceback.format_exc())
    finally:
        record["nvml"] = monitor.finish()
        Path(args.output).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: record[key] for key in ("backend", "mode", "batch_views", "pillars", "status")}), flush=True)
    return 0 if record["status"] in ("ok", "oom") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("p3d", "p3d_conditioned", "current", "current_gpu"), required=True)
    parser.add_argument("--mode", choices=("tof", "camera"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--pillars", type=int, default=500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--bin-capacity", default=None)
    sys.exit(main(parser.parse_args()))
