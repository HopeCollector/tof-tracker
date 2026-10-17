"""One isolated GL batch timing cell. Execute inside the experiment Docker image."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pytorch3d_render_eval.fixtures import CONFIG, load_scene, make_poses, poses_tensor
from experiments.pytorch3d_render_eval.postprocess import postprocess_camera, postprocess_depth
from experiments.pytorch3d_render_eval.worker import MemoryMonitor, allocator, measure


def backend_memory(renderer):
    """GL allocations are outside the Torch allocator and must be reported separately."""
    info = getattr(renderer, "memory_info", None)
    if callable(info):
        info = info()
    return info if isinstance(info, dict) else {"status": "backend_not_reported"}


def close_renderer(renderer):
    if renderer is not None:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()


def main(args):
    if args.backend == "current_gpu":
        # Preserve the existing comparison exactly: one GL draw/readback and CPU
        # postprocess per view, then stack and upload all required outputs.
        from experiments.pytorch3d_render_eval.worker import main as baseline_main
        return baseline_main(args)

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(7)
    monitor = MemoryMonitor()
    monitor.start()
    renderer = None
    record = {"status": "running", "backend": args.backend, "mode": args.mode,
              "batch_views": args.batch, "pillars": args.pillars,
              "torch_memory_fraction_cap": .8, "output_location": "cuda:0",
              "timing_scope": "prepared CUDA input poses; warm scene; raster + GL/CUDA transfer/synchronization + required GPU postprocessing; synchronized wall",
              "grad_enabled": False,
              "memory_scope": "Torch allocator, explicitly tracked GL allocations, and sampled whole-device NVML are separate measurements"}
    try:
        torch.cuda.set_per_process_memory_fraction(.8)
        torch.cuda.init()
        scene, scene_info = load_scene(args.pillars)
        config = getattr(CONFIG.sensors, args.mode)
        bounds = scene.get_geometry(exclude_ids={"vehicle"}).map_bounds_m
        poses = make_poses(args.batch, args.mode)
        matrices = poses_tensor(poses)
        record.update(scene=scene_info, config=asdict(config))
        import_started = time.perf_counter()
        from experiments.opengl_batch_eval.backend import GLBatchRenderer
        record["backend_import_ms"] = (time.perf_counter() - import_started) * 1000
        started = time.perf_counter()
        renderer = GLBatchRenderer(scene)
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
        record["gl_memory_after_prepare"] = backend_memory(renderer)

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
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        geometry_before_first = renderer.geometry_info
        started = time.perf_counter()
        first = action()
        torch.cuda.synchronize()
        record["first_forward_ms"] = (time.perf_counter() - started) * 1000
        geometry_after_first = renderer.geometry_info
        record["geometry_after_first_forward"] = geometry_after_first
        expected_calls = 2 if args.mode == "camera" else 1
        draw_delta = geometry_after_first["instanced_draw_count"] - geometry_before_first["instanced_draw_count"]
        render_delta = geometry_after_first["render_calls"] - geometry_before_first["render_calls"]
        unchanged_geometry = (
            geometry_before_first["scene_geometry_buffer_name"] == geometry_after_first["scene_geometry_buffer_name"]
            and geometry_before_first["scene_geometry_buffer_bytes"] == geometry_after_first["scene_geometry_buffer_bytes"]
            and geometry_before_first["geometry_uploads"] == geometry_after_first["geometry_uploads"] == 1)
        record["first_forward_contract"] = {
            "expected_calls_per_action": expected_calls,
            "instanced_draw_count_delta": draw_delta, "render_calls_delta": render_delta,
            "geometry_uploads_before": geometry_before_first["geometry_uploads"],
            "geometry_uploads_after": geometry_after_first["geometry_uploads"],
            "scene_geometry_buffer_name_before": geometry_before_first["scene_geometry_buffer_name"],
            "scene_geometry_buffer_name_after": geometry_after_first["scene_geometry_buffer_name"],
            "one_shared_geometry_unchanged": unchanged_geometry,
            "passed": draw_delta == render_delta == expected_calls and unchanged_geometry}
        if not record["first_forward_contract"]["passed"]:
            raise RuntimeError("first-forward draw count or shared-geometry contract failed")
        record["first_forward_memory"] = allocator()
        record["gl_memory_after_first_forward"] = backend_memory(renderer)
        record["output_tensors"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                           "device": str(value.device)} for key, value in first.items()}
        record["first_output_summary"] = {
            "finite": int(torch.isfinite(first["range_m"]).sum().item()),
            "inf": int(torch.isposinf(first["range_m"]).sum().item()),
            "nan": int(torch.isnan(first["range_m"]).sum().item())}
        del first
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        repeats = max(3, min(args.repeats, int(2000 / max(record["first_forward_ms"], .1))))
        record["end_to_end"] = measure(action, cuda=True, repeats=repeats, warmup=args.warmup)
        record["end_to_end"]["views_per_second"] = args.batch * 1000 / record["end_to_end"]["wall_median_ms"]
        record["warm_memory"] = allocator()
        record["gl_warm_memory"] = backend_memory(renderer)
        record["geometry_after_warm"] = renderer.geometry_info
        # Independent probes; these numbers are not additive with the complete call.
        record["raster_stage"] = measure(raw, cuda=True, repeats=min(repeats, 5), warmup=1)
        raw_result = raw()
        torch.cuda.synchronize()
        record["batched_postprocess_stage"] = measure(lambda: post(raw_result), cuda=True,
                                                      repeats=min(repeats, 5), warmup=1)
        del raw_result
        record["gl_memory_after_probes"] = backend_memory(renderer)
        record["status"] = "ok"
    except torch.cuda.OutOfMemoryError as error:
        record.update(status="oom", error=str(error), memory_at_failure=allocator())
    except Exception as error:
        record.update(status="error", error=str(error), traceback=traceback.format_exc())
    finally:
        # Measure process-resident allocations before releasing the GL context.
        record["nvml"] = monitor.finish()
        try:
            close_renderer(renderer)
        except Exception as error:
            record["cleanup_error"] = str(error)
            if record["status"] == "ok":
                record["status"] = "cleanup_error"
        Path(args.output).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: record[key] for key in ("backend", "mode", "batch_views", "pillars", "status")}), flush=True)
    return 0 if record["status"] in ("ok", "oom") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("current_gpu", "gl_batch"), required=True)
    parser.add_argument("--mode", choices=("tof", "camera"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--pillars", type=int, default=500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    # Baseline worker namespace compatibility; unused by the OpenGL paths.
    parser.add_argument("--bin-capacity", default=None, help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
