"""Local, untracked benchmark for the current GPU sensor implementation.

Run only in project Docker, for example:
  python outputs/sensor_perf/current_benchmark.py \
      --results outputs/sensor_perf/current-results.json --gpu-timer

Baseline calls run unchanged. A separate traced run temporarily wraps functions
and ModernGL objects in this process to expose inclusive CPU stages, draw-submit
and blocking readback times. Traced timings include instrumentation overhead and
must not replace baseline numbers. No project source file is edited.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).resolve().parent


def output_path(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT / "outputs"):
        raise ValueError(f"benchmark writes must stay in {ROOT / 'outputs'}: {result}")
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summary(values):
    return {"count": len(values), "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95), "min_ms": min(values),
            "max_ms": max(values), "mean_ms": statistics.mean(values),
            "total_ms": sum(values)}


def measure(action, warmup, repeats, units=1, poses=None):
    for index in range(warmup):
        action(poses[index % len(poses)]) if poses is not None else action()
    walls, cpus = [], []
    for index in range(repeats):
        wall, cpu = time.perf_counter(), time.process_time()
        action(poses[index % len(poses)]) if poses is not None else action()
        cpus.append((time.process_time() - cpu) * 1000)
        walls.append((time.perf_counter() - wall) * 1000)
    return {"wall": summary(walls), "process_cpu": summary(cpus),
            "units_per_call": units, "wall_median_ms_per_unit": statistics.median(walls) / units,
            "wall_ms_samples": walls, "process_cpu_ms_samples": cpus}


def prepare_poses(args):
    """Seven plain-text columns, with deterministic full-RPY motion if new."""
    path = output_path(args.poses)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        rows = []
        for index in range(args.pose_count):
            phase = 2 * math.pi * index / args.pose_count
            roll, pitch, yaw = (math.radians(8) * math.sin(3*phase),
                                math.radians(6) * math.cos(2*phase),
                                math.radians(12) * math.sin(phase))
            sr, cr = math.sin(roll/2), math.cos(roll/2)
            sp, cp = math.sin(pitch/2), math.cos(pitch/2)
            sy, cy = math.sin(yaw/2), math.cos(yaw/2)
            rows.append((0.15*math.sin(phase), 0.10*math.cos(phase),
                         1.5+0.08*math.sin(2*phase), sr*cp*cy-cr*sp*sy,
                         cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy))
        path.write_text("".join(" ".join(format(value, ".17g") for value in row) + "\n"
                                for row in rows), encoding="utf-8")
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = tuple(float(value) for value in line.split())
        if len(row) != 7 or not all(math.isfinite(value) for value in row) or abs(math.hypot(*row[3:]) - 1) > 1e-6:
            raise ValueError(f"invalid pose row {number}: expected x y z qx qy qz qw")
        values.append(row)
    if not values:
        raise ValueError("poses file is empty")
    return values, {"path": str(path), "count": len(values),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "columns": "x_m y_m z_m qx qy qz qw",
                    "order": "timed calls restart at row zero after warmup and cycle in file order"}


def run_command(arguments):
    completed = subprocess.run([str(item) for item in arguments], cwd=ROOT,
                               capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed: {arguments}\n{completed.stdout}\n{completed.stderr}")


def map_generator(path):
    if path is not None:
        executable = Path(path).resolve()
        if not executable.is_file():
            raise ValueError(f"missing map generator: {executable}")
        return executable
    existing = ROOT / "outputs" / "mapgen" / "build" / "offline_mapgen"
    if existing.is_file():
        return existing
    build = output_path(ARTIFACTS / "mapgen-build")
    build.mkdir(parents=True, exist_ok=True)
    run_command(("cmake", "-S", ROOT / "tools" / "mapgen", "-B", build,
                 "-DCMAKE_BUILD_TYPE=Release"))
    run_command(("cmake", "--build", build, "--parallel", "2"))
    return build / "offline_mapgen"


def prepare_map(generator, count, args):
    specification = {"count": count, "seed": args.seed, "size": args.size,
                     "width_min": 0.3, "width_max": 1.0, "height_min": 1.0,
                     "height_max": min(10.0, args.size[2])}
    token = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:12]
    folder = output_path(ARTIFACTS / "maps")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"pillars-{count}-{token}.obj"
    created = not path.exists()
    started = time.perf_counter()
    if created:
        run_command((generator, "--type", "pillars", "--seed", args.seed,
                     "--size-x", args.size[0], "--size-y", args.size[1], "--size-z", args.size[2],
                     "--count", count, "--width-min", 0.3, "--width-max", 1.0,
                     "--height-min", 1.0, "--height-max", specification["height_max"],
                     "--output", path))
    return path, {"specification": specification, "newly_generated": created,
                  "prepare_wall_ms": (time.perf_counter() - started) * 1000,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class Recorder:
    def __init__(self, gpu_timer):
        self.phase = None
        self.samples = defaultdict(lambda: defaultdict(lambda: {"wall": [], "cpu": []}))
        self.counters = defaultdict(lambda: defaultdict(int))
        self.queries = []
        self.gpu_timer = gpu_timer
        self.gpu_timer_errors = []

    def call(self, name, action, *args, **kwargs):
        if self.phase is None:
            return action(*args, **kwargs)
        phase = self.phase
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            return action(*args, **kwargs)
        finally:
            entry = self.samples[phase][name]
            entry["wall"].append((time.perf_counter() - wall) * 1000)
            entry["cpu"].append((time.process_time() - cpu) * 1000)

    def count(self, name, amount):
        if self.phase is not None:
            self.counters[self.phase][name] += amount

    def finish_gpu_queries(self):
        for phase, query in self.queries:
            # All baseline measurements are already finished. These reads can
            # synchronize; they are reported only in the instrumented section.
            nanoseconds = int(query.elapsed)
            entry = self.samples[phase]["gpu_draw_elapsed"]
            entry["wall"].append(nanoseconds / 1_000_000)
            entry["cpu"].append(0.0)
        self.queries.clear()

    def result(self):
        return {"stages": {phase: {name: {"wall": summary(entry["wall"]),
                                         "process_cpu": summary(entry["cpu"])}
                                   for name, entry in entries.items()}
                           for phase, entries in self.samples.items()},
                "counters": {phase: dict(values) for phase, values in self.counters.items()},
                "gpu_timer_requested": self.gpu_timer, "gpu_timer_errors": self.gpu_timer_errors,
                "notes": ["Nested stages are inclusive: do not add their totals.",
                          "process_cpu is process CPU time, not GPU time; GPU waiting appears in wall time.",
                          "draw_submit excludes blocking readback; readback includes queued GPU completion and driver transfer.",
                          "Optional GPU queries wrap draw commands only, not readback/CPU postprocessing.",
                          "These traced numbers include wrappers/query overhead; use baseline for throughput."]}


class FramebufferProxy:
    def __init__(self, raw, recorder):
        self.raw, self.recorder = raw, recorder

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def read(self, *args, **kwargs):
        result = self.recorder.call("fbo_readback_sync", self.raw.read, *args, **kwargs)
        self.recorder.count("readbacks", 1)
        self.recorder.count("readback_bytes", len(result))
        return result


class VertexArrayProxy:
    def __init__(self, raw, context, recorder):
        self.raw, self.context, self.recorder = raw, context, recorder

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def render(self, *args, **kwargs):
        self.recorder.count("draws", 1)
        if self.recorder.phase is None or not self.recorder.gpu_timer:
            return self.recorder.call("draw_submit", self.raw.render, *args, **kwargs)
        try:
            query = self.context.query(time=True)
        except Exception as error:
            self.recorder.gpu_timer_errors.append(str(error))
            self.recorder.gpu_timer = False
            return self.recorder.call("draw_submit", self.raw.render, *args, **kwargs)
        with query:
            result = self.recorder.call("draw_submit", self.raw.render, *args, **kwargs)
        self.recorder.queries.append((self.recorder.phase, query))
        return result


class ContextProxy:
    def __init__(self, raw, recorder):
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "recorder", recorder)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __setattr__(self, name, value):
        setattr(self.raw, name, value)

    def program(self, *args, **kwargs):
        return self.recorder.call("gl_program_compile_link", self.raw.program, *args, **kwargs)

    def buffer(self, *args, **kwargs):
        self.recorder.count("vbo_upload_bytes", len(args[0]) if args else len(kwargs.get("data", b"")))
        return self.recorder.call("gl_vbo_upload", self.raw.buffer, *args, **kwargs)

    def vertex_array(self, *args, **kwargs):
        raw = self.recorder.call("gl_vao_create", self.raw.vertex_array, *args, **kwargs)
        return VertexArrayProxy(raw, self.raw, self.recorder)

    def framebuffer(self, *args, **kwargs):
        raw = self.recorder.call("gl_framebuffer_create", self.raw.framebuffer, *args, **kwargs)
        return FramebufferProxy(raw, self.recorder)


class Instrumentation:
    def __init__(self, recorder, sensors, moderngl, np):
        self.recorder, self.sensors, self.moderngl, self.np = recorder, sensors, moderngl, np
        self.originals = []

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            return self.recorder.call(label, original, *args, **kwargs)

        self.originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    def __enter__(self):
        for name in ("_sphere_mesh", "_rasterize", "_coverage", "_readonly",
                     "pixel_directions", "mounted_pose"):
            owner = self.sensors.MeshSensor if name == "_rasterize" else self.sensors
            self.wrap(owner, name, name)
        self.wrap(self.sensors.Pose, "as_transform", "pose_as_transform_including_validation")
        self.wrap(self.np.linalg, "svd", "numpy_svd")
        original = self.moderngl.create_context

        def create_context(*args, **kwargs):
            raw = self.recorder.call("egl_context_create", original, *args, **kwargs)
            return ContextProxy(raw, self.recorder)

        self.originals.append((self.moderngl, "create_context", original))
        self.moderngl.create_context = create_context
        return self

    def __exit__(self, *args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


def depth_only_rig(sensor, config, pose, sensors):
    frames = tuple(sensor.sample(config.tof, sensors.mounted_pose(pose, mount))
                   for mount in config.tof_mounts)
    camera = sensor.sample(config.camera, sensors.mounted_pose(pose, config.camera_mount))
    return frames, camera


def profile_scene(scene, config, pose, args, sensors, moderngl, np, depth_only_mode=None):
    depth_only_mode = args.depth_only_target_outside if depth_only_mode is None else depth_only_mode
    recorder = Recorder(args.gpu_timer)
    with Instrumentation(recorder, sensors, moderngl, np):
        recorder.phase = "constructor"
        sensor = recorder.call("MeshSensor_constructor", sensors.MeshSensor, scene)
        try:
            recorder.phase = None
            for _ in range(args.warmup):
                if depth_only_mode:
                    depth_only_rig(sensor, config, pose, sensors)
                else:
                    sensors.sample_rig(sensor, config, pose)
            recorder.phase = "steady_depth_only_rig" if depth_only_mode else "steady_normal_rig"
            for _ in range(args.profile_repeats):
                if depth_only_mode:
                    recorder.call("depth_only_rig", depth_only_rig, sensor, config, pose, sensors)
                else:
                    recorder.call("sample_rig", sensors.sample_rig, sensor, config, pose)
            recorder.finish_gpu_queries()
        finally:
            recorder.phase = None
            sensor.close()
    result = recorder.result()
    result["profile_rig_calls"] = args.profile_repeats
    result["workload"] = "depth_only_target_outside" if depth_only_mode else "normal_target_mask_rig"
    return result


def rebuild(scene, config, pose, repeats, sensors):
    construction, first_rig, closure, whole = [], [], [], []
    for index in range(repeats):
        started = time.perf_counter()
        changed = scene.with_target_position((4 + 0.01 * index, 0.4, 1.5))
        before = time.perf_counter()
        sensor = sensors.MeshSensor(changed)
        construction.append((time.perf_counter() - before) * 1000)
        try:
            before = time.perf_counter()
            sensors.sample_rig(sensor, config, pose)
            first_rig.append((time.perf_counter() - before) * 1000)
        finally:
            before = time.perf_counter()
            sensor.close()
            closure.append((time.perf_counter() - before) * 1000)
        whole.append((time.perf_counter() - started) * 1000)
    return {"constructor_wall": summary(construction), "first_rig_wall": summary(first_rig),
            "close_wall": summary(closure), "target_change_to_close_wall": summary(whole),
            "notes": "New target snapshot shares MeshMap, but each iteration recreates EGL/program/full VBO; no OBJ reparse included."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=(0, 20, 500, 5000))
    parser.add_argument("--size", type=float, nargs=3, default=(60, 60, 15), metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-repeats", type=int, default=20)
    parser.add_argument("--rebuild-repeats", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--mapgen", type=Path)
    parser.add_argument("--gpu-timer", action="store_true")
    parser.add_argument("--depth-only-target-outside", action="store_true")
    parser.add_argument("--poses", type=Path, default=ARTIFACTS / "poses.txt")
    parser.add_argument("--pose-count", type=int, default=200)
    parser.add_argument("--write-poses-only", action="store_true")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        parser.error("run this benchmark inside project Docker")
    if any(value <= 0 for value in (args.repeats, args.batch_size, args.batch_repeats, args.rebuild_repeats,
                                     args.profile_repeats, args.pose_count)) or args.warmup < 1:
        parser.error("repeat/batch/warmup values must be positive")
    if any(count < 0 for count in args.counts) or any(not math.isfinite(v) or v <= 0 for v in args.size) or args.size[2] < 2:
        parser.error("counts must be nonnegative and sizes finite/positive with Z>=2")
    pose_rows, pose_info = prepare_poses(args)
    if args.write_poses_only:
        print(json.dumps(pose_info, indent=2), flush=True)
        return
    if args.results is None:
        parser.error("--results is required unless --write-poses-only is used")
    result_path = output_path(args.results)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    import moderngl
    import numpy as np
    import devtools.sensors as sensors
    from devtools.config import load_config
    from devtools.mesh import MeshMap
    from devtools.scene import Pose, Scene, Sphere

    configuration = load_config(ROOT / "configs" / "devtools.json")
    config = configuration.sensors
    if config is None:
        raise ValueError("explicit sensor configuration required")
    generator = map_generator(args.mapgen)
    poses = tuple(Pose(row[:3], row[3:]) for row in pose_rows)
    pose = poses[0]
    batch_poses = tuple(poses[index % len(poses)] for index in range(args.batch_size))
    cases = []
    for count in args.counts:
        print(f"benchmark: preparing {count} pillars", flush=True)
        asset, map_info = prepare_map(generator, count, args)
        started = time.perf_counter()
        background = MeshMap.from_obj(asset)
        parse_ms = (time.perf_counter() - started) * 1000
        if len(background.objects) != count + 1 or background.size_xyz_m != tuple(args.size):
            raise ValueError(f"cached map does not match requested geometry: {asset}")
        normal = Scene(background, Sphere(pose, configuration.vehicle_radius_m),
                       Sphere(Pose((4, 0.4, 1.5)), configuration.target_radius_m), 0)
        current = normal.with_target_position((1000, 1000, 1000)) if args.depth_only_target_outside else normal
        case = {"pillars": count, "map": str(asset), "map_info": map_info,
                "obj_parse_and_validation_wall_ms_single": parse_ms,
                "mesh_vertices": background.vertex_count, "mesh_triangles": background.triangle_count,
                "measurement_target_m": [1000, 1000, 1000] if args.depth_only_target_outside else [4, 0.4, 1.5],
                "comparison_mode": args.depth_only_target_outside}
        started = time.perf_counter()
        with sensors.MeshSensor(current) as sensor:
            case["initial_constructor_wall_ms_single"] = (time.perf_counter() - started) * 1000
            case["renderer_info"] = dict(sensor.renderer_info)
            case["render_triangles"] = sensor.triangle_count
            case["expanded_vbo_bytes_formula"] = sensor.triangle_count * 3 * 7 * 4
            def four_tof(body_pose):
                return tuple(sensor.sample(config.tof, sensors.mounted_pose(body_pose, mount))
                             for mount in config.tof_mounts)

            actions = [("single_front_tof", lambda p: sensor.sample(config.tof, p)),
                       ("four_tof_with_mounts", four_tof),
                       ("camera_depth_only", lambda p: sensor.sample(config.camera, sensors.mounted_pose(p, config.camera_mount))),
                       ("matched_depth_rig", lambda p: depth_only_rig(sensor, config, p, sensors))]
            if not args.depth_only_target_outside:
                actions.extend((("camera_depth_mask_hint", lambda p: sensor.camera(config.camera, sensors.mounted_pose(p, config.camera_mount), config.min_visible_fraction)),
                                ("rig", lambda p: sensors.sample_rig(sensor, config, p))))
            case["baseline"] = {name: measure(action, args.warmup, args.repeats, poses=poses)
                                for name, action in actions}
            batches = (("tof_sample_many", lambda: sensor.sample_many(config.tof, batch_poses)),
                       ("tof_explicit_sample_loop", lambda: tuple(sensor.sample(config.tof, p) for p in batch_poses)),
                       ("camera_depth_sample_many", lambda: sensor.sample_many(config.camera, batch_poses)))
            case["baseline_batch_loops"] = {name: measure(action, args.warmup, args.batch_repeats, args.batch_size)
                                             for name, action in batches}
        if args.depth_only_target_outside:
            with sensors.MeshSensor(normal) as sensor:
                case["normal_target_mask_rig_separate"] = measure(
                    lambda p: sensors.sample_rig(sensor, config, p), args.warmup, args.repeats, poses=poses)
        case["target_changed_rebuild"] = rebuild(normal, config, pose, args.rebuild_repeats, sensors)
        case["instrumented"] = profile_scene(current, config, pose, args, sensors, moderngl, np)
        if args.depth_only_target_outside:
            case["instrumented_normal_rig"] = profile_scene(normal, config, pose, args, sensors, moderngl, np,
                                                            depth_only_mode=False)
        cases.append(case)
        print(f"benchmark: {count} pillars matched depth rig median={case['baseline']['matched_depth_rig']['wall']['median_ms']:.3f} ms", flush=True)
    result = {"scope": "current implementation on generated pillar maps; no reference-project performance claim",
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "moderngl": moderngl.__version__},
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("devtools/sensors.py", "devtools/config.py", "src/tof_tracker/geometry.py")},
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "config": asdict(config), "poses": pose_info, "cases": cases,
              "static_work": {"target_sphere_vertices": 8066, "target_sphere_triangles": 16128,
                              "expanded_target_vertices": 48384, "expanded_target_vbo_bytes": 1354752,
                              "draws_per_rig": 6, "readbacks_per_rig": 6,
                              "full_scene_draws_per_rig": 5, "target_only_draws_per_rig": 1,
                              "default_rig_readback_bytes": 4*8*8*8 + 2*160*120*8,
                              "matched_depth_rig_draws_readbacks": 5,
                              "matched_depth_rig_readback_bytes": 4*8*8*8 + 160*120*8,
                              "pose_transform_svd_calls_per_rig_with_bounds": 17,
                              "pixel_directions_calls_per_rig_with_bounds": 6,
                              "readonly_tobytes_copies_per_rig": 13},
              "limitations": ["No concurrent benchmark or training process should share the GPU.",
                              "Each baseline operation has separate warmup; framebuffer creation excluded from steady state.",
                              "Constructor/rebuild include EGL/program/VBO, and first rig includes new framebuffers.",
                              "Map preparation/OBJ parse are outside steady-state sensing measurements.",
                              "sample_many is a sequential public-API loop, not a GPU batch.",
                              "Camera sample_many measures depth only; camera_depth_mask_hint also performs target-only rasterization.",
                              "Comparison mode moves the target sphere outside all ranges/views but retains its 16128 triangles in VBO preparation and draws.",
                              "Normal target=(4,.4,1.5) mask rig is separately reported and has no equivalent old mask workload.",
                              "Pillar count changes geometry; visible pixels, overdraw and occlusion also affect GPU timings.",
                              "Traced stages are inclusive and perturb runtime; only baseline wall times support throughput comparisons."]}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"benchmark results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
