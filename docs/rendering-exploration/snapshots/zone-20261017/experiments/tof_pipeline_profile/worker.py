"""One isolated ToF pipeline attribution cell. Run only in project Docker."""
from pathlib import Path
import argparse
import contextlib
from dataclasses import fields
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from devtools.config import load_config
from devtools.rendering import BatchRenderer, RasterBatch
from devtools.scene import Scene
from devtools.sensors import poses_to_tensor
from experiments.sensor_zone_perf.benchmark import make_poses, TARGET_POSITION
from experiments.tof_pipeline_profile.stages import profile, draw_probe
from experiments.tof_pipeline_profile.variants import get_variants, STATIC_ANALYSIS
from experiments.tof_pipeline_profile import counters


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected):
    report = {}
    for field in fields(RasterBatch):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, torch.Tensor):
            a, b = a.detach().cpu(), b.detach().cpu()
            if a.is_floating_point():
                finite = torch.isfinite(a) & torch.isfinite(b)
                report[field.name] = {"exact": bool(torch.allclose(a, b, atol=0, rtol=0, equal_nan=True)),
                                     "different_elements": int((~torch.isclose(a,b,atol=0,rtol=0,equal_nan=True)).sum()),
                                     "max_abs_finite": float((a[finite]-b[finite]).abs().max()) if bool(finite.any()) else 0.}
            else:
                report[field.name] = {"exact": bool(torch.equal(a,b)), "different_elements": int((a!=b).sum())}
        else:
            report[field.name] = {"exact": a == b}
    return {"exact": all(value["exact"] for value in report.values()), "fields": report}


def wall(action, repeats, warmup=2):
    for _ in range(warmup):
        value = action(); torch.cuda.synchronize(); del value
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = action()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
        del value
    return {"median_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples,.95)), "samples_ms": samples}


def install(renderer, shaders):
    """Only swap the experimental program; VBO, poses and native output path stay."""
    renderer._gl.activate()
    program = renderer._ctx.program(**shaders)
    renderer._zone_program = program
    renderer._zone_uniforms = {name: renderer._gl.glGetUniformLocation(program.glo, name.encode())
                              for name in ("tangent_fov", "image_size")}
    return program


@contextlib.contextmanager
def object_selection(renderer, selection):
    draw = renderer._gl.glDrawArraysInstanced
    first, count = renderer._object_ranges["target"]
    if selection == "map":
        assert first+count == renderer.vertex_count, "map-only diagnostic requires target as last VBO object"
        renderer._gl.glDrawArraysInstanced = lambda mode, start, size, batch: draw(mode, 0, first, batch)
    try:
        yield
    finally:
        renderer._gl.glDrawArraysInstanced = draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=("stages","objects","variants","counts"), default=["stages","objects","variants","counts"])
    parser.add_argument("--variants", nargs="+", default=["baseline","cheap_fragment","rasterizer_discard","fp32_gs","fp32_fs","fp32_both","clip_fast_gs","clip_fast_fs","clip_fast_both"])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    record = {"status":"running", "batch":args.batch, "map":str(args.map),
              "groups":args.groups, "scope":"All timings use native 8x8 cells, original scene VBO and CUDA poses. No subrays, high-resolution rendering, mesh simplification or production changes.",
              "source_hashes":{str(p.relative_to(ROOT)):digest(p) for p in (
                  ROOT/"devtools/rendering.py", ROOT/"devtools/_zone_shaders.py", ROOT/"devtools/_gl_cuda.py",
                  *sorted(Path(__file__).parent.glob("*.py")))},
              "map_sha256":digest(args.map), "torch":torch.__version__, "static_analysis":STATIC_ANALYSIS}

    def save(stage):
        record["last_stage"] = stage
        args.output.write_text(json.dumps(record,indent=2)+"\n")
        print(stage, flush=True)

    config = load_config(ROOT/"configs/devtools.json")
    poses_cpu = make_poses(args.batch)
    scene = Scene.from_obj(args.map, config, poses_cpu[0], TARGET_POSITION)
    snapshot = scene.get_geometry(exclude_ids={"vehicle"})
    with BatchRenderer(snapshot) as renderer:
        poses = poses_to_tensor(poses_cpu, device=renderer.device)
        depth = config.sensors.tof
        record["gpu"] = torch.cuda.get_device_name()
        record["geometry"] = renderer.geometry_info
        record["object_draw_ranges"] = renderer._object_ranges
        record["pose_sha256"] = hashlib.sha256(np.asarray([p.as_transform().matrix for p in poses_cpu],dtype=np.float32).tobytes()).hexdigest()
        save("warmup")
        baseline = renderer.render(poses, depth, sampling="zone")
        torch.cuda.synchronize()
        baseline_program, baseline_uniforms = renderer._zone_program, renderer._zone_uniforms
        action = lambda: renderer.render(poses, depth, sampling="zone")
        if "stages" in args.groups:
            save("stages")
            record["stages"] = profile(renderer, poses, depth, warmup=3, repeats=args.repeats)
            save("stages_done")
        if "objects" in args.groups:
            record["objects"] = {}
            for selection in ("full", "map", "target"):
                save("objects_"+selection)
                with object_selection(renderer, selection):
                    call = (lambda: renderer.render(poses, depth, only_object="target", sampling="zone")) if selection=="target" else action
                    record["objects"][selection] = {"wall":wall(call,args.repeats), "draw":draw_probe(renderer,call,repeats=3),
                                                       "scope":"Selected-object workload diagnostic; not an equivalent full-scene optimization"}
        if "variants" in args.groups:
            record["variants"] = {}
            for name in args.variants:
                save("variant_"+name)
                variant = get_variants()[name]
                if name == "baseline":
                    renderer._zone_program, renderer._zone_uniforms = baseline_program, baseline_uniforms
                    program = None
                else:
                    program = install(renderer, {key:getattr(variant,key) for key in ("vertex_shader","geometry_shader","fragment_shader")})
                if variant.rasterizer_discard:
                    renderer._gl.glEnable(0x8C89)
                try:
                    raw = action(); torch.cuda.synchronize()
                    parity = compare(raw,baseline)
                    del raw
                    record["variants"][name] = {"metadata":variant.metadata(), "output_comparison":parity,
                        "wall":wall(action,args.repeats), "draw":draw_probe(renderer,action,repeats=3)}
                finally:
                    if variant.rasterizer_discard:
                        renderer._gl.glDisable(0x8C89)
                    renderer._zone_program, renderer._zone_uniforms = baseline_program, baseline_uniforms
                    if program:
                        program.release()
                save("variant_"+name+"_done")
        if "counts" in args.groups:
            save("counts")
            program = install(renderer,counters.sources())
            storage = renderer._ctx.buffer(bytes(16*4))
            storage.bind_to_storage_buffer(2)
            program["profile_target_id"].value = renderer.object_ids.index("target")+1
            try:
                raw = action(); torch.cuda.synchronize(); renderer._ctx.finish()
                record["counts"] = counters.decode(storage.read())
                record["counts"]["output_comparison"] = compare(raw,baseline)
                assert record["counts"]["output_comparison"]["exact"]
                assert record["counts"]["all"]["gs_input_triangles"] == renderer.triangle_count*args.batch
            finally:
                renderer._zone_program, renderer._zone_uniforms = baseline_program, baseline_uniforms
                storage.release(); program.release()
        record["status"] = "ok"
        save("complete")


if __name__ == "__main__":
    main()
