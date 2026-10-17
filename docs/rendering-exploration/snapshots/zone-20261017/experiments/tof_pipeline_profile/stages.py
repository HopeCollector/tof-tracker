"""Experimental decomposition of the unchanged formal zone renderer.

Only ``profile`` is the caller-facing entry point. Run it in Docker, on the
renderer's owning thread, with CUDA poses already constructed. The native
baseline has no hooks. The decomposition deliberately drains each stage and
therefore measures isolated costs rather than an additive native timeline.
No production module, shader, geometry, resolution, or sampling is changed.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import fields
import math
import statistics
import time

import numpy as np
import torch

from devtools.rendering import RasterBatch, _validate_render_config
from devtools._gl_cuda import _library


_SSBO, _PBO, _FRAMEBUFFER = 0x90D2, 0x88EB, 0x8D40
_TIME_ELAPSED, _QUERY_RESULT, _QUERY_COUNTER_BITS = 0x88BF, 0x8866, 0x8864
_STATISTICS = {
    "vertices_submitted": 0x82EE,
    "primitives_submitted": 0x82EF,
    "vertex_shader_invocations": 0x82F0,
    "geometry_shader_invocations": 0x887F,
    "geometry_shader_primitives_emitted": 0x82F3,
    "fragment_shader_invocations": 0x82F4,
    "clipping_input_primitives": 0x82F6,
    "clipping_output_primitives": 0x82F7,
}


def _summary(samples):
    ordered = sorted(samples)
    if not ordered:
        return None
    return {"median": statistics.median(ordered),
            "p95": float(np.percentile(ordered, 95)),
            "min": ordered[0], "max": ordered[-1], "samples": list(samples)}


def _milliseconds(started):
    return (time.perf_counter() - started) * 1000


class _Queries:
    """Typed query functions; no changes to the production GL facade."""

    def __init__(self):
        self.library = _library(["OpenGL", "libOpenGL.so.0", "GL", "libGL.so.1"])
        specifications = {
            "glGenQueries": (None, C.c_int, C.POINTER(C.c_uint)),
            "glDeleteQueries": (None, C.c_int, C.POINTER(C.c_uint)),
            "glBeginQuery": (None, C.c_uint, C.c_uint),
            "glEndQuery": (None, C.c_uint),
            "glGetQueryObjectui64v": (None, C.c_uint, C.c_uint,
                                     C.POINTER(C.c_uint64)),
            "glGetQueryiv": (None, C.c_uint, C.c_uint, C.POINTER(C.c_int)),
        }
        for name, (restype, *argtypes) in specifications.items():
            function = getattr(self.library, name)
            function.restype, function.argtypes = restype, argtypes
            setattr(self, name, function)

    def create(self):
        result = C.c_uint()
        self.glGenQueries(1, C.byref(result))
        return result.value

    def delete(self, value):
        self.glDeleteQueries(1, C.byref(C.c_uint(value)))

    def result(self, value):
        result = C.c_uint64()
        self.glGetQueryObjectui64v(value, _QUERY_RESULT, C.byref(result))
        return result.value

    def bits(self, target):
        result = C.c_int()
        self.glGetQueryiv(target, _QUERY_COUNTER_BITS, C.byref(result))
        return result.value


def _exact(actual, expected):
    """Download only for validation, outside any measured interval."""
    report = {}
    for field in fields(RasterBatch):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                passed = False
            else:
                a, b = left.detach().cpu(), right.detach().cpu()
                passed = a.shape == b.shape and a.dtype == b.dtype
                if passed:
                    passed = (bool(torch.allclose(a, b, rtol=0, atol=0, equal_nan=True))
                              if a.is_floating_point() else bool(torch.equal(a, b)))
            report[field.name] = bool(passed)
        else:
            report[field.name] = left == right
    if not all(report.values()):
        raise AssertionError(f"staged render disagrees with native render: {report}")
    return {"passed": True, "comparison": "exact values, dtypes, shapes and metadata",
            "fields": report}


def _gl_stage(renderer, queries, function):
    """No GL/CUDA work from the preceding isolated stage remains pending."""
    query = queries.create()
    queries.glBeginQuery(_TIME_ELAPSED, query)
    started = time.perf_counter()
    try:
        value = function()
        submitted = _milliseconds(started)
    finally:
        queries.glEndQuery(_TIME_ELAPSED)
    renderer._ctx.finish()
    complete = _milliseconds(started)
    try:
        elapsed = queries.result(query) / 1e6
    finally:
        queries.delete(query)
    return value, {"cpu_submit_ms": submitted, "wall_complete_ms": complete,
                   "gl_gpu_ms": elapsed}


def _cuda_stage(renderer, function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    started = time.perf_counter()
    value = function()
    submitted = _milliseconds(started)
    end.record()
    end.synchronize()
    return value, {"cpu_submit_ms": submitted, "wall_complete_ms": _milliseconds(started),
                   "cuda_event_ms": start.elapsed_time(end)}


def _interop_stage(renderer, resource, pointer_destination, pointer_source, byte_count,
                   stream, *, mapped_is_destination):
    """Group completion plus API host costs and an event around the D2D copy.

    The CUDA event copy interval can include host enqueue gaps. It is neither
    a bus bandwidth measurement nor the GPU duration of map/unmap operations.
    """
    cuda = renderer._cuda
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    overall = time.perf_counter()
    started = time.perf_counter()
    pointer, size = cuda.map(resource, stream)
    map_host = _milliseconds(started)
    try:
        if size < byte_count:
            raise RuntimeError("mapped interop buffer is smaller than the requested copy")
        destination = pointer if mapped_is_destination else pointer_destination
        source = pointer_source if mapped_is_destination else pointer
        start.record()
        started = time.perf_counter()
        cuda.copy(destination, source, byte_count, stream)
        copy_host = _milliseconds(started)
        end.record()
    finally:
        started = time.perf_counter()
        cuda.unmap(resource, stream)
        unmap_host = _milliseconds(started)
    submitted = _milliseconds(overall)
    torch.cuda.synchronize(renderer.device)
    return {"cpu_submit_ms": submitted, "wall_complete_ms": _milliseconds(overall),
            "map_host_ms": map_host, "copy_submit_host_ms": copy_host,
            "unmap_host_ms": unmap_host,
            "copy_cuda_event_ms": start.elapsed_time(end), "bytes": byte_count}


@torch.no_grad()
def _staged_render(renderer, poses, config, queries):
    """Line-for-line equivalent zone path, with isolated stage completion."""
    renderer._require_open()
    _validate_render_config(config)
    renderer._validate_sampling("zone")
    if (not isinstance(poses, torch.Tensor) or poses.device != renderer.device
            or poses.dtype != torch.float32 or poses.ndim != 3
            or tuple(poses.shape[1:]) != (4, 4)):
        raise ValueError("poses must be float32 CUDA [B,4,4] on renderer.device")
    batch = len(poses)
    if batch != renderer._batch_size:
        renderer.prepare(batch)
    program, uniforms = renderer._pipeline("zone")
    target = renderer._target(config, "zone")
    gl = renderer._gl
    gl.activate()
    stream = torch.cuda.current_stream(renderer.device).cuda_stream
    stages = {}

    def pack():
        maximum = poses[:, :3, 3].to(torch.float64).abs().amax(dim=1)
        maximum = maximum.clamp_min(max(1., config.max_range_m))
        roundoff = 32 * np.finfo(np.float32).eps * maximum
        guard = torch.maximum(roundoff * 4,
                              torch.full_like(roundoff, config.max_range_m * .001))
        radius = (config.max_range_m + guard).to(torch.float32)
        return torch.cat((poses.reshape(batch, 16), radius[:, None]), dim=1).contiguous()

    packed, stages["pose_pack_cuda"] = _cuda_stage(renderer, pack)
    stages["pose_cuda_to_gl"] = _interop_stage(
        renderer, renderer._pose_resource, None, packed.data_ptr(),
        packed.numel() * packed.element_size(), stream, mapped_is_destination=True)

    def clear():
        gl.glBindFramebuffer(_FRAMEBUFFER, target["fbo"])
        gl.glClearBufferfv(0x1800, 0, (C.c_float * 4)(0, 0, 0, 0))
        gl.glClearBufferfv(0x1800, 1, (C.c_float * 4)(0, 0, 0, 0))
        gl.glClearBufferfv(0x1801, 0, (C.c_float * 1)(1))

    _, stages["gl_clear"] = _gl_stage(renderer, queries, clear)

    def state():
        gl.glViewport(0, 0, config.width, config.height)
        gl.glUseProgram(program.glo)
        gl.glBindVertexArray(renderer._vao)
        gl.glBindBufferBase(_SSBO, 0, renderer._pose_buffer)
        gl.glBindBufferBase(_SSBO, 1, renderer._object_buffer)
        gl.glUniform2f(uniforms["tangent_fov"],
                      math.tan(math.radians(config.hfov_deg) / 2),
                      math.tan(math.radians(config.vfov_deg) / 2))
        gl.glUniform2f(uniforms["image_size"], config.width, config.height)

    _, stages["gl_draw_state"] = _gl_stage(renderer, queries, state)
    _, stages["gl_draw"] = _gl_stage(renderer, queries,
        lambda: gl.glDrawArraysInstanced(0x0004, 0, renderer.vertex_count, batch))
    renderer._draw_count += 1
    renderer._render_calls += 1

    def transfer():
        gl.glBindBuffer(_PBO, target["pbo"])
        gl.glGetTextureImage(target["color"], 0, 0x8227, 0x1406,
                             target["color_bytes"], C.c_void_p(0))
        gl.glGetTextureImage(target["hit"], 0, 0x1908, 0x1406,
                             target["hit_bytes"], C.c_void_p(target["color_bytes"]))
        gl.glBindBuffer(_PBO, 0)
        gl.glFlush()

    _, stages["texture_to_pbo"] = _gl_stage(renderer, queries, transfer)
    started = time.perf_counter()
    raw = torch.empty((target["bytes"] // 4,), dtype=torch.float32, device=renderer.device)
    stages["output_allocate"] = {"cpu_submit_ms": _milliseconds(started)}
    stages["pbo_to_cuda"] = _interop_stage(
        renderer, target["resource"], raw.data_ptr(), None, target["bytes"],
        stream, mapped_is_destination=False)
    gl.check("profiled zone render and whole-array GPU transfer")
    renderer._last_output_bytes = target["bytes"]

    def format_output():
        image = raw[:target["pixels"] * 2].view(batch, config.height, config.width, 2)
        distance = image[..., 0].flip((1,)).contiguous()
        identity = image[..., 1].flip((1,)).to(torch.int64) - 1
        hit = raw[target["pixels"] * 2:].view(batch, config.height, config.width, 4)
        point = hit[..., :3].flip((1,)).contiguous()
        triangle = hit[..., 3].flip((1,)).to(torch.int64) - 1
        return RasterBatch(distance, identity, renderer.object_ids, renderer._scene_version,
                           renderer._snapshot.t_ref, renderer._snapshot.map_bounds_m,
                           point, triangle)

    output, stages["torch_output_format"] = _cuda_stage(renderer, format_output)
    return output, stages


def _statistics_probe(renderer, action, queries):
    """Separate native draw counter pass; no timer results use these queries."""
    if "GL_ARB_pipeline_statistics_query" not in renderer._ctx.extensions:
        return {"supported": False, "reason": "GL_ARB_pipeline_statistics_query absent"}
    counter_bits = {name: queries.bits(target) for name, target in _STATISTICS.items()}
    active = {name: (target, queries.create()) for name, target in _STATISTICS.items()
              if counter_bits[name] > 0}
    gl = renderer._gl
    original = gl.glDrawArraysInstanced
    calls = 0

    def counted(*args):
        nonlocal calls
        for target, query in active.values():
            queries.glBeginQuery(target, query)
        try:
            return original(*args)
        finally:
            for target, _ in active.values():
                queries.glEndQuery(target)
            calls += 1

    try:
        gl.glDrawArraysInstanced = counted
        output = action()
        torch.cuda.synchronize(renderer.device)
        result = {name: queries.result(query) for name, (_, query) in active.items()}
        gl.check("draw-only pipeline statistics")
    finally:
        gl.glDrawArraysInstanced = original
        for _, query in active.values():
            queries.delete(query)
    return {"supported": True, "draw_calls": calls, "counter_bits": counter_bits,
            "counts": result,
            "interpretation": "Hardware invocation counts can include helper/replayed work; "
                              "they are not an exact count of distinct intersecting cells.",
            "specification": "https://registry.khronos.org/OpenGL/extensions/ARB/"
                             "ARB_pipeline_statistics_query.txt"}, output


def draw_probe(renderer, action, repeats=3):
    """Time all draws in ``action`` and count them in a separate native pass.

    ``action`` normally calls ``renderer.render`` and can select any object or
    use an experimental shader installed by the caller. Returns only JSON
    values. Per-draw GL timers do not include a finish inside the draw; query
    retrieval occurs after the entire action completes. Statistics use another
    pass, separate from the GPU timing samples. No output equality claim is
    made here because shader ablations intentionally change their outputs.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    with torch.cuda.device(renderer.device), torch.no_grad():
        renderer._require_open()
        renderer._gl.activate()
        gl = renderer._gl
        queries = _Queries()
        original = gl.glDrawArraysInstanced
        timed_draws, action_wall = [], []
        pending = []

        def timed(*args):
            query = queries.create()
            pending.append(query)
            queries.glBeginQuery(_TIME_ELAPSED, query)
            try:
                return original(*args)
            finally:
                queries.glEndQuery(_TIME_ELAPSED)

        try:
            gl.glDrawArraysInstanced = timed
            for _ in range(repeats):
                torch.cuda.synchronize(renderer.device)
                renderer._ctx.finish()
                started = time.perf_counter()
                output = action()
                torch.cuda.synchronize(renderer.device)
                renderer._ctx.finish()
                action_wall.append(_milliseconds(started))
                timed_draws.append([queries.result(query) / 1e6 for query in pending])
                for query in pending:
                    queries.delete(query)
                pending.clear()
                del output
            gl.check("draw-only elapsed probes")
        finally:
            gl.glDrawArraysInstanced = original
            for query in pending:
                queries.delete(query)

        counts = _statistics_probe(renderer, action, queries)
        counters = counts[0] if isinstance(counts, tuple) else counts
        draw_counts = [len(sample) for sample in timed_draws]
        per_index = ([[_sample[index] for _sample in timed_draws]
                      for index in range(draw_counts[0])]
                     if len(set(draw_counts)) == 1 else [])
        return {
            "repeats": repeats, "draw_calls_per_action": draw_counts,
            "draw_gpu_sum_ms": _summary([sum(sample) for sample in timed_draws]),
            "draw_gpu_ms_by_index": [_summary(sample) for sample in per_index],
            "instrumented_action_wall_ms": _summary(action_wall),
            "pipeline_statistics": counters,
            "measurement_notes": [
                "GL TIME_ELAPSED queries surround only glDrawArraysInstanced.",
                "No finish or CUDA synchronization is inserted inside each timed draw.",
                "The action wall includes timer enqueue and final GL/CUDA synchronization; "
                "it is not the uninstrumented production baseline.",
                "Pipeline statistics use a separate untimed action and can include helpers.",
            ],
        }


def profile(renderer, poses, config, warmup=3, repeats=7):
    """Return JSON-serializable native baseline, isolated stages and counters.

    All counters/probes run sequentially on the owning thread. GPU inputs and
    outputs stay resident during timing; validation downloads are afterwards.
    This function profiles only the whole-scene zone path, with no postprocess.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    with torch.cuda.device(renderer.device), torch.no_grad():
        renderer._require_open()
        renderer._gl.activate()
        renderer.prepare(len(poses), config=config, sampling="zone")
        queries = _Queries()
        for _ in range(warmup):
            output = renderer.render(poses, config, sampling="zone")
            torch.cuda.synchronize(renderer.device)
            del output
        native_samples = []
        for _ in range(repeats):
            torch.cuda.synchronize(renderer.device)
            started = time.perf_counter()
            reference = renderer.render(poses, config, sampling="zone")
            torch.cuda.synchronize(renderer.device)
            native_samples.append(_milliseconds(started))

        # Allocate/initialize all event, query and output paths before sampling.
        renderer._ctx.finish()
        output, _ = _staged_render(renderer, poses, config, queries)
        parity = {"staged": _exact(output, reference)}
        all_stages, isolated_samples = {}, []
        for _ in range(repeats):
            torch.cuda.synchronize(renderer.device)
            renderer._ctx.finish()
            started = time.perf_counter()
            output, stages = _staged_render(renderer, poses, config, queries)
            isolated_samples.append(_milliseconds(started))
            for name, metrics in stages.items():
                target = all_stages.setdefault(name, {})
                for metric, value in metrics.items():
                    target.setdefault(metric, []).append(value)
        parity["last_staged"] = _exact(output, reference)
        probed = _statistics_probe(
            renderer, lambda: renderer.render(poses, config, sampling="zone"), queries)
        if isinstance(probed, tuple):
            counters, statistics_output = probed
            parity["statistics_probe"] = _exact(statistics_output, reference)
        else:
            counters = probed
        return {
            "scope": "ToF raw zone render; pre-existing CUDA poses to owned CUDA RasterBatch",
            "batch": len(poses), "resolution": [config.width, config.height],
            "warmup": warmup, "repeats": repeats,
            "native_hot_render_wall_ms": _summary(native_samples),
            "isolated_profile_wall_ms": _summary(isolated_samples),
            "stages": {name: {metric: (_summary(values) if metric != "bytes" else values[0])
                              for metric, values in metrics.items()}
                       for name, metrics in all_stages.items()},
            "pipeline_statistics": counters, "exact_output_parity": parity,
            "measurement_notes": [
                "Native hot baseline calls unchanged BatchRenderer.render without hooks.",
                "Isolated stages explicitly finish GL or synchronize CUDA at each boundary; "
                "their times must not be added to reconstruct native scheduling.",
                "CPU submit measures API enqueue/host wait time. A CUDA interop map can block.",
                "GL elapsed queries measure GPU command spans; host-only draw state need not "
                "have meaningful GPU cost. Query collection is outside recorded stage times.",
                "CUDA events around tiny operations can include host enqueue gaps; copy event "
                "duration is not a pure device memcpy bandwidth benchmark.",
                "Pipeline statistics are collected in a separate untimed native render; "
                "no statistics query overlaps timed draw measurements.",
                "Pose construction/mounts, scene preparation, shader compilation, sensor "
                "postprocessing and validation downloads are outside this profile.",
                "No sampling, geometry, precision, occlusion, output or resolution change.",
            ],
        }
