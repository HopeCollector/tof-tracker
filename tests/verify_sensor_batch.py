"""Docker/GPU contracts for immediate sensors and explicit shared batches.

Numerical expectations come from the existing independent float64 geometry
oracle. Collection, ownership and failure checks exercise the public API;
the development consumer is used only for same-query diagnostic boundaries.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import gc
import json
import math
from pathlib import Path
import sys
import traceback
from unittest.mock import patch
import weakref

import numpy as np
import torch

# Review fixtures are source tools, not part of the installed sensor API.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from devtools.config import DepthConfig, SamplingRuntimeConfig
from devtools.sensors import CameraSensor, RayRenderer, SensorSampler, ToFSensor
from devtools.sensors._ops import SampleBatch, poses_to_tensor
from devtools.scene import Pose
from tools.sensor_review.fixtures.capture import capture_sample
from verify_ray_sensors import (box, close, compare_tensor_fields, cpu_selection,
                                ground, host, independent_rays, oracle,
                                rejected, snapshot, sphere, tensor_fields)

TOF = DepthConfig(4, 3, 45., 45., .015, 4., samples_per_axis=4)
CAMERA = DepthConfig(5, 4, 65., 50., .015, 6., samples_per_axis=2)
RUNTIME = SamplingRuntimeConfig(max_gpu_memory_fraction=.5)
TOF_QUANTILE = .2
TOF_SIGNAL_FLOOR = .08


def source_scene(target=(3., .1, 2.)):
    return snapshot(("ground", ground(half=30.)),
                    ("wall", box("wall", ((1.5, 1.6), (.3, .6), (.5, 3.)))),
                    ("target", sphere(target, .4)),
                    bounds=((-30., 30.), (-30., 30.), (-30., 30.)))


def poses(count=3):
    return tuple(Pose((-.1 + .07 * k, .025 * k, 2.01),
                      (0., 0., math.sin(.01 * k), math.cos(.01 * k))) for k in range(count))


@contextmanager
def no_readback():
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("sampling read back to CPU")), \
         patch.object(torch.Tensor, "numpy", side_effect=AssertionError("sampling converted to NumPy")), \
         patch.object(torch.Tensor, "item", side_effect=AssertionError("sampling read back a scalar")):
        yield


def expected_ranges(source, views, config, *, quantile=None):
    origins, directions = independent_rays(views, config)
    distance, identity, _ = oracle(source, origins, directions, config.max_range_m)
    shape = (len(views), config.height, config.width, config.samples_per_axis ** 2)
    distance, identity = distance.reshape(shape), identity.reshape(shape)
    selected = np.empty(shape[:-1], dtype=np.int64)
    for index in np.ndindex(shape[:-1]):
        selected[index] = (cpu_selection(distance[index], identity[index], config.max_range_m,
                                         quantile=quantile, floor=TOF_SIGNAL_FLOOR)
                           if quantile is not None else
                           np.argmin(np.where(identity[index] >= 0, distance[index], config.max_range_m)))
    chosen = np.take_along_axis(distance, selected[..., None], axis=-1)[..., 0]
    chosen_id = np.take_along_axis(identity, selected[..., None], axis=-1)[..., 0]
    return np.where(chosen_id < 0, np.inf,
                    np.where(chosen < config.min_range_m, np.nan, chosen))


def compact(result, count, config):
    assert result.range_m.shape == (count, config.height, config.width)
    assert result.range_m.dtype == torch.float32 and result.range_m.is_cuda
    assert result.raw is result.subray_range_m is None
    assert result.object_index is result.hit_point_sensor_m is result.triangle_index is None
    assert not hasattr(result, "target_mask") and not hasattr(result, "target_hint")


def verify_inputs_and_oracle():
    source, views = source_scene(), poses()
    rejected(lambda: DepthConfig(4, 3, 45., 45., .015, 4.), (TypeError,))
    rejected(SamplingRuntimeConfig, (TypeError,))
    with RayRenderer(source) as renderer:
        rejected(lambda: SensorSampler(renderer), (TypeError,))
        rejected(lambda: SensorSampler(renderer, runtime=None), (TypeError,))
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        for options in ({}, {"quantile": TOF_QUANTILE}, {"signal_floor": TOF_SIGNAL_FLOOR}):
            rejected(lambda options=options: ToFSensor(sampler, TOF, **options), (TypeError,))
        camera = CameraSensor(sampler, CAMERA)
        tof = ToFSensor(sampler, TOF, quantile=TOF_QUANTILE, signal_floor=TOF_SIGNAL_FLOOR)
        matrices = poses_to_tensor(views)
        for sensor, config, quantile in ((camera, CAMERA, None), (tof, TOF, TOF_QUANTILE)):
            expected = expected_ranges(source, views, config, quantile=quantile)
            for value, count in ((views[0], 1), (views, 3), (matrices[0], 1), (matrices, 3)):
                with no_readback():
                    result = sensor.sample(value)
                compact(result, count, config)
                close(host(result.range_m), expected[:count])
            rejected(lambda: sensor.sample([]))
            rejected(lambda: sensor.sample(torch.eye(4)))
            rejected(lambda: sensor.sample(torch.empty((2, 4, 3), device=renderer.device)))
            rejected(lambda: sensor.sample(torch.eye(4, dtype=torch.float64, device=renderer.device)))
        zero = ToFSensor(sampler, TOF, quantile=0., signal_floor=TOF_SIGNAL_FLOOR).sample(views)
        close(host(zero.range_m), expected_ranges(source, views, TOF, quantile=0.))
        for invalid in (-.01, 1.01, math.nan, True):
            rejected(lambda invalid=invalid: ToFSensor(
                sampler, TOF, quantile=invalid, signal_floor=TOF_SIGNAL_FLOOR))
        for invalid in (0., -.1, math.nan, True):
            rejected(lambda invalid=invalid: ToFSensor(
                sampler, TOF, quantile=TOF_QUANTILE, signal_floor=invalid))
        rejected(lambda: setattr(CAMERA, "width", 99), (FrozenInstanceError, AttributeError))
        runtime = SamplingRuntimeConfig(max_gpu_memory_fraction=.25)
        custom = CameraSensor(SensorSampler(renderer, runtime=runtime), CAMERA).sample(views[0])
        planner = custom.execution_info["planner"]
        assert planner["max_gpu_memory_fraction"] == runtime.max_gpu_memory_fraction
        assert planner["sampling_workset_limit_bytes"] == int(planner["cuda_total_bytes"] * .25)

    half_cell = snapshot(("wall", box("wall", ((1., 1.1), (-1., 0.), (1., 3.)))),
                         bounds=((-30., 30.), (-30., 30.), (-30., 30.)))
    config = DepthConfig(1, 1, 45., 45., .015, 4., samples_per_axis=2)
    pose = Pose((0., 0., 2.))
    with RayRenderer(half_cell) as renderer:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        near = ToFSensor(sampler, config, quantile=.8, signal_floor=TOF_SIGNAL_FLOOR).sample(pose)
        high_floor = ToFSensor(sampler, config, quantile=.8, signal_floor=10.).sample(pose)
        final_quantile = ToFSensor(sampler, config, quantile=1., signal_floor=TOF_SIGNAL_FLOOR).sample(pose)
        # Two of four rays cross x=1; the other two have a known empty full range.
        expected_near = math.sqrt(1 + 2 * (.5 * math.tan(math.radians(45.) / 2)) ** 2)
        close(host(near.range_m), np.full((1, 1, 1), expected_near))
        assert np.isposinf(host(high_floor.range_m)).all()
        assert np.isposinf(host(final_quantile.range_m)).all()
    return {"world_pose_forms": ["Pose", "sequence", "CUDA44", "CUDAB44"],
            "independent_float64_oracle": "passed", "quantile_zero": "passed",
            "required_public_parameters": "omissions rejected",
            "explicit_runtime_fraction": runtime.max_gpu_memory_fraction,
            "quantile_and_signal_floor_change_public_ranges": "passed"}


def verify_interleaved_freezing_and_grouping():
    source, views = source_scene(), poses()
    with RayRenderer(source) as renderer:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        camera_a, camera_b = CameraSensor(sampler, CAMERA), CameraSensor(sampler, CAMERA)
        camera_other = CameraSensor(sampler, DepthConfig(3, 2, 55., 40., .015, 5., samples_per_axis=2))
        tof_a = ToFSensor(sampler, TOF, quantile=TOF_QUANTILE, signal_floor=TOF_SIGNAL_FLOOR)
        tof_b = ToFSensor(sampler, TOF, quantile=TOF_QUANTILE, signal_floor=TOF_SIGNAL_FLOOR)
        tof_other = ToFSensor(sampler, TOF, quantile=.8, signal_floor=TOF_SIGNAL_FLOOR)
        mutable_matrix = poses_to_tensor(views)
        mutable_list = list(views[:2])
        requests = ((camera_a, mutable_matrix), (tof_a, mutable_list),
                    (camera_b, views[1]), (camera_other, views[:2]),
                    (tof_b, views[2]), (tof_other, views[:2]))
        baselines = [sensor.sample(value) for sensor, value in requests]
        batch = sampler.batch()
        before = renderer.geometry_info["trace_calls"]
        handles = [sensor.sample(value, batch=batch) for sensor, value in requests]
        assert renderer.geometry_info["trace_calls"] == before
        for handle in handles:
            rejected(handle.result, (RuntimeError,))
        assert renderer.geometry_info["trace_calls"] == before, "pending results must not execute"
        mutable_matrix[:, :3, 3] += 10.
        mutable_list[:] = [poses(1)[0]]
        with no_readback():
            batch.execute()
        assert renderer.geometry_info["trace_calls"] - before == 1
        for baseline, handle in zip(baselines, handles):
            compare_tensor_fields(baseline, handle.result())
            assert handle.result().range_m is handle.result().range_m
        info = handles[0].result().execution_info
        assert info["request_count"] == 6
        assert info["postprocess_calls"] == 4, info
        rejected(batch.execute, (RuntimeError,))
        rejected(lambda: camera_a.sample(views[0], batch=batch), (RuntimeError,))
        foreign = SensorSampler(renderer, runtime=RUNTIME).batch()
        rejected(lambda: camera_a.sample(views[0], batch=foreign), (ValueError, RuntimeError))
        return {"interleaved_requests": 6, "unique_reduction_groups": 4,
                "full_scene_queries": 1, "frozen_GPU_and_sequence_inputs": "passed"}


def verify_epoch_failure_and_execution_guards():
    source, views = source_scene(), poses()
    with RayRenderer(source) as renderer:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        sensor = CameraSensor(sampler, CAMERA)
        batch = sampler.batch()
        stale = sensor.sample(views, batch=batch)
        renderer.sync(source_scene((3., 1., 2.)))
        before = renderer.geometry_info["trace_calls"]
        rejected(batch.execute, (RuntimeError,))
        rejected(stale.result, (RuntimeError,))
        assert renderer.geometry_info["trace_calls"] == before
        fresh = sensor.sample(views)
        assert fresh.scene_version == renderer.scene_version
        renderer.sync(source)
        back = sensor.sample(views)
        close(host(back.range_m), expected_ranges(source, views, CAMERA))
        assert renderer.geometry_info["mesh_bvh_builds"] == 1
        assert renderer.geometry_info["geometry_uploads"] == 1
        protected = []

        def guarded(depth, raw, frozen, start, stop):
            rejected(lambda: renderer.sync(source), (RuntimeError,))
            rejected(renderer.close, (RuntimeError,))
            protected.append((start, stop))

        running = sampler.batch()
        good = capture_sample(sensor, views, batch=running, diagnostics=False, consumer=guarded)
        running.execute()
        assert protected == [(0, len(views))]
        compare_tensor_fields(back, good.result())

        def broken(*args):
            raise ValueError("deliberate development consumer failure")

        failed = sampler.batch()
        first = sensor.sample(views[:1], batch=failed)
        second = capture_sample(sensor, views[1:], batch=failed, diagnostics=False, consumer=broken)
        rejected(failed.execute, (ValueError,))
        for handle in (first, second):
            try:
                handle.result()
            except RuntimeError as error:
                assert error.__cause__ is not None
                assert error.__cause__.__traceback__ is None, "failed handle retains execution tensor locals"
                assert error.__cause__.__context__ is None
            else:
                raise AssertionError("failed handle returned a result")
        rejected(failed.execute, (RuntimeError,))
        # Failures must release execution guards so the renderer remains usable.
        renderer.sync(source)
        compare_tensor_fields(back, sensor.sample(views))
    return {"scene_epoch_rejection_before_trace": "passed", "dynamic_A_B_A": "passed",
            "consumer_failure_fails_all_handles": "passed", "sync_close_during_execute": "rejected"}


def verify_chunk_budget_consumers_and_diagnostics():
    from devtools.sensors import _sampling as module
    source, views = source_scene(), poses(5)
    with RayRenderer(source) as renderer:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        camera = CameraSensor(sampler, CAMERA)
        tof = ToFSensor(sampler, TOF, quantile=TOF_QUANTILE, signal_floor=TOF_SIGNAL_FLOOR)
        requests = ((tof, views), (camera, views[:3]), (tof, views[-1]))

        def run():
            batch, consumed, handles = sampler.batch(), [[], [], []], []
            for index, (sensor, value) in enumerate(requests):
                def consume(depth, raw, frozen, start, stop, index=index):
                    assert stop - start == len(frozen) == len(depth.range_m)
                    assert raw.subray_distance_m.is_cuda
                    consumed[index].append((start, stop))
                handles.append(capture_sample(sensor, value, batch=batch, consumer=consume))
            before = renderer.geometry_info["trace_calls"]
            with no_readback():
                batch.execute()
            values = [handle.result() for handle in handles]
            assert values[0].execution_info["chunks"] == renderer.geometry_info["trace_calls"] - before
            for intervals, (_, value) in zip(consumed, requests):
                covered = [row for start, stop in intervals for row in range(start, stop)]
                count = 1 if isinstance(value, Pose) else len(value)
                assert covered == list(range(count))
            return values, consumed

        whole, _ = run()
        original = module._plan_chunks
        capacity = 2 * TOF.width * TOF.height * TOF.samples_per_axis ** 2

        def bounded(*args, **kwargs):
            count, info = original(*args, **kwargs)
            count = min(count, capacity)
            return count, {**info, "chunk_items": count}

        with patch.object(module, "_plan_chunks", side_effect=bounded):
            chunks, intervals = run()
        assert chunks[0].execution_info["chunks"] > 1
        for a, b in zip(whole, chunks):
            compare_tensor_fields(a, b)
        before = renderer.geometry_info["trace_calls"]
        tiny_runtime = SamplingRuntimeConfig(max_gpu_memory_fraction=1e-9)
        tiny = CameraSensor(SensorSampler(renderer, runtime=tiny_runtime), CAMERA)
        rejected(lambda: tiny.sample(views), (MemoryError,))
        assert renderer.geometry_info["trace_calls"] == before
        return {"ray_capacity": capacity, "chunks": chunks[0].execution_info["chunks"],
                "consumer_intervals": intervals, "all_diagnostic_fields_parity": "passed",
                "tiny_budget_rejects_before_trace": "passed"}


def verify_fixture_query_inputs_across_chunks():
    from devtools.sensors import _sampling as module
    source, views = source_scene(), poses(5)
    rays_per_view = CAMERA.width * CAMERA.height * CAMERA.samples_per_axis ** 2
    with RayRenderer(source) as renderer:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        sensor = CameraSensor(sampler, CAMERA)
        batch = sampler.batch()
        first = capture_sample(sensor, views[:1], batch=batch)
        ordinary = sensor.sample(views[1:2], batch=batch)
        tail = capture_sample(sensor, views[2:], batch=batch)
        queries = []
        trace, plan = renderer.trace, module._plan_chunks

        def observe_trace(origins, directions, *args, **kwargs):
            queries.append(directions.clone())
            return trace(origins, directions, *args, **kwargs)

        def two_views(*args, **kwargs):
            capacity, info = plan(*args, **kwargs)
            capacity = min(capacity, 2 * rays_per_view)
            return capacity, {**info, "chunk_items": capacity}

        original_process, original_concat = module._process_samples, module._concat_depth
        with patch.object(renderer, "trace", side_effect=observe_trace), \
             patch.object(module, "_plan_chunks", side_effect=two_views), no_readback():
            batch.execute()
        assert module._process_samples is original_process
        assert module._concat_depth is original_concat
        assert "execute" not in batch.__dict__, "completed fixture batch retains execution observer"
        assert len(queries) == 3
        assert batch.execution_info["postprocess_calls"] == 3, "equal policies were not coalesced"
        compact(ordinary.result(), 1, CAMERA)
        actual_inputs = torch.cat(queries).reshape(
            len(views), CAMERA.height, CAMERA.width, CAMERA.samples_per_axis ** 2, 3)
        expected = (actual_inputs[:1].clone(), actual_inputs[2:].clone())
        values = (first.result(), tail.result())
        differs_from_normalized = False
        for value, exact in zip(values, expected):
            captured = value.raw.query_directions_world
            assert torch.equal(captured, exact), "capture changed pre-normalization query bits"
            differs_from_normalized |= not torch.equal(captured, value.raw.ray_directions_world)
            assert captured.untyped_storage().nbytes() == captured.numel() * captured.element_size(), (
                "captured input retains another request or chunk's allocation")
        assert differs_from_normalized, "probe did not exercise directions changed by diagnostic normalization"
        assert values[0].raw.query_directions_world.data_ptr() != values[1].raw.query_directions_world.data_ptr()
        del queries, actual_inputs, batch, ordinary, first, tail
        sensor.sample(views[::-1])
        renderer.sync(source_scene((3., 1., 2.)))
        renderer.close()
        for value, exact in zip(values, expected):
            assert torch.equal(value.raw.query_directions_world, exact)
    return {"mixed_handles": 3, "forced_chunks": 3,
            "trace_inputs_equal_captured_inputs_bitwise": True,
            "input_bits_differ_from_normalized_diagnostics": True,
            "owned_query_inputs_survive_sync_close": True}


def verify_fixture_observer_isolation_and_recovery():
    from devtools.sensors import _sampling as module
    source, views = source_scene(), poses(1)
    original_process, original_concat = module._process_samples, module._concat_depth

    def ordinary_diagnostics():
        with RayRenderer(source) as renderer:
            sensor = CameraSensor(SensorSampler(renderer, runtime=RUNTIME), CAMERA)
            batch = sensor._sampler.batch()
            handle = sensor.sample(views, batch=batch)
            handle._diagnostics = True
            batch.execute()
            raw = handle.result().raw
            assert type(raw) is SampleBatch
            assert not hasattr(raw, "query_directions_world"), "fixture observation leaked into ordinary batch"
        return True

    ordinary_diagnostics()
    with RayRenderer(source) as renderer, ThreadPoolExecutor(max_workers=1) as executor:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        sensor = CameraSensor(sampler, CAMERA)
        observations = []

        def check_other_thread(depth, raw, frozen, start, stop):
            assert raw.query_directions_world.is_cuda
            observations.append(executor.submit(ordinary_diagnostics).result(timeout=30))

        batch = sampler.batch()
        capture_sample(sensor, views, batch=batch, diagnostics=False, consumer=check_other_thread)
        batch.execute()
        assert observations == [True]
        assert module._process_samples is original_process
        assert module._concat_depth is original_concat
        assert "execute" not in batch.__dict__

        def fail_consumer(*args):
            raise ValueError("deliberate capture observer failure")

        failed = sampler.batch()
        handle = capture_sample(sensor, views, batch=failed, consumer=fail_consumer)
        rejected(failed.execute, (ValueError,))
        rejected(handle.result, (RuntimeError,))
        assert module._process_samples is original_process, "process observer survived a consumer exception"
        assert module._concat_depth is original_concat, "concat observer survived a consumer exception"
        assert "execute" not in failed.__dict__, "failed fixture batch retains execution observer"
        compact(sensor.sample(views), 1, CAMERA)
        retried = sampler.batch()
        good = capture_sample(sensor, views, batch=retried)
        retried.execute()
        assert good.result().raw.query_directions_world is not None
    ordinary_diagnostics()
    assert module._process_samples is original_process and module._concat_depth is original_concat
    return {"ordinary_batches_before_after": "unobserved",
            "ordinary_batch_other_thread_during_capture": "unobserved",
            "observers_restored_after_consumer_exception": True,
            "normal_and_fixture_sampling_after_failure": "passed"}


def verify_fixture_query_retained_budget():
    from devtools.sensors import _sampling as module
    source, views = source_scene(), poses(3)
    rays_per_view = CAMERA.width * CAMERA.height * CAMERA.samples_per_axis ** 2
    extra_consumer_bytes = 4096
    with RayRenderer(source) as renderer:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        sensor = CameraSensor(sampler, CAMERA)

        def reserved_bytes(diagnostics, capture):
            batch, reservations, observed = sampler.batch(), [], []
            if capture:
                def consume(depth, raw, frozen, start, stop):
                    assert raw.query_directions_world.shape == (*raw.subray_distance_m.shape, 3)
                    observed.append((start, stop))
                handle = capture_sample(sensor, views, batch=batch, diagnostics=diagnostics,
                                        consumer=consume, retained_output_bytes=extra_consumer_bytes)
            else:
                handle = sensor.sample(views, batch=batch)
                handle._diagnostics = diagnostics
            plan = module._plan_chunks

            def observe_plan(*args, **kwargs):
                reservations.append(args[3])
                return plan(*args, **kwargs)

            with patch.object(module, "_plan_chunks", side_effect=observe_plan), no_readback():
                batch.execute()
            assert len(reservations) == 1
            if capture:
                assert observed == [(0, len(views))]
            if diagnostics:
                assert (getattr(handle.result().raw, "query_directions_world", None) is not None) == capture
            else:
                compact(handle.result(), len(views), CAMERA)
            return reservations[0]

        baseline_retained = reserved_bytes(True, False)
        fixture_retained = reserved_bytes(True, True)
        extra_query_bytes = 2 * len(views) * rays_per_view * 3 * 4
        assert fixture_retained - baseline_retained == extra_consumer_bytes + extra_query_bytes
        baseline_compact = reserved_bytes(False, False)
        fixture_compact = reserved_bytes(False, True)
        assert fixture_compact - baseline_compact == extra_consumer_bytes, (
            "consumer-only observation must not reserve retained query copies")
    return {"captured_views": len(views), "owned_inputs_plus_concat_overlap_bytes": extra_query_bytes,
            "consumer_retained_bytes": extra_consumer_bytes,
            "planner_charges_owned_query_inputs": True,
            "consumer_only_observation_has_no_retained_input_copies": True}


def verify_output_ownership_and_cycle_free_release():
    source, views = source_scene(), poses()
    renderer = RayRenderer(source)
    sampler = SensorSampler(renderer, runtime=RUNTIME)
    sensor = CameraSensor(sampler, CAMERA)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(5):
            value = sensor.sample(views)
            del value
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(24):
            value = sensor.sample(views)
            del value
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        assert after <= before + 64 * 1024, (before, after)
        batch = sampler.batch()
        keep = sensor.sample(views[0], batch=batch)
        large_sensor = CameraSensor(sampler, DepthConfig(128, 96, 65., 50., .015, 6., samples_per_axis=2))
        release = large_sensor.sample(views, batch=batch)
        batch.execute()
        large = release.result()
        weak = weakref.ref(large.range_m)
        old = keep.result()
        reference = host(old.range_m).copy()
        del large, release, batch
        torch.cuda.synchronize()
        assert weak() is None, "one surviving handle retains an unrelated large result"
        # A surviving slice can retain another request's allocation even when
        # that other Tensor object itself has disappeared. Check storage too.
        shared = sampler.batch()
        small = large_sensor.sample(views[0], batch=shared)
        large_handle = large_sensor.sample(views * 8, batch=shared)
        shared.execute()
        small_value = small.result()
        assert small_value.range_m.untyped_storage().nbytes() == (
            small_value.range_m.numel() * small_value.range_m.element_size())
        del large_handle, shared
        diagnostic = sampler.batch()
        small_raw = capture_sample(sensor, views[0], batch=diagnostic)
        big_raw = capture_sample(sensor, views * 8, batch=diagnostic)
        diagnostic.execute()
        for name, tensor in tensor_fields(small_raw.result()).items():
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(), name
        del big_raw, diagnostic
        for _ in range(3):
            sensor.sample(poses(1)[0])
        renderer.sync(source_scene((3., 1., 2.)))
        close(host(old.range_m), reference)
        renderer.close()
        close(host(keep.result().range_m), reference)
        return {"gc_disabled": True, "warm_allocated_bytes": before,
                "after_24_calls_bytes": after, "unrelated_result_weakref_released": True,
                "small_request_storage_excludes_other_requests": True,
                "owned_output_survives_new_sample_sync_close": True}
    finally:
        renderer.close()
        if enabled:
            gc.enable()


def verify_cross_stream_collection_lifetime():
    from devtools.sensors import _sampling as module
    source, views = source_scene(), poses(16)
    with RayRenderer(source) as renderer:
        sampler = SensorSampler(renderer, runtime=RUNTIME)
        sensor = CameraSensor(sampler, CAMERA)
        expected = host(sensor.sample(views).range_m).copy()
        torch.cuda.synchronize()
        producer, executor, receiver = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        shape, dtype = (len(views), 4, 4), torch.float32
        # Prepare all three stream pools before introducing pending work. In
        # particular a fresh cudaMalloc must not supply an accidental barrier.
        with torch.cuda.stream(producer):
            reserve = [torch.empty(shape, dtype=dtype, device=renderer.device) for _ in range(96)]
        producer.synchronize()
        del reserve
        warm_views = tuple(Pose((view.position_m[0] - .8, *view.position_m[1:]),
                                view.quaternion_xyzw) for view in views)
        with torch.cuda.stream(executor):
            warm = sensor.sample(warm_views)
        executor.synchronize()
        with torch.cuda.stream(receiver):
            warm_calculation = torch.nan_to_num(warm.range_m, nan=-7., posinf=11., neginf=-13.).mul(1.25).add(.5)
        receiver.synchronize()
        del warm, warm_calculation

        def churn():
            with torch.cuda.stream(producer):
                buffers = [torch.empty(shape, dtype=dtype, device=renderer.device).fill_(math.nan)
                           for _ in range(64)]
            return buffers, [tensor.data_ptr() for tensor in buffers]

        # Negative control intentionally borrows a producer allocation without
        # record_stream. Retain the churn allocations so best-fit reuse cannot
        # keep cycling through one other block and miss the endangered block.
        with torch.cuda.stream(producer):
            unsafe = poses_to_tensor(views)
        unsafe_pointer = unsafe.data_ptr()
        executor.wait_stream(producer)
        with torch.cuda.stream(executor):
            torch.cuda._sleep(100_000_000)
            unsafe_copy = unsafe.clone()
            unsafe_done = torch.cuda.Event()
            unsafe_done.record()
        assert not unsafe_done.query(), "negative control did not establish pending reads"
        del unsafe
        unsafe_buffers, unsafe_pointers = churn()
        assert not unsafe_done.query(), "negative control finished before allocator pressure"
        assert unsafe_pointer in unsafe_pointers, "unprotected source allocation was not reused"
        unsafe_done.synchronize()
        assert np.isnan(host(unsafe_copy)).any(), "negative-control reuse did not corrupt the delayed read"
        del unsafe_buffers, unsafe_copy

        with torch.cuda.stream(producer):
            matrices = poses_to_tensor(views)
        protected_pointer = matrices.data_ptr()
        executor.wait_stream(producer)
        batch = sampler.batch()
        with torch.cuda.stream(executor):
            # Collection must freeze asynchronously, without a hidden execute.
            torch.cuda._sleep(500_000_000)
            handle = sensor.sample(matrices, batch=batch)
            cloned = torch.cuda.Event()
            cloned.record()
        assert not cloned.query(), "probe did not establish pending collection pressure"
        del matrices
        protected_buffers, protected_pointers = churn()
        assert not cloned.query(), "protected clone finished before allocator pressure"
        assert protected_pointer not in protected_pointers, "pending protected storage was reused"

        # Budget inspection can complete earlier queued work. Place this
        # separate delay after planning and before the actual result writes,
        # rather than sleeping after a result that was already computed.
        original = module._compact_rays
        delayed = False

        def delay_before_rays(*args, **kwargs):
            nonlocal delayed
            if not delayed:
                torch.cuda._sleep(100_000_000)
                delayed = True
            return original(*args, **kwargs)

        with torch.cuda.stream(executor), patch.object(module, "_compact_rays", side_effect=delay_before_rays):
            batch.execute()
            produced = torch.cuda.Event()
            produced.record()
        assert delayed and not produced.query(), "result writes were not pending before cross-stream result()"
        with torch.cuda.stream(receiver):
            value = handle.result()
            calculated = torch.nan_to_num(value.range_m, nan=-7., posinf=11., neginf=-13.).mul(1.25).add(.5)
            consumed = torch.cuda.Event()
            consumed.record()
        # No executor/result-ready CPU synchronization precedes consumption.
        consumed.synchronize()
        expected_calculation = np.nan_to_num(expected, nan=-7., posinf=11., neginf=-13.) * 1.25 + .5
        close(host(calculated), expected_calculation)
        del protected_buffers
    return {"caller_waits_for_producer": True, "pending_clone_pressure": True,
            "negative_control": {"source_pointer": unsafe_pointer,
                                 "reused_while_read_pending": True, "delayed_copy_corrupted": True},
            "protected_source_pointer": protected_pointer,
            "protected_pointer_absent_from_64_pending_churn_allocations": True,
            "result_pending_before_third_stream_access": True,
            "only_receiver_event_synchronized_for_result_check": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/sensors/batch-verification.json")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    checks = (("inputs_independent_geometry", verify_inputs_and_oracle),
              ("interleaved_requests_freezing_grouping", verify_interleaved_freezing_and_grouping),
              ("epoch_failure_and_execution_guards", verify_epoch_failure_and_execution_guards),
              ("chunk_consumers_and_diagnostic_parity", verify_chunk_budget_consumers_and_diagnostics),
              ("fixture_query_inputs_across_chunks", verify_fixture_query_inputs_across_chunks),
              ("fixture_observer_isolation_recovery", verify_fixture_observer_isolation_and_recovery),
              ("fixture_query_retained_budget", verify_fixture_query_retained_budget),
              ("output_lifetime_without_gc", verify_output_ownership_and_cycle_free_release),
              ("cross_stream_collection_storage", verify_cross_stream_collection_lifetime))
    if args.only and set(args.only) - {name for name, _ in checks}:
        parser.error("unknown check name")
    records = []
    for name, action in checks:
        if args.only and name not in args.only:
            continue
        try:
            records.append({"name": name, "status": "passed", "detail": action()})
            print(name + ": passed", flush=True)
        except Exception as error:
            records.append({"name": name, "status": "failed", "error_type": type(error).__name__,
                            "error": str(error), "traceback": traceback.format_exc()})
            print(name + ": failed: " + str(error), flush=True)
    result = {"status": "passed" if all(r["status"] == "passed" for r in records) else "failed",
              "checks": records, "device": torch.cuda.get_device_name()}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
