"""Independent physical sensors and explicit, shared GPU sampling batches.

A sensor consumes complete sensor-to-world poses. Device mounting, vehicle
layouts and recognition belong to development callers. Immediate sampling uses
exactly the same request collection and execution path as an explicit batch.
"""

from collections.abc import Sequence
from dataclasses import fields, replace
import math

import torch

from ..config import DepthConfig, SamplingRuntimeConfig
from ._rendering import RayRenderer
from ..scene import Pose
from ._budget import plan
from ._ops import (
    DepthBatch, _check_poses, _compact_rays, _slice_hits,
    compact_depth as _compact_depth, concat_depth as _concat_depth,
    depth_output_bytes as _depth_output_bytes, poses_to_tensor as _poses_to_tensor,
    process_samples as _process_samples,
)


__all__ = ["DepthBatch", "SensorSampler", "ToFSensor", "CameraSensor"]


def _plan_chunks(renderer, item_count, rays_per_item, retained_output_bytes,
                 runtime, extra_fixed_bytes=0, *, retained_output_allocated_bytes=0):
    return plan(renderer, item_count, rays_per_item, retained_output_bytes, runtime,
                extra_fixed_bytes=extra_fixed_bytes,
                retained_output_allocated_bytes=retained_output_allocated_bytes)


def _freeze_poses(world_poses, device):
    if isinstance(world_poses, torch.Tensor):
        if world_poses.ndim == 2:
            world_poses = world_poses.unsqueeze(0)
        _check_poses(world_poses)
        if world_poses.device != device:
            raise ValueError("poses must use the sampler's CUDA device")
        # A queued request is a value snapshot, not a borrowed tensor reference.
        # Readiness of a tensor produced on another stream remains the caller's
        # responsibility, as for ordinary PyTorch operations.
        world_poses.record_stream(torch.cuda.current_stream(device))
        return world_poses.detach().clone(memory_format=torch.contiguous_format)
    if isinstance(world_poses, Pose):
        world_poses = (world_poses,)
    if not isinstance(world_poses, Sequence):
        raise TypeError("world_poses must be Pose values or a CUDA float32 pose tensor")
    return _poses_to_tensor(world_poses, device)


def _slice_raw(raw, start, stop):
    # Internally generated raw always has one compact origin per view.
    return replace(raw, **{field.name: getattr(raw, field.name)[start:stop]
                           for field in fields(raw)
                           if isinstance(getattr(raw, field.name), torch.Tensor)})


def _slice_depth(depth, start, stop):
    updates = {field.name: getattr(depth, field.name)[start:stop]
               for field in fields(depth)
               if isinstance(getattr(depth, field.name), torch.Tensor)}
    updates["raw"] = _slice_raw(depth.raw, start, stop)
    return replace(depth, **updates)


def _owned_diagnostics(depth):
    # A tiny diagnostic request must not retain the entire mixed trace's
    # storage through slices. The explicit diagnostic reservation covers these
    # owned copies, including temporary concatenation overlap.
    raw = replace(depth.raw, **{field.name: getattr(depth.raw, field.name).clone()
                               for field in fields(depth.raw)
                               if isinstance(getattr(depth.raw, field.name), torch.Tensor)})
    return replace(depth, raw=raw, **{field.name: getattr(depth, field.name).clone()
                                     for field in fields(depth)
                                     if isinstance(getattr(depth, field.name), torch.Tensor)})


def _record_result(depth, stream):
    for value in (depth, depth.raw):
        if value is not None:
            for field in fields(value):
                tensor = getattr(value, field.name)
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(stream)


class _SamplingHandle:
    def __init__(self, sensor, poses):
        self._sensor = sensor
        self._poses = poses
        self._diagnostics = False
        self._consumer = None
        self._retained_output_bytes = 0
        self._pieces = []
        self._result = None
        self._state = "pending"
        self._error = None
        self._result_ready = None
        self._ready = torch.cuda.Event()
        self._ready.record(torch.cuda.current_stream(poses.device))

    def result(self) -> DepthBatch:
        """Return this request's CUDA result; never execute or synchronize CPU."""
        if self._state == "failed":
            raise RuntimeError("sampling batch failed") from self._error
        if self._state != "executed":
            raise RuntimeError("sampling request is pending; execute its batch first")
        stream = torch.cuda.current_stream(self._result.range_m.device)
        stream.wait_event(self._result_ready)
        _record_result(self._result, stream)
        return self._result


class _SamplingBatch:
    def __init__(self, sampler):
        sampler.renderer._require_open()
        self._sampler = sampler
        self._scene_version = sampler.renderer.scene_version
        self._state = "collecting"
        self._requests = []
        self._error = None
        self._ready = None
        self.execution_info = None

    def _submit(self, sensor, world_poses):
        if self._state != "collecting":
            raise RuntimeError("sampling batch is no longer collecting requests")
        if not isinstance(sensor, (ToFSensor, CameraSensor)) or sensor._sampler is not self._sampler:
            raise ValueError("sensor and batch must belong to the same sampler")
        self._sampler.renderer._require_open()
        poses = _freeze_poses(world_poses, self._sampler.renderer.device)
        handle = _SamplingHandle(sensor, poses)
        self._requests.append(handle)
        return handle

    def _schedule(self, ray_budget):
        chunks, current, used = [], [], 0
        for request in self._requests:
            config = request._sensor._config
            rays = config.height * config.width * config.samples_per_axis ** 2
            if rays > ray_budget:
                raise MemoryError("sampling budget cannot fit one complete sensor view")
            start = 0
            while start < len(request._poses):
                capacity = (ray_budget - used) // rays
                if capacity == 0:
                    chunks.append(current)
                    current, used = [], 0
                    continue
                stop = min(len(request._poses), start + capacity)
                current.append((request, start, stop))
                used += (stop - start) * rays
                start = stop
        if current:
            chunks.append(current)
        return chunks

    def _execute_chunk(self, segments):
        # Group equal sensor policies first, so independent requests do not turn
        # into separate ray generation or tensor reduction dispatches.
        groups = {}
        for request, start, stop in segments:
            sensor = request._sensor
            key = (sensor._config, sensor._reduction, sensor._quantile, sensor._signal_floor)
            groups.setdefault(key, []).append((request, start, stop))
        prepared, origin_parts, direction_parts, index_parts, limit_parts = [], [], [], [], []
        origin_offset, ray_offset = 0, 0
        for key, parts in groups.items():
            config = key[0]
            pose_parts = [request._poses[start:stop] for request, start, stop in parts]
            poses = pose_parts[0] if len(pose_parts) == 1 else torch.cat(pose_parts)
            origins, directions, indices = _compact_rays(poses, config)
            origin_parts.append(origins)
            direction_parts.append(directions)
            index_parts.append(indices + origin_offset)
            limit_parts.append(torch.full((len(directions),), config.max_range_m,
                                          device=poses.device, dtype=torch.float32))
            prepared.append((key, parts, poses, origins, ray_offset, ray_offset + len(directions)))
            origin_offset += len(origins)
            ray_offset += len(directions)
        def joined(parts):
            return parts[0] if len(parts) == 1 else torch.cat(parts)
        origins, directions = joined(origin_parts), joined(direction_parts)
        indices, limits = joined(index_parts), joined(limit_parts)
        hits = self._sampler.renderer.trace(origins, directions, limits, origin_indices=indices)
        self.execution_info["trace_calls"] += 1
        # Drop redundant full groups before postprocessing. The trace retains
        # its asynchronous inputs via record_stream.
        del origin_parts, direction_parts, index_parts, limit_parts, indices, limits
        for key, parts, poses, group_origins, ray_start, ray_stop in prepared:
            config, reduction, quantile, signal_floor = key
            depth = _process_samples(_slice_hits(hits, ray_start, ray_stop), group_origins,
                                    directions[ray_start:ray_stop], poses, config,
                                    reduction=reduction, quantile=quantile,
                                    signal_floor=signal_floor)
            self.execution_info["postprocess_calls"] += 1
            local = 0
            for request, start, stop in parts:
                length = stop - start
                piece = _slice_depth(depth, local, local + length)
                if request._diagnostics:
                    piece = _owned_diagnostics(piece)
                if request._consumer is not None:
                    request._consumer(piece, piece.raw, poses[local:local + length], start, stop)
                if request._diagnostics:
                    retained_piece = piece
                else:
                    retained_piece = _compact_depth(piece)
                    if len(parts) > 1:
                        # Independent handles must not retain other requests'
                        # pixel outputs through the merged reduction storage.
                        retained_piece = replace(retained_piece, range_m=piece.range_m.clone())
                request._pieces.append(retained_piece)
                local += length
                del piece
            del depth

    def execute(self):
        """Submit this batch once on the current CUDA stream, without CPU waits."""
        if self._state != "collecting":
            raise RuntimeError("sampling batch can only execute once")
        self._state = "executing"
        renderer = self._sampler.renderer
        guarded = False
        try:
            renderer._begin_sampling(self._scene_version)
            guarded = True
            device = renderer.device
            stream = torch.cuda.current_stream(device)
            for request in self._requests:
                stream.wait_event(request._ready)
                request._poses.record_stream(stream)
            pose_bytes = sum(request._poses.numel() * request._poses.element_size()
                             for request in self._requests)
            retained = pose_bytes + sum(
                _depth_output_bytes(request._sensor._config, len(request._poses),
                                   diagnostics=request._diagnostics)
                + request._retained_output_bytes for request in self._requests)
            ray_count = sum(len(request._poses) * request._sensor._rays_per_view
                            for request in self._requests)
            if ray_count:
                capacity, budget = _plan_chunks(renderer, ray_count, 1, retained,
                                                self._sampler.runtime,
                                                retained_output_allocated_bytes=pose_bytes)
                chunks = self._schedule(capacity)
            else:
                capacity, budget, chunks = 0, None, []
            self.execution_info = {
                "chunks": len(chunks), "chunk_count": len(chunks), "chunk_ray_budget": capacity,
                "trace_calls": 0, "postprocess_calls": 0, "rays_per_item": 1,
                "request_count": len(self._requests),
                "view_count": sum(len(request._poses) for request in self._requests),
                "ray_count": ray_count, "compact_origins": True,
                "full_scene_queries_per_chunk": 1, "scene_version": self._scene_version,
                "planner": budget,
            }
            for segments in chunks:
                self._execute_chunk(segments)
            for request in self._requests:
                request._result = _concat_depth(request._pieces, diagnostics=request._diagnostics,
                                               execution_info=self.execution_info)
                request._pieces.clear()
                # Caller geometry and directions survive only in explicit raw
                # diagnostics; ordinary requests keep pixel-sized outputs.
                request._poses = None
            self._ready = torch.cuda.Event()
            self._ready.record(stream)
            for request in self._requests:
                request._result_ready = self._ready
                request._state = "executed"
            self._state = "executed"
        except BaseException as error:
            # Keep a useful failure cause without retaining this execution's
            # traceback and all of its temporary CUDA tensors through handles.
            self._error = RuntimeError(f"{type(error).__name__}: {error}")
            self._state = "failed"
            for request in self._requests:
                request._pieces.clear()
                request._result = None
                request._poses = None
                request._state = "failed"
                request._error = self._error
            raise
        finally:
            for request in self._requests:
                request._sensor = request._consumer = request._ready = None
                request._poses = None
            self._requests.clear()
            if guarded:
                renderer._end_sampling()
        return self


class SensorSampler:
    """Shared scene executor and soft GPU workset policy, without device rigs."""

    def __init__(self, renderer: RayRenderer, runtime: SamplingRuntimeConfig):
        if not isinstance(renderer, RayRenderer):
            raise TypeError("renderer must be a RayRenderer")
        renderer._require_open()
        if not isinstance(runtime, SamplingRuntimeConfig):
            raise TypeError("runtime must be a SamplingRuntimeConfig")
        self.renderer = renderer
        self.runtime = runtime

    def batch(self):
        return _SamplingBatch(self)


class _Sensor:
    def sample(self, world_poses, *, batch=None):
        """Return compact depth now, or collect a handle in an explicit batch."""
        if batch is not None:
            if not isinstance(batch, _SamplingBatch):
                raise TypeError("batch must come from SensorSampler.batch()")
            return batch._submit(self, world_poses)
        batch = self._sampler.batch()
        handle = batch._submit(self, world_poses)
        batch.execute()
        return handle.result()

    def _initialize(self, sampler, config):
        if not isinstance(sampler, SensorSampler) or not isinstance(config, DepthConfig):
            raise TypeError("expected SensorSampler and DepthConfig")
        self._sampler, self._config = sampler, config
        self._rays_per_view = config.height * config.width * config.samples_per_axis ** 2


class ToFSensor(_Sensor):
    """ToF sampling with per-cell distance-weighted subray quantile reduction."""

    def __init__(self, sampler: SensorSampler, config: DepthConfig, *,
                 quantile, signal_floor):
        self._initialize(sampler, config)
        if (isinstance(quantile, bool) or not isinstance(quantile, (float, int))
                or not math.isfinite(quantile) or not 0 <= quantile <= 1):
            raise ValueError("quantile must be in [0,1]")
        if (isinstance(signal_floor, bool) or not isinstance(signal_floor, (float, int))
                or not math.isfinite(signal_floor) or signal_floor <= 0):
            raise ValueError("signal_floor must be positive")
        self._reduction = "weighted_quantile"
        self._quantile, self._signal_floor = float(quantile), float(signal_floor)


class CameraSensor(_Sensor):
    """Depth camera using its own nearest-subray policy; no recognition facts."""

    def __init__(self, sampler: SensorSampler, config: DepthConfig):
        self._initialize(sampler, config)
        self._reduction = "nearest"
        self._quantile, self._signal_floor = 0.2, 0.08
