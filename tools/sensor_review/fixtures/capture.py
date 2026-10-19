"""Same-pass diagnostic capture for review and tests only.

Register the normal sensor request first, then opt that private handle into
capture before execution. Full raw tensors and any consumer outputs are charged
to the same workset planner; the fixture never adds another scene query.
"""

from dataclasses import dataclass, fields, replace
import operator
import threading
from unittest.mock import patch

import torch

from devtools.sensors import _sampling
from devtools.sensors._ops import SampleBatch, process_samples


@dataclass(frozen=True)
class CapturedSamples(SampleBatch):
    # process_samples normalizes diagnostic directions with Torch. Keep the
    # actual query inputs separately: repeating that normalization loses bits.
    query_directions_world: torch.Tensor | None = None


def _with_query(raw, directions):
    return CapturedSamples(**{field.name: getattr(raw, field.name) for field in fields(SampleBatch)},
                           query_directions_world=directions)


def capture_process_samples(hits, origins, directions, poses, config, **kwargs):
    """Observe query inputs before the ordinary physical reduction."""
    depth = process_samples(hits, origins, directions, poses, config, **kwargs)
    query = directions.reshape(*depth.raw.subray_distance_m.shape, 3)
    return replace(depth, raw=_with_query(depth.raw, query))


def _concat_captured(depths, **kwargs):
    depth = _sampling_concat_depth(depths, **kwargs)
    if depth.raw is not None:
        values = [part.raw.query_directions_world for part in depths]
        directions = values[0] if len(values) == 1 else torch.cat(values)
        depth = replace(depth, raw=_with_query(depth.raw, directions))
    return depth


_sampling_concat_depth = _sampling._concat_depth
_capture_lock = threading.RLock()


def _observe_execution(batch):
    if getattr(batch, "_fixture_capture", False):
        return
    execute = batch.execute

    def captured_execute():
        # Instrument only this execution/thread and restore even after a failed
        # consumer. Ordinary samplers on other threads retain their own path.
        owner = threading.get_ident()
        with _capture_lock:
            previous_process, previous_concat = _sampling._process_samples, _sampling._concat_depth

            def process(*args, **kwargs):
                if threading.get_ident() != owner:
                    return previous_process(*args, **kwargs)
                return capture_process_samples(*args, **kwargs)

            def concat(*args, **kwargs):
                if threading.get_ident() != owner:
                    return previous_concat(*args, **kwargs)
                return _concat_captured(*args, **kwargs)

            try:
                with patch.object(_sampling, "_process_samples", process), \
                     patch.object(_sampling, "_concat_depth", concat):
                    return execute()
            finally:
                # Do not keep the executor/renderer alive through a closure on
                # the completed batch, including the failed-consumer path.
                del batch.execute

    batch.execute = captured_execute
    batch._fixture_capture = True


def capture_sample(sensor, world_poses, *, batch, diagnostics=True, consumer=None,
                   retained_output_bytes=0):
    """Observe a normal sample through intentionally private handle fields.

    consumer(depth, raw, frozen_poses, start, stop) receives one request-local
    interval and must release raw on return. Charge any outputs it retains,
    concatenation overlap and extra temporary space to retained_output_bytes.
    diagnostics=True retains owned raw copies in the final handle result.
    """
    if batch is None:
        raise TypeError("capture requires an explicit sampling batch")
    if not isinstance(diagnostics, bool):
        raise TypeError("diagnostics must be bool")
    if consumer is not None and not callable(consumer):
        raise TypeError("consumer must be callable")
    if isinstance(retained_output_bytes, bool):
        raise ValueError("retained_output_bytes must be a nonnegative integer")
    try:
        retained_output_bytes = operator.index(retained_output_bytes)
    except TypeError as error:
        raise ValueError("retained_output_bytes must be a nonnegative integer") from error
    if retained_output_bytes < 0:
        raise ValueError("retained_output_bytes must be a nonnegative integer")
    handle = sensor.sample(world_poses, batch=batch)
    _observe_execution(batch)
    handle._diagnostics = diagnostics
    handle._consumer = consumer
    # Explicitly reserve owned query-direction copies and concatenation overlap.
    query_bytes = 24 * len(handle._poses) * sensor._rays_per_view if diagnostics else 0
    handle._retained_output_bytes = retained_output_bytes + query_bytes
    return handle
