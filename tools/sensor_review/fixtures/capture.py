"""Same-pass diagnostic capture for review and tests only.

Register the normal sensor request first, then opt that private handle into
capture before execution. Full raw tensors and any consumer outputs are charged
to the same workset planner; the fixture never adds another scene query.
"""

import operator


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
    handle._diagnostics = diagnostics
    handle._consumer = consumer
    handle._retained_output_bytes = retained_output_bytes
    return handle
