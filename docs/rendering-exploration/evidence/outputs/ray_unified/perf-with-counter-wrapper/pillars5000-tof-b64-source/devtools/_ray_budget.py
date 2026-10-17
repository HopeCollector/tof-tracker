"""A conservative, per-request workset budget for batched ray sampling.

The configured fraction limits sampling's own scene, retained result and chunk
workset. Existing training tensors constrain physical free space, but are not
charged again against this fraction. Estimates are planning aids, not driver
reservations or hard VRAM isolation; fragmentation and other processes can
still cause an allocation failure after planning.
"""

import math
import operator

import torch


WORKSPACE_BYTES_PER_RAY = 384
SAFETY_MARGIN_BYTES = 64 * 1024 * 1024


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def plan(renderer, item_count, rays_per_item, retained_output_bytes, runtime,
         extra_fixed_bytes=0, *, retained_output_allocated_bytes=0):
    """Return ``(chunk_items, info)`` without changing GPU allocation state.

    ``retained_output_bytes`` is the total output storage this request intends
    to retain, usually estimated before allocating it. During a later replan,
    pass the already-allocated portion separately: it still counts against the
    sampling workset cap, but need not be allocated again from physical space.
    ``extra_fixed_bytes`` is any additional request-wide planned storage.

    The 384 bytes/ray allowance includes raw trace output, directions/indices,
    postprocessing and the outer sphere-perception temporary storage. Retained output is also
    counted separately; their overlap intentionally makes the plan conservative.
    A native Warp BVH allowance is estimated, never reported as measured bytes.
    """
    item_count = _integer(item_count, "item_count", 1)
    rays_per_item = _integer(rays_per_item, "rays_per_item", 1)
    retained_output_bytes = _integer(retained_output_bytes, "retained_output_bytes")
    extra_fixed_bytes = _integer(extra_fixed_bytes, "extra_fixed_bytes")
    retained_output_allocated_bytes = _integer(retained_output_allocated_bytes,
                                               "retained_output_allocated_bytes")
    if retained_output_allocated_bytes > retained_output_bytes:
        raise ValueError("already-allocated output cannot exceed planned retained output")
    fraction = getattr(runtime, "max_gpu_memory_fraction", None)
    if (isinstance(fraction, bool) or not isinstance(fraction, (float, int))
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("runtime.max_gpu_memory_fraction must be in (0,1]")
    device = renderer.device
    memory = renderer.memory_info
    scene_bytes = int(memory["scene_geometry_buffer_bytes"])
    native_bytes = memory.get("warp_bvh_estimated_bytes")
    if native_bytes is None:
        count = int(getattr(renderer, "triangle_count", 0))
        native_bytes = count * 128 + 4096 if count else 0
    native_bytes = int(native_bytes)
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    reusable_cache = max(0, reserved - allocated)
    physical_available = min(total, free + reusable_cache)
    fraction_limit = int(total * fraction)
    fixed_workset = (scene_bytes + native_bytes + retained_output_bytes
                     + extra_fixed_bytes + SAFETY_MARGIN_BYTES)
    future_retained = retained_output_bytes - retained_output_allocated_bytes
    # Scene/BVH and already-live caller tensors are already reflected in free.
    # Only future storage and the safety allowance are subtracted on this side.
    fraction_workspace = fraction_limit - fixed_workset
    physical_workspace = (physical_available - future_retained
                          - extra_fixed_bytes - SAFETY_MARGIN_BYTES)
    workspace_available = min(fraction_workspace, physical_workspace)
    per_item = rays_per_item * WORKSPACE_BYTES_PER_RAY
    chunk_items = min(item_count, max(0, workspace_available // per_item))
    info = {
        "max_gpu_memory_fraction": float(fraction), "cuda_total_bytes": total,
        "sampling_workset_limit_bytes": fraction_limit,
        "scene_tensor_bytes": scene_bytes, "native_bvh_estimated_bytes": native_bytes,
        "native_bvh_measured_bytes": memory.get("warp_bvh_bytes"),
        "native_estimate_scope": "128 bytes/triangle plus 4 KiB allowance; not measured and not a hard upper bound",
        "retained_output_bytes": retained_output_bytes,
        "retained_output_allocated_bytes": retained_output_allocated_bytes,
        "future_retained_output_bytes": future_retained,
        "extra_fixed_bytes": extra_fixed_bytes, "safety_margin_bytes": SAFETY_MARGIN_BYTES,
        "cuda_free_bytes": free, "torch_allocated_bytes": allocated,
        "torch_reserved_bytes": reserved, "torch_reusable_cache_bytes": reusable_cache,
        "physical_available_estimated_bytes": physical_available,
        "fraction_workspace_bytes": fraction_workspace,
        "physical_workspace_bytes": physical_workspace,
        "workspace_bytes_per_ray": WORKSPACE_BYTES_PER_RAY,
        "rays_per_item": rays_per_item, "workspace_bytes_per_item": per_item,
        "requested_items": item_count, "chunk_items": chunk_items,
        "chunk_count": ((item_count + chunk_items - 1) // chunk_items
                        if chunk_items else None),
        "limiting_factor": "sampling_workset_fraction" if fraction_workspace <= physical_workspace else "physical_free_plus_torch_cache",
        "scope": "soft sampling workset cap, not a process-wide allocator limit or driver reservation; existing training allocations constrain free space only; cache reuse and native BVH size are estimates",
    }
    info["planned_chunk_workset_bytes"] = fixed_workset + chunk_items * per_item
    if chunk_items < 1:
        error = MemoryError(
            "ray sampling budget cannot fit scene, retained output and one item: "
            f"fraction workspace {fraction_workspace} B, physical workspace "
            f"{physical_workspace} B, required per item {per_item} B; "
            "reduce retained output or batch size, or raise the sampling workset fraction")
        error.budget_info = info
        raise error
    return chunk_items, info
