"""Target-only geometry for review, matching the physical core's float32 math."""

import torch
import warp as wp

from devtools.sensors._ops import sample_ray_origins


@wp.kernel
def _sphere_query(origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
                  centers: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
                  limit: float, target: int, distances: wp.array(dtype=float),
                  identities: wp.array(dtype=wp.int64), points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    distances[i] = 0.0
    identities[i] = wp.int64(-1)
    points[i] = wp.vec3(0.0)
    origin = origins[i]
    direction = directions[i]
    finite_origin = wp.isfinite(origin[0]) and wp.isfinite(origin[1]) and wp.isfinite(origin[2])
    finite_direction = wp.isfinite(direction[0]) and wp.isfinite(direction[1]) and wp.isfinite(direction[2])
    if finite_origin and finite_direction and wp.isfinite(limit) and limit > 0.0:
        scale = wp.max(wp.abs(direction[0]), wp.max(wp.abs(direction[1]), wp.abs(direction[2])))
        if scale > 0.0:
            direction = wp.normalize(direction / scale)
            query_limit = limit + wp.max(1.0e-6, limit * 1.0e-6)
            relative = origin - centers[0]
            b = wp.dot(relative, direction)
            perpendicular = relative - b * direction
            discriminant = radii[0] * radii[0] - wp.dot(perpendicular, perpendicular)
            if discriminant >= 0.0:
                root = wp.sqrt(discriminant)
                t = -b - root
                if t < 0.0:
                    t = -b + root
                if t >= 0.0 and t <= query_limit and wp.isfinite(t):
                    distances[i] = t
                    identities[i] = wp.int64(target)
                    points[i] = origin + t * direction


def sphere_returns(raw, geometry, max_range_m, target_index):
    """Use captured query bits, never reconstructed poses or physical hits."""
    directions = getattr(raw, "query_directions_world", None)
    if directions is None:
        raise ValueError("target perception requires same-pass query capture")
    shape, device = raw.subray_distance_m.shape, directions.device
    directions = directions.reshape(-1, 3).contiguous()
    origins = sample_ray_origins(raw).expand(*shape, 3).reshape(-1, 3).contiguous()
    centers = torch.tensor((geometry.pose.position_m,), dtype=torch.float32, device=device)
    radii = torch.tensor((geometry.radius_m,), dtype=torch.float32, device=device)
    distances = torch.empty(shape, dtype=torch.float32, device=device)
    identities = torch.empty(shape, dtype=torch.int64, device=device)
    points = torch.empty((*shape, 3), dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream(device)
    with wp.ScopedStream(wp.stream_from_torch(stream)):
        wp.launch(_sphere_query, dim=distances.numel(), inputs=[
            wp.from_torch(origins.detach(), dtype=wp.vec3),
            wp.from_torch(directions.detach(), dtype=wp.vec3),
            wp.from_torch(centers, dtype=wp.vec3), wp.from_torch(radii), max_range_m, target_index,
            wp.from_torch(distances.reshape(-1)), wp.from_torch(identities.reshape(-1)),
            wp.from_torch(points.reshape(-1, 3), dtype=wp.vec3)])
    for tensor in (origins, directions, centers, radii, distances, identities, points):
        tensor.record_stream(stream)
    return distances, identities, points
