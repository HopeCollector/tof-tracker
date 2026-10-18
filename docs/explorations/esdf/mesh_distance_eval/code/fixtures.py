"""Small deterministic scenes and independent CPU geometry references.

The reference never calls Warp, the candidate backend, or a proximity package.
Distances use projection onto a triangle plane plus its three closed edges;
solid-angle winding supplies a separate inside/outside test. Analytic primitives
and an exposed-rectangle union reference provide additional checks.
"""

from pathlib import Path

import numpy as np

from devtools.mesh import MeshMap, MeshObject
from devtools.scene import Pose, Scene, Sphere


def make_box(name, lo, hi):
    """Return one outward-wound, closed, axis-aligned box in world metres."""
    x0, y0, z0 = map(float, lo)
    x1, y1, z1 = map(float, hi)
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError("box limits must be strictly increasing")
    vertices = ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    triangles = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                 (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                 (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
    return MeshObject(name, vertices, triangles, "obstacle")


def transform_mesh(mesh, pose, name=None):
    """Bake a full 3D rigid pose into world vertices, preserving topology."""
    return MeshObject(name or mesh.name,
                      tuple(pose.body_to_world(v) for v in mesh.vertices_m),
                      mesh.triangles, mesh.kind)


def make_ground(size=(10.0, 10.0, 6.0)):
    x, y = float(size[0]) / 2, float(size[1]) / 2
    return MeshObject("ground", ((-x, -y, 0.0), (x, -y, 0.0),
                                 (x, y, 0.0), (-x, y, 0.0)),
                      ((0, 1, 2), (0, 2, 3)), "ground")


def make_scene(obstacles=(), target_position=(3.0, 3.0, 2.0),
               size=(10.0, 10.0, 6.0), target_radius=0.3):
    """Construct the complete public scene contract, including finite ground."""
    background = MeshMap(Path("<analytic-fixture>"),
                         (*tuple(obstacles), make_ground(size)),
                         ("unit metre",), tuple(map(float, size)))
    return Scene(background, Sphere(Pose((-3.0, -3.0, 1.0)), 0.2),
                 Sphere(Pose(tuple(map(float, target_position))), float(target_radius)),
                 t_ref=0.0)


def write_obj(path, obstacles=(), size=(10.0, 10.0, 6.0)):
    """Write a complete fixture consumable by Scene.from_obj()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# unit metre", "# size_xyz " + " ".join(map(str, size))]
    offset = 1
    for obj in (*tuple(obstacles), make_ground(size)):
        lines.append(f"o {obj.name}")
        lines.extend("v " + " ".join(format(x, ".17g") for x in vertex)
                     for vertex in obj.vertices_m)
        lines.extend("f " + " ".join(str(i + offset) for i in face)
                     for face in obj.triangles)
        offset += len(obj.vertices_m)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def box_sdf(points, lo, hi):
    points, lo, hi = (np.asarray(v, dtype=np.float64) for v in (points, lo, hi))
    q = np.abs(points - (lo + hi) / 2) - (hi - lo) / 2
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def sphere_sdf(points, center, radius):
    return np.linalg.norm(np.asarray(points, dtype=np.float64) - center, axis=-1) - radius


def ground_distance(points, size=(10.0, 10.0, 6.0)):
    points = np.asarray(points, dtype=np.float64)
    nearest = points.copy()
    nearest[:, 0] = np.clip(nearest[:, 0], -size[0] / 2, size[0] / 2)
    nearest[:, 1] = np.clip(nearest[:, 1], -size[1] / 2, size[1] / 2)
    nearest[:, 2] = 0
    return np.linalg.norm(points - nearest, axis=-1)


def triangle_distance(points, vertices, triangles, chunk_size=128):
    """Nearest original triangle distance and closest point, in float64.

    Project onto each triangle's plane and accept its barycentric interior;
    independently minimize distance to each of its three segments. This simple
    exhaustive CPU reference shares no query code or acceleration with Warp.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e0, e1 = b - a, c - a
    normal = np.cross(e0, e1)
    normal2 = np.einsum("ti,ti->t", normal, normal)
    d00, d01, d11 = (np.einsum("ti,ti->t", u, v)
                     for u, v in ((e0, e0), (e0, e1), (e1, e1)))
    denom = d00 * d11 - d01 * d01
    distances, closest = [], []
    for start in range(0, len(points), chunk_size):
        p = points[start:start + chunk_size, None, :]
        pa = p - a
        projection = p - (np.einsum("pti,ti->pt", pa, normal) / normal2)[..., None] * normal
        projected_a = projection - a
        d20 = np.einsum("pti,ti->pt", projected_a, e0)
        d21 = np.einsum("pti,ti->pt", projected_a, e1)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        interior = (v >= -1e-13) & (w >= -1e-13) & (v + w <= 1 + 1e-13)
        candidate_points = [projection]
        candidate_dist2 = [np.where(interior, np.sum((p - projection) ** 2, axis=-1), np.inf)]
        for first, second in ((a, b), (b, c), (c, a)):
            edge = second - first
            t = np.clip(np.einsum("pti,ti->pt", p - first, edge) /
                        np.einsum("ti,ti->t", edge, edge), 0, 1)
            q = first + t[..., None] * edge
            candidate_points.append(q)
            candidate_dist2.append(np.sum((p - q) ** 2, axis=-1))
        dist2 = np.stack(candidate_dist2, axis=2)
        best = dist2.reshape(len(p), -1).argmin(axis=1)
        face_index, feature_index = np.unravel_index(best, dist2.shape[1:])
        all_closest = np.stack(candidate_points, axis=2)
        closest.append(all_closest[np.arange(len(p)), face_index, feature_index])
        distances.append(np.sqrt(dist2[np.arange(len(p)), face_index, feature_index]))
    return np.concatenate(distances), np.concatenate(closest)


def winding_number(points, vertices, triangles, chunk_size=128):
    """Independent solid-angle winding; overlapping outward shells add."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    result = []
    for start in range(0, len(points), chunk_size):
        vectors = tri[None, :, :, :] - points[start:start + chunk_size, None, None, :]
        a, b, c = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        dot = lambda u, v: np.sum(u * v, axis=-1)
        numerator = dot(a, np.cross(b, c))
        denominator = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb
        result.append(np.sum(2 * np.arctan2(numerator, denominator), axis=1) / (4 * np.pi))
    return np.concatenate(result)


def joined_mesh(obstacles):
    vertices, triangles = [], []
    for obj in obstacles:
        offset = len(vertices)
        vertices.extend(obj.vertices_m)
        triangles.extend(tuple(i + offset for i in face) for face in obj.triangles)
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def mesh_signed_distance(points, obstacles):
    """Signed distance to original faces, not a Boolean-union interior SDF."""
    obstacles = tuple(obstacles)
    if not obstacles:
        return np.full(len(points), np.inf)
    vertices, triangles = joined_mesh(obstacles)
    distance, _ = triangle_distance(points, vertices, triangles)
    winding = winding_number(points, vertices, triangles)
    return np.where(winding > 0.5, -distance, distance)


def snapshot_oracle(points, snapshot):
    """Reference the experiment's stated min-composition through public data."""
    points = np.asarray(points, dtype=np.float64)
    candidates, obstacles = [], []
    for obj in snapshot.objects:
        if obj.role == "obstacle":
            obstacles.append(obj.geometry)
        elif isinstance(obj.geometry, Sphere):
            candidates.append(sphere_sdf(points, obj.pose.position_m, obj.geometry.radius_m))
        elif obj.role == "ground":
            distance, _ = triangle_distance(points, obj.geometry.vertices_m, obj.geometry.triangles)
            candidates.append(distance)
    candidates.append(mesh_signed_distance(points, obstacles))
    return np.min(np.asarray(candidates), axis=0)


def box_union_sdf(points, boxes):
    """Exact SDF of a small union of axis-aligned boxes, including interiors.

    Subdivide each box face at every box boundary. Retain rectangles whose
    outward-side midpoint is outside every box; these rectangles tile the true
    union boundary. Closest distance to their closed rectangles is exact.
    This independent construction exposes hidden-surface errors in min/mesh SDFs.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    boxes = [(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
             for lo, hi in boxes]
    rectangles = []
    for lo, hi in boxes:
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            cuts = [sorted({float(lo[k]), float(hi[k]),
                            *(float(np.clip(bound[k], lo[k], hi[k]))
                              for box in boxes for bound in box)}) for k in other]
            for sign, value in ((-1, lo[axis]), (1, hi[axis])):
                for u0, u1 in zip(cuts[0][:-1], cuts[0][1:]):
                    for v0, v1 in zip(cuts[1][:-1], cuts[1][1:]):
                        midpoint = (lo + hi) / 2
                        midpoint[axis] = value + sign * 1e-8
                        midpoint[other[0]], midpoint[other[1]] = (u0 + u1) / 2, (v0 + v1) / 2
                        if any(np.all(midpoint > low) and np.all(midpoint < high)
                               for low, high in boxes):
                            continue
                        lower, upper = lo.copy(), hi.copy()
                        lower[axis] = upper[axis] = value
                        lower[other[0]], upper[other[0]] = u0, u1
                        lower[other[1]], upper[other[1]] = v0, v1
                        rectangles.append((lower, upper))
    distance = np.min([np.linalg.norm(points - np.clip(points, lo, hi), axis=1)
                       for lo, hi in rectangles], axis=0)
    inside = np.any([np.all((points >= lo) & (points <= hi), axis=1)
                     for lo, hi in boxes], axis=0)
    return np.where(inside, -distance, distance)
