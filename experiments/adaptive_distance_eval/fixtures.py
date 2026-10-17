"""Small closed mesh fixtures and independent float64 geometric references."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from devtools.mesh import MeshObject
from devtools.scene import GeometrySnapshot, SceneObject


@dataclass(frozen=True)
class Box:
    center: tuple
    half: tuple
    rotation: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))

    def mesh(self, name):
        h = np.asarray(self.half)
        vertices = np.array(((-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
                             (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1))) * h
        vertices = vertices @ np.asarray(self.rotation).T + self.center
        triangles = ((0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
                     (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7))
        return MeshObject(name, tuple(map(tuple, vertices.tolist())), triangles, "obstacle")

    def distance_gradient(self, points):
        r = np.asarray(self.rotation)
        local = (np.asarray(points, dtype=np.float64) - self.center) @ r
        q = np.abs(local) - self.half
        positive = np.maximum(q, 0.)
        norm = np.linalg.norm(positive, axis=1)
        inside = np.max(q, axis=1) <= 0
        distance = norm + np.minimum(np.max(q, axis=1), 0.)
        gradient = np.sign(local) * positive / np.maximum(norm[:, None], 1e-100)
        closest_axis = np.argmax(q, axis=1)
        gradient[inside] = 0
        ii = np.flatnonzero(inside)
        gradient[ii, closest_axis[ii]] = np.sign(local[ii, closest_axis[ii]])
        # At interior medial sheets no unique derivative exists. The gradient
        # accuracy comparison explicitly excludes a 2 mm band around ties.
        ordered = np.sort(q, axis=1)
        same_axis_unique = np.abs(local[np.arange(len(local)), closest_axis]) > .002
        unique = (~inside | ((ordered[:, -1] - ordered[:, -2] > .002) & same_axis_unique)) & (np.abs(distance) > 1e-6)
        return distance, gradient @ r.T, unique


def cases():
    roll, pitch, yaw = .31, -.22, .43
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((cy,-sy,0),(sy,cy,0),(0,0,1))) @ np.array(((cp,0,sp),(0,1,0),(-sp,0,cp))) @ np.array(((1,0,0),(0,cr,-sr),(0,sr,cr)))
    return {
        "box": (Box((.08,-.06,.02), (.27,.21,.32)),),
        "rotated_box": (Box((.03,-.04,.01), (.29,.19,.26), tuple(map(tuple, rotation))),),
        "thin_box": (Box((.031,-.02,.03), (.009,.31,.35)),),
        "ridge": (Box((-.62,0,0), (.08,.69,.69)), Box((.62,0,0), (.08,.69,.69))),
    }


def write_case(directory, name, boxes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    objects = tuple(SceneObject(f"map/{name}_{i}", "obstacle", box.mesh(f"{name}_{i}"))
                    for i, box in enumerate(boxes))
    geometry = GeometrySnapshot(objects, 0., ((-.8,.8),(-.8,.8),(-.8,.8)))
    path = directory / f"{name}.obj"
    lines, offset = ["# metre; experiment closed solids only, no ground or dynamic objects"], 0
    for obj in geometry.objects:  # public snapshot data, no scene internals
        mesh = obj.geometry
        lines.append(f"o {mesh.name}")
        lines.extend("v " + " ".join(format(x, ".17g") for x in v) for v in mesh.vertices_m)
        lines.extend("f " + " ".join(str(i + 1 + offset) for i in f) for f in mesh.triangles)
        offset += len(mesh.vertices_m)
    path.write_text("\n".join(lines) + "\n")
    return path, geometry


def reference(points, boxes):
    outputs = [b.distance_gradient(points) for b in boxes]
    values = np.stack([o[0] for o in outputs], axis=1)
    winner = np.argmin(values, axis=1)
    gradient = np.stack([o[1] for o in outputs], axis=1)[np.arange(len(points)), winner]
    unique = np.stack([o[2] for o in outputs], axis=1)[np.arange(len(points)), winner]
    if len(boxes) > 1:
        sorted_values = np.sort(values, axis=1)
        unique &= sorted_values[:, 1] - sorted_values[:, 0] > .002
    # These fixture solids are disjoint; min is the exact union SDF here.
    return values.min(1), gradient, unique


def exhaustive_triangle_distance(points, geometry):
    """Independent unsigned point-to-triangle enumeration, float64 NumPy."""
    points = np.asarray(points, dtype=np.float64)
    best = np.full(len(points), np.inf)
    for obj in geometry.objects:
        vertices = np.asarray(obj.geometry.vertices_m)
        for indices in obj.geometry.triangles:
            a, b, c = vertices[list(indices)]
            ab, ac, ap = b-a, c-a, points-a
            normal = np.cross(ab, ac)
            n2 = np.dot(normal, normal)
            v = np.cross(ap, ac) @ normal / n2
            w = np.cross(ab, ap) @ normal / n2
            face = (v >= 0) & (w >= 0) & (v+w <= 1)
            plane = np.abs(ap @ normal) / np.sqrt(n2)
            best = np.minimum(best, np.where(face, plane, np.inf))
            for start, end in ((a,b),(b,c),(c,a)):
                edge = end-start
                t = np.clip((points-start) @ edge / np.dot(edge, edge), 0., 1.)
                best = np.minimum(best, np.linalg.norm(points-start-t[:,None]*edge, axis=1))
    return best
