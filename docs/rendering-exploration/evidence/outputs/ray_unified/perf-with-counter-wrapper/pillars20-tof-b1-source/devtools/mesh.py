"""Complete, immutable mesh-map assets in world metres.

The OBJ subset has object-local vertex blocks and triangle faces using positive
file-wide indices. Obstacles are connected, closed, outward-oriented surfaces;
the separate ground is a +Z rectangle at z=0. Overlapping obstacle objects remain
separate entities. This module neither samples surfaces nor defines an ESDF.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path

from .config import _positive_number


Vector3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


def _vertex(value):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("mesh vertices must contain exactly three finite numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError("mesh vertices must contain exactly three finite numbers")
    try:
        result = tuple(float(v) for v in value)
    except OverflowError as exc:
        raise ValueError("mesh vertex coordinates must be finite") from exc
    if not all(math.isfinite(v) for v in result):
        raise ValueError("mesh vertex coordinates must be finite")
    return result


def _subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


@dataclass(frozen=True)
class MeshObject:
    """One entity's vertices and zero-based local triangle indices.

    Ground is an open plane. Each obstacle is independently closed; closure and
    orientation do not imply that the complete map is a nonoverlapping surface.
    """

    name: str
    vertices_m: tuple[Vector3, ...]
    triangles: tuple[Triangle, ...]
    kind: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or any(c.isspace() for c in self.name):
            raise ValueError("mesh object name must be one nonempty token")
        if (not isinstance(self.kind, str) or self.kind not in {"ground", "obstacle"}
                or (self.name == "ground") != (self.kind == "ground")):
            raise ValueError("ground must be named ground; other objects must be obstacles")
        vertices = tuple(_vertex(v) for v in self.vertices_m)
        if len(vertices) < 3 or not self.triangles:
            raise ValueError("mesh object must have vertices and triangle faces")
        triangles, normals, edges, used, face_keys = [], [], {}, set(), set()
        for face_index, face in enumerate(self.triangles):
            if (not isinstance(face, Sequence) or isinstance(face, (str, bytes)) or len(face) != 3
                    or any(isinstance(i, bool) or not isinstance(i, int) for i in face)):
                raise ValueError("mesh faces must contain exactly three integer indices")
            face = tuple(face)
            if len(set(face)) != 3 or min(face) < 0 or max(face) >= len(vertices):
                raise ValueError("mesh face indices must reference three distinct local vertices")
            key = tuple(sorted(face))
            if key in face_keys:
                raise ValueError("mesh contains a duplicate triangle")
            face_keys.add(key)
            a, b, c = (vertices[i] for i in face)
            normal = _cross(_subtract(b, a), _subtract(c, a))
            if not all(math.isfinite(n) for n in normal) or math.hypot(*normal) == 0:
                raise ValueError("mesh triangle must be finite and nondegenerate")
            triangles.append(face)
            normals.append(normal)
            used.update(face)
            for start, end in zip(face, (*face[1:], face[0])):
                edge = (min(start, end), max(start, end))
                edges.setdefault(edge, []).append((face_index, 1 if start < end else -1))
        if len(used) != len(vertices):
            raise ValueError("mesh vertices must all be used by faces")

        if self.kind == "ground":
            xs, ys = sorted({v[0] for v in vertices}), sorted({v[1] for v in vertices})
            if (len(vertices) != 4 or len(triangles) != 2 or len(xs) != 2 or len(ys) != 2
                    or any(v[2] != 0 for v in vertices)
                    or {(v[0], v[1]) for v in vertices} != {(x, y) for x in xs for y in ys}
                    or any(normal[2] <= 0 for normal in normals)):
                raise ValueError("ground must be one +Z rectangle at z=0, with four vertices and two triangles")
            if (sum(len(incidence) == 1 for incidence in edges.values()) != 4
                    or sum(len(incidence) == 2 for incidence in edges.values()) != 1
                    or any(len(incidence) == 2 and sum(direction for _, direction in incidence) != 0
                           for incidence in edges.values())):
                raise ValueError("ground faces must consistently tile the rectangle")
        else:
            adjacency = [set() for _ in triangles]
            for incidence in edges.values():
                if len(incidence) != 2 or sum(direction for _, direction in incidence) != 0:
                    raise ValueError("obstacle mesh must be closed with consistent face winding")
                (first, _), (second, _) = incidence
                adjacency[first].add(second)
                adjacency[second].add(first)
            pending, connected = [0], set()
            while pending:
                current = pending.pop()
                if current not in connected:
                    connected.add(current)
                    pending.extend(adjacency[current] - connected)
            if len(connected) != len(triangles):
                raise ValueError("obstacle mesh must be one connected surface")
            # Shift near the object before summation to avoid cancellation far from the world origin.
            origin = vertices[0]
            volume_terms = []
            for face in triangles:
                a, b, c = (_subtract(vertices[i], origin) for i in face)
                cross = _cross(b, c)
                volume_terms.append(sum(a[k] * cross[k] for k in range(3)))
            volume6 = math.fsum(volume_terms)
            if not math.isfinite(volume6) or volume6 <= 0:
                raise ValueError("obstacle mesh must enclose nonzero volume with outward winding")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", tuple(triangles))


@dataclass(frozen=True)
class MeshMap:
    """A complete mesh map containing one ground and independent obstacles."""

    source: Path
    objects: tuple[MeshObject, ...]
    comments: tuple[str, ...] = ()
    size_xyz_m: Vector3 | None = None
    bounds_m: tuple[tuple[float, float], ...] = field(init=False)

    def __post_init__(self):
        objects, comments = tuple(self.objects), tuple(self.comments)
        if not objects or any(not isinstance(obj, MeshObject) for obj in objects):
            raise ValueError("mesh map must contain MeshObject instances")
        if len({obj.name for obj in objects}) != len(objects):
            raise ValueError("mesh object names must be unique")
        ground = [obj for obj in objects if obj.kind == "ground"]
        if len(ground) != 1:
            raise ValueError("mesh map must contain exactly one ground plane")
        if any(not isinstance(comment, str) for comment in comments):
            raise ValueError("mesh comments must be strings")
        if self.size_xyz_m is not None:
            sizes = tuple(_positive_number(v, "size_xyz_m") for v in _vertex(self.size_xyz_m))
            expected = {(-sizes[0]/2, -sizes[1]/2, 0), (sizes[0]/2, -sizes[1]/2, 0),
                        (sizes[0]/2, sizes[1]/2, 0), (-sizes[0]/2, sizes[1]/2, 0)}
            if set(ground[0].vertices_m) != expected:
                raise ValueError("ground must span the complete size_xyz XY rectangle")
            for obj in objects:
                if any(abs(v[0]) > sizes[0]/2 or abs(v[1]) > sizes[1]/2 or not 0 <= v[2] <= sizes[2]
                       for v in obj.vertices_m):
                    raise ValueError("mesh vertices exceed size_xyz map bounds")
            object.__setattr__(self, "size_xyz_m", sizes)
        vertices = tuple(v for obj in objects for v in obj.vertices_m)
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "comments", comments)
        object.__setattr__(self, "bounds_m", tuple((min(v[k] for v in vertices), max(v[k] for v in vertices))
                                                  for k in range(3)))

    @property
    def vertex_count(self) -> int:
        return sum(len(obj.vertices_m) for obj in self.objects)

    @property
    def triangle_count(self) -> int:
        return sum(len(obj.triangles) for obj in self.objects)

    @classmethod
    def from_obj(cls, path: str | Path) -> "MeshMap":
        """Read only o/v/f/comments; reject foreign or malformed OBJ features.

        Face indices must refer to the current object's preceding vertex block.
        Coordinates are metres; optional unit and size_xyz comments are checked.
        """
        path = Path(path)
        objects, comments, names = [], [], set()
        name, vertices, triangles, offset, total_vertices = None, [], [], 0, 0
        sizes, unit_seen, faces_started = None, False, False

        def finish_object():
            if name is not None:
                objects.append(MeshObject(name, tuple(vertices), tuple(triangles),
                                          "ground" if name == "ground" else "obstacle"))

        with path.open(encoding="ascii") as stream:
            for line_number, line in enumerate(stream, 1):
                if len(line) > 16_384:
                    raise ValueError(f"OBJ line {line_number} is too long")
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    comment = stripped[1:].strip()
                    fields = comment.split()
                    comments.append(comment)
                    if fields[:1] == ["unit"]:
                        if fields != ["unit", "metre"] or unit_seen:
                            raise ValueError("OBJ unit must be metre and declared at most once")
                        unit_seen = True
                    elif fields[:1] == ["size_xyz"]:
                        if len(fields) != 4 or sizes is not None:
                            raise ValueError("OBJ size_xyz metadata is malformed or duplicated")
                        sizes = tuple(_positive_number(float(v), "size_xyz") for v in fields[1:])
                    continue
                fields = stripped.split()
                if fields[0] == "o" and len(fields) == 2:
                    finish_object()
                    name = fields[1]
                    if name in names:
                        raise ValueError("OBJ object names must be unique")
                    names.add(name)
                    vertices, triangles, offset, faces_started = [], [], total_vertices, False
                elif fields[0] == "v" and len(fields) == 4 and name is not None and not faces_started:
                    vertices.append(_vertex(tuple(float(v) for v in fields[1:])))
                    total_vertices += 1
                elif fields[0] == "f" and len(fields) == 4 and name is not None:
                    faces_started = True
                    if any(not value.isascii() or not value.isdecimal() for value in fields[1:]):
                        raise ValueError("OBJ faces require plain positive integer indices")
                    indices = tuple(int(value) for value in fields[1:])
                    if any(not offset < index <= total_vertices for index in indices):
                        raise ValueError("OBJ faces must refer to the current object's preceding vertices")
                    triangles.append(tuple(index - offset - 1 for index in indices))
                else:
                    raise ValueError(f"unsupported or misplaced OBJ record on line {line_number}")
        finish_object()
        return cls(path, tuple(objects), tuple(comments), sizes)
