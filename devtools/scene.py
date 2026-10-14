"""Immutable development geometry: a complete background and two spheres.

World positions are metres. Body axes follow FLU (+X forward, +Y left, +Z up).
Geometry is ground truth for development backends, not online observations.
This module does not advance dynamics or infer occupied/free space from points.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import math
from pathlib import Path

from tof_tracker.geometry import Transform

from .config import SceneConfig, _positive_number


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _finite_vector(value, length, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{name} must contain exactly {length} finite numbers")
        try:
            number = float(component)
        except OverflowError as exc:
            raise ValueError(f"{name} must contain finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class Pose:
    """Body-to-world rigid pose, with a unit quaternion in (x, y, z, w) order.

    Norm must be within 1e-6 of one; accepted rounding error is normalized once.
    This numerical correction preserves attitude and permits float32 inputs.
    Full roll, pitch and yaw are retained; no horizontal leveling is applied.
    """

    position_m: Vector3
    quaternion_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        position = _finite_vector(self.position_m, 3, "position_m")
        quaternion = _finite_vector(self.quaternion_xyzw, 4, "quaternion_xyzw")
        norm = math.hypot(*quaternion)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("quaternion_xyzw must be a unit quaternion (norm tolerance 1e-6)")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_xyzw", tuple(q / norm for q in quaternion))

    def as_transform(self, source_frame: str = "B", target_frame: str = "W") -> Transform:
        """Adapt this body-to-world snapshot to the shared numerical transform.

        Frame names label the body's FLU frame and the scene's world frame;
        changing a label does not change the pose's body-to-world meaning.
        """
        return Transform.from_quaternion(source_frame, target_frame,
                                         self.position_m, self.quaternion_xyzw)

    def body_to_world(self, point_b_m: Vector3) -> Vector3:
        """Transform one body-frame point: p_world = R * p_body + position_m."""
        point = _finite_vector(point_b_m, 3, "point_b_m")
        transformed = self.as_transform().apply_points(point)
        result = tuple(float(value) for value in transformed)
        return _finite_vector(result, 3, "transformed point")

    def world_to_body(self, point_w_m: Vector3) -> Vector3:
        """Transform one world-frame point through the inverse body pose."""
        point = _finite_vector(point_w_m, 3, "point_w_m")
        transformed = self.as_transform().inverse().apply_points(point)
        result = tuple(float(value) for value in transformed)
        return _finite_vector(result, 3, "transformed point")


@dataclass(frozen=True)
class Sphere:
    """A physical sphere centered at pose.position_m, with radius in metres."""

    pose: Pose
    radius_m: float

    def __post_init__(self):
        if not isinstance(self.pose, Pose):
            raise ValueError("sphere pose must be a Pose")
        object.__setattr__(self, "radius_m", _positive_number(self.radius_m, "radius_m"))


@dataclass(frozen=True)
class PointCloudMap:
    """All original XYZ points, without display sampling, voxelization or filling.

    resolution_m and size_xyz_m are optional source metadata, not a declaration
    of occupied volume. bounds_m are the actual point extrema along X, Y and Z.
    """

    points_m: tuple[Vector3, ...]
    source: Path
    comments: tuple[str, ...] = ()
    resolution_m: float | None = None
    size_xyz_m: Vector3 | None = None
    bounds_m: tuple[tuple[float, float], ...] = field(init=False)

    def __post_init__(self):
        points = tuple(_finite_vector(point, 3, "background point") for point in self.points_m)
        if not points:
            raise ValueError("background point cloud must not be empty")
        comments = tuple(self.comments)
        if not all(isinstance(comment, str) for comment in comments):
            raise ValueError("background comments must be strings")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "comments", comments)
        if self.resolution_m is not None:
            object.__setattr__(self, "resolution_m", _positive_number(self.resolution_m, "resolution_m"))
        if self.size_xyz_m is not None:
            sizes = _finite_vector(self.size_xyz_m, 3, "size_xyz_m")
            object.__setattr__(self, "size_xyz_m", tuple(_positive_number(v, "size_xyz_m") for v in sizes))
        object.__setattr__(self, "bounds_m", tuple((min(p[k] for p in points), max(p[k] for p in points))
                                                   for k in range(3)))

    @property
    def count(self) -> int:
        return len(self.points_m)

    @classmethod
    def from_ply(cls, path: str | Path) -> "PointCloudMap":
        """Load complete metre XYZ data from the project's ASCII PLY subset."""
        path = Path(path)
        with path.open(encoding="ascii") as stream:
            if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
                raise ValueError("only format ascii 1.0 PLY is supported")
            count, properties, comments = None, [], []
            for line_number in range(3, 259):
                line = stream.readline()
                if not line or len(line) > 16_384:
                    raise ValueError("PLY header lacks end_header or contains an overlong line")
                fields = line.strip().split()
                if not fields:
                    raise ValueError("PLY header contains an empty line")
                if fields == ["end_header"]:
                    break
                if fields[0] == "comment":
                    comments.append(line.strip()[8:])
                elif fields[:2] == ["element", "vertex"] and len(fields) == 3 and count is None:
                    count = int(fields[2])
                elif fields[0] == "property" and count is not None and len(fields) == 3:
                    if fields[1] not in {"float", "double", "float32", "float64"}:
                        raise ValueError("XYZ must use float or double scalar properties")
                    properties.append(fields[2])
                else:
                    raise ValueError(f"unsupported PLY header field on line {line_number}")
            else:
                raise ValueError("PLY header too long or missing end_header")
            if count is None or count <= 0 or sorted(properties) != ["x", "y", "z"]:
                raise ValueError("PLY must declare nonempty vertex data with exactly x/y/z properties")

            resolution, sizes = None, None
            for comment in comments:
                fields = comment.split()
                if fields[:1] == ["unit"] and fields != ["unit", "metre"]:
                    raise ValueError("PLY unit must be metre; coordinates are not converted")
                if fields[:1] == ["size_xyz"]:
                    if len(fields) != 6 or fields[4] != "resolution" or sizes is not None:
                        raise ValueError("size_xyz / resolution metadata is malformed or duplicated")
                    sizes = tuple(_positive_number(float(v), "size_xyz") for v in fields[1:4])
                    resolution = _positive_number(float(fields[5]), "resolution")
            order = [properties.index(axis) for axis in "xyz"]
            points = []
            for index in range(count):
                fields = stream.readline().split()
                if len(fields) != 3:
                    raise ValueError(f"vertex {index + 1} is missing or does not contain three values")
                point = tuple(float(fields[k]) for k in order)
                if not all(math.isfinite(v) for v in point):
                    raise ValueError(f"vertex {index + 1} contains nonfinite coordinates")
                points.append(point)
            if any(line.strip() for line in stream):
                raise ValueError("vertex data exceeds the PLY declared count")
        return cls(tuple(points), path, tuple(comments), resolution, sizes)


@dataclass(frozen=True)
class Scene:
    """An immutable physical snapshot, before sensing or distance-field queries."""

    background: PointCloudMap
    vehicle: Sphere
    target: Sphere

    def __post_init__(self):
        if not isinstance(self.background, PointCloudMap):
            raise ValueError("background must be a PointCloudMap")
        if not isinstance(self.vehicle, Sphere) or not isinstance(self.target, Sphere):
            raise ValueError("vehicle and target must be Sphere instances")

    @classmethod
    def from_ply(cls, path: str | Path, config: SceneConfig, vehicle_pose: Pose,
                 target_position_m: Vector3) -> "Scene":
        """Combine a background asset, explicit sphere sizes and world state."""
        if not isinstance(config, SceneConfig):
            raise ValueError("config must be a SceneConfig")
        vehicle = Sphere(vehicle_pose, config.vehicle_radius_m)
        target = Sphere(Pose(target_position_m), config.target_radius_m)
        return cls(PointCloudMap.from_ply(path), vehicle, target)

    def with_vehicle_pose(self, pose: Pose) -> "Scene":
        """Replace only vehicle state, retaining the same background and target."""
        return replace(self, vehicle=Sphere(pose, self.vehicle.radius_m))

    def with_target_position(self, position_m: Vector3) -> "Scene":
        """Replace only target center; spheres need no separate target attitude."""
        pose = replace(self.target.pose, position_m=position_m)
        return replace(self, target=Sphere(pose, self.target.radius_m))
