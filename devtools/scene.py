"""Immutable development scenes and public, filtered world geometry snapshots.

World positions are metres. Body axes follow FLU (+X forward, +Y left, +Z up).
Geometry is ground truth for development backends, not online observations.
This module does not advance dynamics or compute occupancy.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from pathlib import Path

from tof_tracker.geometry import Transform

from .config import SceneConfig, _positive_number
from .mesh import MeshMap, MeshObject


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _reference_time(value):
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError("t_ref must be finite seconds in the caller's clock domain, or None")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("t_ref must be finite seconds") from exc
    if not math.isfinite(result):
        raise ValueError("t_ref must be finite seconds")
    return result


def _object_id(value):
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("object_id must be one nonempty string token")
    return value


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


_WORLD_IDENTITY_POSE = Pose((0.0, 0.0, 0.0))


@dataclass(frozen=True)
class SceneObject:
    """Stable identity and role attached to existing immutable geometry.

    Mesh vertices already lie in W, so their pose is the world identity.
    A sphere retains its world center and body-to-world attitude in geometry.pose.
    Geometry is shared; this record does not convert spheres into triangles.
    """

    object_id: str
    role: str
    geometry: MeshObject | Sphere

    def __post_init__(self):
        _object_id(self.object_id)
        if isinstance(self.geometry, MeshObject):
            valid_roles = (self.geometry.kind,)
        elif isinstance(self.geometry, Sphere):
            valid_roles = ("vehicle", "target")
        else:
            raise ValueError("geometry must be a MeshObject or Sphere")
        if not isinstance(self.role, str) or self.role not in valid_roles:
            raise ValueError("object role must match its geometry")

    @property
    def pose(self) -> Pose:
        return self.geometry.pose if isinstance(self.geometry, Sphere) else _WORLD_IDENTITY_POSE


@dataclass(frozen=True)
class GeometrySnapshot:
    """Selected world geometry at a caller-supplied reference time.

    map_bounds_m describes the declared map domain, not the extrema of vertices.
    None means the asset has no declared domain. Selection never
    changes this metadata or decides visibility, occupancy or collision policy.
    """

    objects: tuple[SceneObject, ...]
    t_ref: float | None
    map_bounds_m: tuple[tuple[float, float], ...] | None

    def __post_init__(self):
        if isinstance(self.objects, (str, bytes, Mapping)):
            raise ValueError("objects must contain SceneObject instances")
        try:
            objects = tuple(self.objects)
        except TypeError as exc:
            raise ValueError("objects must contain SceneObject instances") from exc
        if any(not isinstance(obj, SceneObject) for obj in objects):
            raise ValueError("objects must contain SceneObject instances")
        if len({obj.object_id for obj in objects}) != len(objects):
            raise ValueError("snapshot object IDs must be unique")
        bounds = self.map_bounds_m
        if bounds is not None:
            if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 3:
                raise ValueError("map_bounds_m must contain three finite increasing coordinate intervals")
            bounds = tuple(_finite_vector(interval, 2, "map bounds interval") for interval in bounds)
            if any(low >= high for low, high in bounds):
                raise ValueError("map bounds intervals must be increasing")
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "t_ref", _reference_time(self.t_ref))
        object.__setattr__(self, "map_bounds_m", bounds)

    @property
    def frame(self) -> str:
        return "W"

    @property
    def unit(self) -> str:
        return "metre"

    def get_object(self, object_id: str) -> SceneObject:
        """Return the selected object with this ID; reject unknown IDs."""
        _object_id(object_id)
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise ValueError(f"unknown object ID {object_id!r} in this geometry snapshot")


@dataclass(frozen=True)
class Scene:
    """Immutable world facts, before sensing or distance-field queries.

    t_ref is finite seconds in the caller's clock domain; None means unknown.
    Replacing a pose or position retains this time and does not advance a clock.
    """

    background: MeshMap
    vehicle: Sphere
    target: Sphere
    t_ref: float | None = None
    _geometry: GeometrySnapshot = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.background, MeshMap):
            raise ValueError("background must be a MeshMap")
        if not isinstance(self.vehicle, Sphere) or not isinstance(self.target, Sphere):
            raise ValueError("vehicle and target must be Sphere instances")
        object.__setattr__(self, "t_ref", _reference_time(self.t_ref))
        objects = tuple(SceneObject("map/" + obj.name, obj.kind, obj) for obj in self.background.objects)
        objects += (SceneObject("vehicle", "vehicle", self.vehicle),
                    SceneObject("target", "target", self.target))
        sizes = self.background.size_xyz_m
        bounds = None if sizes is None else ((-sizes[0]/2, sizes[0]/2),
                                             (-sizes[1]/2, sizes[1]/2), (0.0, sizes[2]))
        object.__setattr__(self, "_geometry", GeometrySnapshot(objects, self.t_ref, bounds))

    @classmethod
    def from_obj(cls, path: str | Path, config: SceneConfig, vehicle_pose: Pose,
                 target_position_m: Vector3, *, t_ref: float | None = None) -> "Scene":
        """Load a complete mesh background and explicit world state."""
        if not isinstance(config, SceneConfig):
            raise ValueError("config must be a SceneConfig")
        vehicle = Sphere(vehicle_pose, config.vehicle_radius_m)
        target = Sphere(Pose(target_position_m), config.target_radius_m)
        return cls(MeshMap.from_obj(path), vehicle, target, t_ref)

    def get_geometry(self, *, include_ids=None, exclude_ids=()) -> GeometrySnapshot:
        """Return geometry selected by exact object IDs, retaining source order.

        None includes all IDs; an empty include iterable selects none. Exclusion
        takes precedence. Both filters accept ID iterables, reject strings and
        unknown IDs or mappings, and affect only this return value. Entities and shapes are
        shared with the unfiltered snapshot; this is no visibility computation.
        """
        known_ids = {obj.object_id for obj in self._geometry.objects}

        def checked_ids(values, name):
            if isinstance(values, (str, bytes, Mapping)):
                raise ValueError(f"{name} must be an iterable of object ID strings, not a string or mapping")
            try:
                iterator = iter(values)
            except TypeError as exc:
                raise ValueError(f"{name} must be an iterable of object ID strings") from exc
            selected = set()
            for value in iterator:
                _object_id(value)
                if value not in known_ids:
                    raise ValueError(f"unknown object ID {value!r} in {name}")
                selected.add(value)
            return selected

        included = known_ids if include_ids is None else checked_ids(include_ids, "include_ids")
        excluded = checked_ids(exclude_ids, "exclude_ids")
        if included == known_ids and not excluded:
            return self._geometry
        objects = tuple(obj for obj in self._geometry.objects
                        if obj.object_id in included and obj.object_id not in excluded)
        return GeometrySnapshot(objects, self.t_ref, self._geometry.map_bounds_m)

    def with_vehicle_pose(self, pose: Pose) -> "Scene":
        """Replace only vehicle state, retaining the same background and target."""
        return replace(self, vehicle=Sphere(pose, self.vehicle.radius_m))

    def with_target_position(self, position_m: Vector3) -> "Scene":
        """Replace only target center; spheres need no separate target attitude."""
        pose = replace(self.target.pose, position_m=position_m)
        return replace(self, target=Sphere(pose, self.target.radius_m))
