"""Explicit development scene and sensor facts, loaded once by the caller.

Sensor values in the checked-in JSON are development assumptions, not hardware
specifications. Mounts retain full sensor-FLU to body-FLU rigid transforms.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

from tof_tracker.geometry import Transform


TOF_DIRECTIONS = ("right", "front", "left", "rear")
_DEPTH_FIELDS = ("width", "height", "hfov_deg", "vfov_deg", "min_range_m", "max_range_m")
_MOUNT_FIELDS = ("position_m", "quaternion_xyzw")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive_number(value, name):
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _finite_vector(value, length, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    return tuple(_finite_number(component, name) for component in value)


@dataclass(frozen=True)
class DepthConfig:
    """Pinhole angular grid and radial range, in degrees and metres.

    Development rendering is limited to 4096 samples per axis and 1,048,576
    total cells. Pixel-centre versus continuous-zone measurement is selected
    by the sampling call. Width and height are positive integers, excluding booleans.
    """

    width: int
    height: int
    hfov_deg: float
    vfov_deg: float
    min_range_m: float
    max_range_m: float

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
                raise ValueError(f"{name} must be an integer in [1, 4096]")
        if self.width * self.height > 1_048_576:
            raise ValueError("depth resolution must not exceed 1,048,576 pixels")
        for name in ("hfov_deg", "vfov_deg"):
            number = _finite_number(getattr(self, name), name)
            if not 0 < number < 179:
                raise ValueError(f"{name} must be in (0, 179) degrees")
            object.__setattr__(self, name, number)
        minimum = _finite_number(self.min_range_m, "min_range_m")
        maximum = _finite_number(self.max_range_m, "max_range_m")
        if not 0 <= minimum < maximum:
            raise ValueError("ranges must satisfy 0 <= min_range_m < max_range_m")
        object.__setattr__(self, "min_range_m", minimum)
        object.__setattr__(self, "max_range_m", maximum)


@dataclass(frozen=True)
class MountConfig:
    """Full sensor-FLU to body-FLU pose, with position in body metres.

    Unit xyzw quaternions admit 1e-6 norm rounding and are normalized once.
    No axis is discarded and no attitude leveling is applied.
    """

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        position = _finite_vector(self.position_m, 3, "position_m")
        quaternion = _finite_vector(self.quaternion_xyzw, 4, "quaternion_xyzw")
        # Use the online core's rigid-transform validation without importing
        # scene: scene itself depends on this configuration module.
        Transform.from_quaternion("S", "B", position, quaternion)
        norm = math.hypot(*quaternion)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_xyzw", tuple(value / norm for value in quaternion))

    def as_transform(self, source_frame: str = "S", target_frame: str = "B") -> Transform:
        """Return the full sensor-to-body transform with explicit frame labels."""
        return Transform.from_quaternion(source_frame, target_frame,
                                         self.position_m, self.quaternion_xyzw)


@dataclass(frozen=True)
class SensorConfig:
    """Development sampling, mounts, and target visible-fraction threshold.

    ``tof_mounts`` always follows right/front/left/rear order. Visible fraction
    uses the target-only depth view as its denominator, not the full image.
    """

    tof: DepthConfig
    camera: DepthConfig
    tof_mounts: tuple[MountConfig, MountConfig, MountConfig, MountConfig]
    camera_mount: MountConfig
    min_visible_fraction: float

    def __post_init__(self):
        if not isinstance(self.tof, DepthConfig) or not isinstance(self.camera, DepthConfig):
            raise ValueError("tof and camera must be DepthConfig values")
        if (self.tof.width, self.tof.height) != (8, 8):
            raise ValueError("ToF depth must use an 8 by 8 grid")
        mounts = self.tof_mounts
        if (isinstance(mounts, (str, bytes)) or not isinstance(mounts, Sequence)
                or len(mounts) != 4 or not all(isinstance(mount, MountConfig) for mount in mounts)):
            raise ValueError("tof_mounts must contain four MountConfig values in right/front/left/rear order")
        if not isinstance(self.camera_mount, MountConfig):
            raise ValueError("camera_mount must be a MountConfig")
        threshold = _finite_number(self.min_visible_fraction, "min_visible_fraction")
        if not 0 <= threshold <= 1:
            raise ValueError("min_visible_fraction must be in [0, 1]")
        object.__setattr__(self, "tof_mounts", tuple(mounts))
        object.__setattr__(self, "min_visible_fraction", threshold)


@dataclass(frozen=True)
class SceneConfig:
    """Physical sphere radii and optional explicit development sensor facts."""

    vehicle_radius_m: float
    target_radius_m: float
    sensors: SensorConfig | None = None

    def __post_init__(self):
        for name in ("vehicle_radius_m", "target_radius_m"):
            object.__setattr__(self, name, _positive_number(getattr(self, name), name))
        if self.sensors is not None and not isinstance(self.sensors, SensorConfig):
            raise ValueError("sensors must be SensorConfig or None")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"configuration contains duplicate key {key!r}")
        result[key] = value
    return result


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(expected)}")


def _load_sensors(value):
    _exact_keys(value, ("tof", "camera", "tof_mounts", "camera_mount", "min_visible_fraction"), "sensors")
    depths = {}
    for name in ("tof", "camera"):
        _exact_keys(value[name], _DEPTH_FIELDS, f"sensors.{name}")
        depths[name] = DepthConfig(**value[name])
    _exact_keys(value["tof_mounts"], TOF_DIRECTIONS, "sensors.tof_mounts")
    mounts = []
    for name in TOF_DIRECTIONS:
        mount = value["tof_mounts"][name]
        _exact_keys(mount, _MOUNT_FIELDS, f"sensors.tof_mounts.{name}")
        mounts.append(MountConfig(**mount))
    _exact_keys(value["camera_mount"], _MOUNT_FIELDS, "sensors.camera_mount")
    return SensorConfig(depths["tof"], depths["camera"], tuple(mounts),
                        MountConfig(**value["camera_mount"]), value["min_visible_fraction"])


def load_config(path: str | Path) -> SceneConfig:
    """Read an explicit JSON path once; reject missing, extra and invalid facts.

    Legacy vehicle/target-only files remain valid, with sensors=None. If a
    sensors object is present, all sensor fields and all four mounts must be
    supplied; no omitted values or unknown fields are silently accepted.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) not in (
            {"vehicle", "target"}, {"vehicle", "target", "sensors"}):
        raise ValueError("configuration must contain vehicle, target, and optionally sensors")
    for name in ("vehicle", "target"):
        _exact_keys(value[name], ("radius_m",), name)
    sensors = _load_sensors(value["sensors"]) if "sensors" in value else None
    return SceneConfig(value["vehicle"]["radius_m"], value["target"]["radius_m"], sensors)
