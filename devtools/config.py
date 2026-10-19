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
_DEPTH_FIELDS = ("width", "height", "hfov_deg", "vfov_deg", "min_range_m", "max_range_m",
                 "samples_per_axis")
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

    Each cell contains samples_per_axis squared rays at uniform subcell
    centres on the image plane. Width and height describe output cells.
    """

    width: int
    height: int
    hfov_deg: float
    vfov_deg: float
    min_range_m: float
    max_range_m: float
    samples_per_axis: int

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
                raise ValueError(f"{name} must be an integer in [1, 4096]")
        if self.width * self.height > 1_048_576:
            raise ValueError("depth resolution must not exceed 1,048,576 pixels")
        if (isinstance(self.samples_per_axis, bool)
                or not isinstance(self.samples_per_axis, int)
                or not 1 <= self.samples_per_axis <= 16):
            raise ValueError("samples_per_axis must be an integer in [1, 16]")
        if self.width * self.height * self.samples_per_axis ** 2 > 16_777_216:
            raise ValueError("a view must not exceed 16,777,216 rays")
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
class RigConfig:
    """Outer development recipe: device settings and a separate mount layout.

    Only rig orchestration consumes mounts. Sensor instances receive their own
    DepthConfig and reduction parameters, never this aggregate recipe.
    """

    tof: DepthConfig
    camera: DepthConfig
    tof_mounts: tuple[MountConfig, MountConfig, MountConfig, MountConfig]
    camera_mount: MountConfig
    tof_depth_quantile: float
    tof_signal_floor: float

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
        object.__setattr__(self, "tof_mounts", tuple(mounts))
        quantile = _finite_number(self.tof_depth_quantile, "tof_depth_quantile")
        if not 0 <= quantile <= 1:
            raise ValueError("tof_depth_quantile must be in [0, 1]")
        object.__setattr__(self, "tof_depth_quantile", quantile)
        object.__setattr__(self, "tof_signal_floor", _positive_number(self.tof_signal_floor,
                                                                     "tof_signal_floor"))


@dataclass(frozen=True)
class SamplingRuntimeConfig:
    """Sampling work-set target as a fraction of total device memory.

    Execution also respects currently available memory. This is a chunking
    budget, not a driver-enforced limit on training or other GPU users.
    """

    max_gpu_memory_fraction: float

    def __post_init__(self):
        value = _finite_number(self.max_gpu_memory_fraction, "max_gpu_memory_fraction")
        if not 0 < value <= 1:
            raise ValueError("max_gpu_memory_fraction must be in (0, 1]")
        object.__setattr__(self, "max_gpu_memory_fraction", value)


@dataclass(frozen=True)
class PerceptionConfig:
    """Development environment's simulated external target recognition."""

    min_visible_fraction: float

    def __post_init__(self):
        value = _finite_number(self.min_visible_fraction, "min_visible_fraction")
        if not 0 <= value <= 1:
            raise ValueError("min_visible_fraction must be in [0, 1]")
        object.__setattr__(self, "min_visible_fraction", value)


@dataclass(frozen=True)
class SceneConfig:
    """Explicit physical scene, sensor, sampling and recognition configuration."""

    vehicle_radius_m: float
    target_radius_m: float
    sensors: RigConfig
    sampling_runtime: SamplingRuntimeConfig
    perception: PerceptionConfig

    def __post_init__(self):
        for name in ("vehicle_radius_m", "target_radius_m"):
            object.__setattr__(self, name, _positive_number(getattr(self, name), name))
        if not isinstance(self.sensors, RigConfig):
            raise ValueError("sensors must be RigConfig")
        if not isinstance(self.sampling_runtime, SamplingRuntimeConfig):
            raise ValueError("sampling_runtime must be SamplingRuntimeConfig")
        if not isinstance(self.perception, PerceptionConfig):
            raise ValueError("perception must be PerceptionConfig")


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
    _exact_keys(value, ("tof", "camera", "tof_mounts", "camera_mount",
                        "tof_depth_quantile", "tof_signal_floor"), "sensors")
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
    return RigConfig(depths["tof"], depths["camera"], tuple(mounts),
                        MountConfig(**value["camera_mount"]),
                        value["tof_depth_quantile"], value["tof_signal_floor"])


def load_config(path: str | Path) -> SceneConfig:
    """Read an explicit JSON path once; reject missing, extra and invalid facts."""
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    _exact_keys(value, ("vehicle", "target", "sensors", "sampling_runtime", "perception"), "configuration")
    for name in ("vehicle", "target"):
        _exact_keys(value[name], ("radius_m",), name)
    sensors = _load_sensors(value["sensors"])
    for key, fields in (("sampling_runtime", ("max_gpu_memory_fraction",)),
                        ("perception", ("min_visible_fraction",))):
        _exact_keys(value[key], fields, key)
    return SceneConfig(value["vehicle"]["radius_m"], value["target"]["radius_m"], sensors,
                       SamplingRuntimeConfig(**value["sampling_runtime"]),
                       PerceptionConfig(**value["perception"]))
