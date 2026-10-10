"""Explicit physical facts for development scenes, loaded once by the caller."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


@dataclass(frozen=True)
class SceneConfig:
    """Sphere radii in metres; these are physical sizes, without safety margins."""

    vehicle_radius_m: float
    target_radius_m: float

    def __post_init__(self):
        for name in ("vehicle_radius_m", "target_radius_m"):
            object.__setattr__(self, name, _positive_number(getattr(self, name), name))


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


def load_config(path: str | Path) -> SceneConfig:
    """Read an explicit JSON path once; reject missing, extra and invalid facts.

    JSON currently has only vehicle.radius_m and target.radius_m. Future sensor
    facts belong in this same file when that capability is implemented.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    _exact_keys(value, ("vehicle", "target"), "configuration")
    for name in ("vehicle", "target"):
        _exact_keys(value[name], ("radius_m",), name)
    return SceneConfig(value["vehicle"]["radius_m"], value["target"]["radius_m"])
