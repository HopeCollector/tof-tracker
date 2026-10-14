"""Verify full scene loading, strict facts, immutable updates and 3D geometry."""

import argparse
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile

from devtools.config import SceneConfig, load_config
from devtools.scene import PointCloudMap, Pose, Scene, Sphere


def ply(points, comments=("unit metre", "size_xyz 8 6 3 resolution 0.2"), properties="xyz"):
    header = ["ply", "format ascii 1.0", *("comment " + value for value in comments),
              f"element vertex {len(points)}", *("property double " + axis for axis in properties),
              "end_header"]
    rows = [" ".join(str(point["xyz".index(axis)]) for axis in properties) for point in points]
    return "\n".join(header + rows) + "\n"


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def close_vector(a, b):
    assert len(a) == len(b)
    for actual, expected in zip(a, b):
        close(actual, expected)


def rejected(action, error=ValueError):
    try:
        action()
    except error:
        return
    raise AssertionError("invalid input was accepted")


def euler_quaternion(roll, pitch, yaw):
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


def euler_point(point, roll, pitch, yaw):
    """Independent reference: sequential elementary Rx, Ry and Rz rotations."""
    x, y, z = point
    y, z = math.cos(roll)*y - math.sin(roll)*z, math.sin(roll)*y + math.cos(roll)*z
    x, z = math.cos(pitch)*x + math.sin(pitch)*z, -math.sin(pitch)*x + math.cos(pitch)*z
    return (math.cos(yaw)*x - math.sin(yaw)*y, math.sin(yaw)*x + math.cos(yaw)*y, z)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "devtools.json")
    assert config == SceneConfig(0.2, 0.3)
    with tempfile.TemporaryDirectory(prefix="scene-check-", dir=args.results.parent) as temp:
        folder = Path(temp)
        config_path = folder / "facts.json"
        config_path.write_text('{"vehicle":{"radius_m":0.2},"target":{"radius_m":0.3}}')
        first_config = load_config(config_path)
        config_path.write_text('{"vehicle":{"radius_m":0.7},"target":{"radius_m":0.8}}')
        assert first_config == config and load_config(config_path) == SceneConfig(0.7, 0.8)
        rejected(lambda: setattr(config, "vehicle_radius_m", 1), FrozenInstanceError)
        invalid_config = [
            {}, [], {"vehicle": {"radius_m": 0.2}},
            {"vehicle": {"radius_m": 0.2}, "target": {"radius_m": 0.3}, "sensor": {}},
            {"vehicle": {"radius": 0.2}, "target": {"radius_m": 0.3}},
            {"vehicle": {"radius_m": 0.2, "margin_m": 0.1}, "target": {"radius_m": 0.3}},
            {"vehicle": [], "target": {"radius_m": 0.3}},
        ]
        for bad in (True, False, "0.2", None, 0, -0.2, math.inf, -math.inf, math.nan):
            for name in ("vehicle", "target"):
                value = {"vehicle": {"radius_m": 0.2}, "target": {"radius_m": 0.3}}
                value[name]["radius_m"] = bad
                invalid_config.append(value)
            rejected(lambda bad=bad: SceneConfig(bad, 0.3))
            rejected(lambda bad=bad: SceneConfig(0.2, bad))
        for value in invalid_config:
            config_path.write_text(json.dumps(value))
            rejected(lambda: load_config(config_path))
        for text in ('{"vehicle":{"radius_m":0.2,"radius_m":0.3},"target":{"radius_m":0.3}}',
                     '{"vehicle":{"radius_m":0.2},"vehicle":{"radius_m":0.3},"target":{"radius_m":0.3}}',
                     '{"vehicle":'):
            config_path.write_text(text)
            rejected(lambda: load_config(config_path))
        checks.append("single explicit JSON read, immutable radii, exact keys/types/ranges, duplicate-key rejection")

        # Binary-exact fractions make every parsed point independently comparable.
        points = tuple((i / 1024, -(i % 11) / 8, (i % 17) / 16) for i in range(100_005))
        points += ((-123.5, 4.25, 8.75), points[123])
        source = folder / "full.ply"
        comments = ("unit metre", "size_xyz 8 6 3 resolution 0.2", "complete background fixture")
        source.write_text(ply(points, comments, properties="zyx"), encoding="ascii")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        scene = Scene.from_ply(source, config, Pose((1, -2, 3)), (4, 5, 6))
        background = scene.background
        assert background.points_m == points and background.count == 100_007
        assert background.points_m[-1] == points[123], "duplicate points must be retained"
        assert background.source == source and background.comments == comments
        assert background.size_xyz_m == (8, 6, 3)
        close(background.resolution_m, 0.2)
        assert background.bounds_m == ((-123.5, 100_004 / 1024), (-10 / 8, 4.25), (0, 8.75))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        rejected(lambda: setattr(background, "points_m", ()), FrozenInstanceError)
        rejected(lambda: background.points_m[0].__setitem__(0, 1), AttributeError)
        checks.append("all 100007 points preserved in original order, duplicate retention, metadata/extrema, unchanged PLY")

        plain = folder / "plain.ply"
        plain.write_text(ply([(1, 2, 3), (-4, 5, -6)], comments=()), encoding="ascii")
        plain_map = PointCloudMap.from_ply(plain)
        assert plain_map.resolution_m is None and plain_map.size_xyz_m is None
        assert plain_map.comments == () and plain_map.bounds_m == ((-4, 1), (2, 5), (-6, 3))
        basic = ply([(1, 2, 3), (-4, 5, -6)])
        invalid_ply = [
            "not a PLY\n", ply([]), basic.replace("ascii", "binary_little_endian"),
            basic.replace("element vertex 2", "element vertex -1"),
            basic.replace("element vertex 2", "element vertex 1"),
            basic.replace("element vertex 2", "element vertex 3"),
            basic.replace("property double z", "property double x"),
            basic.replace("property double z", "property int z"),
            basic.replace("property double z", "property list uchar float z"),
            basic.replace("end_header", "element face 0\nend_header"),
            basic.replace("end_header\n", ""), basic.replace("unit metre", "unit centimetre"),
            basic.replace("resolution 0.2", "resolution 0"),
            basic.replace("resolution 0.2", "resolution nan"),
            basic.replace("size_xyz 8 6 3", "size_xyz 8 -6 3"),
            basic.replace("size_xyz 8 6 3 resolution", "size_xyz 8 6 resolution"),
            basic.replace("end_header", "comment size_xyz 8 6 3 resolution 0.2\nend_header"),
            basic.replace("1 2 3", "1 bad 3"), basic.replace("1 2 3", "1 2"),
            ply([(math.nan, 0, 0)]), ply([(0, math.inf, 0)]), ply([(0, 0, -math.inf)]),
        ]
        for text in invalid_ply:
            plain.write_text(text, encoding="ascii")
            rejected(lambda: PointCloudMap.from_ply(plain))
        checks.append("optional metadata, strict ASCII XYZ format/count/unit/metadata, nonfinite-point rejection")

        assert scene.vehicle.pose.position_m == (1, -2, 3)
        assert scene.target.pose.position_m == (4, 5, 6)
        for sphere, radius in ((scene.vehicle, 0.2), (scene.target, 0.3)):
            close(sphere.radius_m, radius)
            # Independent analytic sphere: six extrema and an oblique surface point.
            directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                          (0, 0, 1), (0, 0, -1), (1/3, 2/3, 2/3)]
            for direction in directions:
                surface = tuple(c + radius * d for c, d in zip(sphere.pose.position_m, direction))
                close(math.dist(surface, sphere.pose.position_m), sphere.radius_m)
        for radius in (0, -1, True, math.nan, math.inf):
            rejected(lambda radius=radius: Sphere(Pose((0, 0, 0)), radius))
        rejected(lambda: Sphere((0, 0, 0), 0.2))
        checks.append("explicit physical sphere centers/radii match analytic 3D surface/extrema; invalid spheres rejected")

        root_half = math.sqrt(0.5)
        roll90 = Pose((1, 2, 3), (root_half, 0, 0, root_half))
        close_vector(roll90.body_to_world((0, 1, 0)), (1, 2, 4))
        close_vector(roll90.body_to_world((0, 0, 1)), (1, 1, 3))
        pitch90 = Pose((-2, 4, 5), (0, root_half, 0, root_half))
        close_vector(pitch90.body_to_world((1, 0, 0)), (-2, 4, 4))
        close_vector(pitch90.body_to_world((0, 0, 1)), (-1, 4, 5))
        float32_half = struct.unpack("f", struct.pack("f", root_half))[0]
        rounded_pose = Pose((1, 2, 3), (float32_half, 0, 0, float32_half))
        close(math.hypot(*rounded_pose.quaternion_xyzw), 1)
        close_vector(rounded_pose.body_to_world((0, 1, 0)), (1, 2, 4))
        probe = (2.3, -4.1, 0.7)
        close_vector(rounded_pose.world_to_body(rounded_pose.body_to_world(probe)), probe)
        roll, pitch, yaw = (math.radians(d) for d in (30, -40, 75))
        mixed = Pose((1.25, -2.75, 3.5), euler_quaternion(roll, pitch, yaw))
        equivalent = Pose(mixed.position_m, tuple(-q for q in mixed.quaternion_xyzw))
        transform = mixed.as_transform("B_ref", "W")
        assert transform.source_frame == "B_ref" and transform.target_frame == "W"
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2.3, -4.1, 0.7)):
            expected = tuple(v + p for v, p in zip(euler_point(point, roll, pitch, yaw), mixed.position_m))
            actual = mixed.body_to_world(point)
            close_vector(actual, expected)
            close_vector(mixed.world_to_body(expected), point)
            close_vector(equivalent.body_to_world(point), expected)
            close_vector(transform.apply_points(point), expected)
            close_vector(transform.apply_vectors(point), euler_point(point, roll, pitch, yaw))
            close(math.dist(actual, mixed.position_m), math.hypot(*point))
        mutable_position, mutable_quaternion = [1, 2, 3], [0, 0, 0, 1]
        pose = Pose(mutable_position, mutable_quaternion)
        mutable_position[2], mutable_quaternion[3] = 100, 0
        assert pose.position_m == (1, 2, 3) and pose.quaternion_xyzw == (0, 0, 0, 1)
        for position in ((0, 0), (0, 0, 0, 0), (0, math.nan, 1), (False, 0, 0), "123"):
            rejected(lambda position=position: Pose(position))
        for quaternion in ((0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 2), (0, math.inf, 0, 1),
                           (0, 0, True, 0), (0, 0, 0, 1.000002)):
            rejected(lambda quaternion=quaternion: Pose((0, 0, 0), quaternion))
        rejected(lambda: mixed.body_to_world((0, 0, math.nan)))
        rejected(lambda: mixed.world_to_body((0, math.inf, 0)))
        checks.append("analytic roll/pitch, float32 quaternion rounding correction, independent Euler/inverse/distance/q-sign checks, immutable poses")

        vehicle_changed = scene.with_vehicle_pose(mixed)
        target_changed = scene.with_target_position((-7, 8, 9))
        combined = vehicle_changed.with_target_position((2, 3, 4))
        assert all(updated.background is background for updated in (vehicle_changed, target_changed, combined))
        assert vehicle_changed.target is scene.target and target_changed.vehicle is scene.vehicle
        assert vehicle_changed.vehicle.pose is mixed and vehicle_changed.vehicle.radius_m == 0.2
        assert target_changed.target.pose.position_m == (-7, 8, 9) and target_changed.target.radius_m == 0.3
        assert scene.vehicle.pose.position_m == (1, -2, 3) and scene.target.pose.position_m == (4, 5, 6)
        assert combined.vehicle is vehicle_changed.vehicle and combined.target.pose.position_m == (2, 3, 4)
        rejected(lambda: setattr(scene, "target", combined.target), FrozenInstanceError)
        rejected(lambda: scene.with_vehicle_pose((0, 0, 0)))
        rejected(lambda: scene.with_target_position((0, 0)))
        checks.append("independent vehicle/target updates share background; original snapshot/radii retained; invalid updates rejected")

    result = {"status": "passed", "checks": checks, "check_groups": len(checks),
              "full_point_count": 100_007, "units": "metres",
              "quaternion": "unit xyzw body-to-world, norm tolerance 1e-6 with rounding correction",
              "scope": "physical scene loading/state/transforms; sensing, ESDF, collision and dynamics are not implemented"}
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"scene: {len(checks)} check groups passed; result: {args.results}")


if __name__ == "__main__":
    main()
