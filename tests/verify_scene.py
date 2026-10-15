"""Verify complete scenes and their public, immutable geometry-query contract.

Known OBJ geometry checks world units, object identity, filtering and time
metadata independently of sensing or distance-field implementations. Existing
asset, sphere and pose checks also cover strict input validation and updates.
"""

import argparse
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile

from devtools.config import SceneConfig, load_config
from devtools.mesh import MeshMap, MeshObject
from devtools.scene import GeometrySnapshot, Pose, Scene, SceneObject, Sphere


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


def mesh_obj():
    """Independent rectangular plane and 2 x 2 x 3 m closed box, in OBJ order."""
    return """# unit metre
# size_xyz 8 6 4
o ground
v -4 -3 0
v 4 -3 0
v 4 3 0
v -4 3 0
f 1 2 3
f 1 3 4
o pillar_0
v -1 -2 0
v 1 -2 0
v 1 0 0
v -1 0 0
v -1 -2 3
v 1 -2 3
v 1 0 3
v -1 0 3
f 5 7 6
f 5 8 7
f 9 10 11
f 9 11 12
f 5 6 10
f 5 10 9
f 6 7 11
f 6 11 10
f 7 8 12
f 7 12 11
f 8 5 9
f 8 9 12
"""


def verify_geometry_contract(mesh_scene, mesh_source, config, checks):
    """Read real, target-only and updated snapshots without changing scene facts."""
    mesh_before = hashlib.sha256(mesh_source.read_bytes()).hexdigest()
    full = mesh_scene.get_geometry()
    expected_ids = ("map/ground", "map/pillar_0", "vehicle", "target")
    assert tuple(obj.object_id for obj in full.objects) == expected_ids
    assert tuple(obj.role for obj in full.objects) == ("ground", "obstacle", "vehicle", "target")
    assert isinstance(full.objects, tuple) and full.frame == "W" and full.unit == "metre"
    assert full.t_ref == -1.25 and full.map_bounds_m == ((-4, 4), (-3, 3), (0, 4))
    # The declared map domain is deliberately larger than the actual mesh Z extent.
    assert mesh_scene.background.bounds_m[2] == (0, 3)
    expected_geometries = (*mesh_scene.background.objects, mesh_scene.vehicle, mesh_scene.target)
    for obj, geometry in zip(full.objects, expected_geometries):
        assert obj.geometry is geometry and full.get_object(obj.object_id) is obj
        if isinstance(geometry, Sphere):
            assert obj.pose is geometry.pose
        else:
            assert obj.pose.position_m == (0, 0, 0)
            assert obj.pose.quaternion_xyzw == (0, 0, 0, 1)
    ground = full.get_object("map/ground").geometry
    obstacle = full.get_object("map/pillar_0").geometry
    assert set(ground.vertices_m) == {(x, y, 0) for x in (-4, 4) for y in (-3, 3)}
    assert set(obstacle.vertices_m) == {(x, y, z) for x in (-1, 1) for y in (-2, 0) for z in (0, 3)}
    for object_id, center, radius in (("vehicle", (1, 2, 3), 0.2), ("target", (-2, 1, 2), 0.3)):
        geometry = full.get_object(object_id).geometry
        assert geometry.pose.position_m == center
        close(geometry.radius_m, radius)
    checks.append("public mesh snapshot preserves independently known entities, world geometry, physical radii and declared domain/time")

    visible = mesh_scene.get_geometry(exclude_ids={"vehicle"})
    target_only = mesh_scene.get_geometry(include_ids=["target"])
    assert tuple(obj.object_id for obj in visible.objects) == ("map/ground", "map/pillar_0", "target")
    assert tuple(obj.object_id for obj in target_only.objects) == ("target",)
    assert target_only.get_object("target").geometry is visible.get_object("target").geometry
    # Include order never reorders source entities; exclusion wins over inclusion.
    selected = mesh_scene.get_geometry(include_ids=("target", "map/pillar_0", "vehicle"),
                                       exclude_ids=("vehicle",))
    assert tuple(obj.object_id for obj in selected.objects) == ("map/pillar_0", "target")
    generated = mesh_scene.get_geometry(include_ids=(object_id for object_id in reversed(expected_ids)))
    assert tuple(obj.object_id for obj in generated.objects) == expected_ids
    empty = mesh_scene.get_geometry(include_ids=[])
    excluded = mesh_scene.get_geometry(exclude_ids=expected_ids)
    assert empty.objects == excluded.objects == ()
    for snapshot in (visible, target_only, selected, generated, empty, excluded):
        assert snapshot.frame == full.frame and snapshot.unit == full.unit
        assert snapshot.t_ref == full.t_ref and snapshot.map_bounds_m == full.map_bounds_m
        for obj in snapshot.objects:
            assert obj is full.get_object(obj.object_id)
            assert obj.geometry is full.get_object(obj.object_id).geometry
    mutable_selection = ["target"]
    isolated = mesh_scene.get_geometry(include_ids=mutable_selection)
    mutable_selection.append("vehicle")
    assert tuple(obj.object_id for obj in isolated.objects) == ("target",)
    rejected(lambda: setattr(target_only, "objects", ()), FrozenInstanceError)
    rejected(lambda: setattr(target_only, "t_ref", 100), FrozenInstanceError)
    rejected(lambda: setattr(target_only, "frame", "B"), FrozenInstanceError)
    rejected(lambda: setattr(target_only, "unit", "centimetre"), FrozenInstanceError)
    rejected(lambda: setattr(target_only, "map_bounds_m", None), FrozenInstanceError)
    target = target_only.get_object("target")
    for field_name, value in (("object_id", "other"), ("role", "obstacle"), ("geometry", obstacle)):
        rejected(lambda field_name=field_name, value=value: setattr(target, field_name, value), FrozenInstanceError)
    rejected(lambda: setattr(target.geometry, "radius_m", 50), FrozenInstanceError)
    rejected(lambda: setattr(target.pose, "position_m", (100, 0, 0)), FrozenInstanceError)
    rejected(lambda: visible.get_object("vehicle"))
    for bad in ("target", b"target", {"target": True}, 42, False):
        rejected(lambda bad=bad: mesh_scene.get_geometry(include_ids=bad))
        rejected(lambda bad=bad: mesh_scene.get_geometry(exclude_ids=bad))
    rejected(lambda: mesh_scene.get_geometry(exclude_ids=None))
    for bad_id in ("unknown", "", " ", 1, True, None):
        rejected(lambda bad_id=bad_id: mesh_scene.get_geometry(include_ids=[bad_id]))
        rejected(lambda bad_id=bad_id: mesh_scene.get_geometry(exclude_ids=[bad_id]))
        rejected(lambda bad_id=bad_id: full.get_object(bad_id))
    for bad_id in ("", " ", 1, True, None):
        rejected(lambda bad_id=bad_id: SceneObject(bad_id, "target", mesh_scene.target))
    rejected(lambda: SceneObject("bad", "background", mesh_scene.target))
    rejected(lambda: SceneObject("bad", "obstacle", None))
    rejected(lambda: GeometrySnapshot((target, target), full.t_ref, full.map_bounds_m))
    rejected(lambda: GeometrySnapshot((None,), full.t_ref, full.map_bounds_m))
    for bounds in (((0, 1),), ((0, 1), (0, 1), (0, 0)),
                   ((0, 1), (0, 1), (0, math.inf)), ((0, 1), (0, 1), (False, 1))):
        rejected(lambda bounds=bounds: GeometrySnapshot((target,), full.t_ref, bounds))
    mutable_objects = [target]
    copied_snapshot = GeometrySnapshot(mutable_objects, full.t_ref, full.map_bounds_m)
    mutable_objects.clear()
    assert copied_snapshot.objects == (target,) and copied_snapshot.objects[0] is target
    assert tuple(obj.object_id for obj in mesh_scene.get_geometry().objects) == expected_ids
    assert hashlib.sha256(mesh_source.read_bytes()).hexdigest() == mesh_before
    checks.append("sensor-style real/target-only reads share immutable geometry; deterministic filters preserve metadata and reject unknown/invalid IDs")

    moved_pose = Pose((2, -3, 1), euler_quaternion(0.2, -0.3, 0.4))
    moved_vehicle = mesh_scene.with_vehicle_pose(moved_pose)
    moved_target = mesh_scene.with_target_position((3, -1, 2))
    both = moved_vehicle.with_target_position((3, -1, 2))
    for changed in (moved_vehicle, moved_target, both):
        snapshot = changed.get_geometry()
        assert changed.background is mesh_scene.background
        assert tuple(obj.object_id for obj in snapshot.objects) == expected_ids
        assert snapshot.t_ref == full.t_ref and snapshot.map_bounds_m == full.map_bounds_m
        for object_id in expected_ids[:2]:
            assert snapshot.get_object(object_id).geometry is full.get_object(object_id).geometry
    assert moved_vehicle.get_geometry().get_object("vehicle").pose is moved_pose
    assert moved_vehicle.get_geometry().get_object("target").geometry is full.get_object("target").geometry
    assert moved_target.get_geometry().get_object("vehicle").geometry is full.get_object("vehicle").geometry
    assert both.get_geometry().get_object("target").pose.position_m == (3, -1, 2)
    assert full.get_object("vehicle").pose.position_m == (1, 2, 3)
    assert full.get_object("target").pose.position_m == (-2, 1, 2)
    assert target_only.get_object("target").pose.position_m == (-2, 1, 2)

    # Source extrema are data bounds, not an inferred scene volume or occupancy.
    unnamed_bounds = MeshMap(mesh_source, mesh_scene.background.objects)
    unbounded_scene = Scene(unnamed_bounds, mesh_scene.vehicle, mesh_scene.target)
    assert unbounded_scene.get_geometry().map_bounds_m is None
    assert unbounded_scene.get_geometry().get_object("map/pillar_0").geometry is obstacle
    assert unnamed_bounds.bounds_m == mesh_scene.background.bounds_m
    checks.append("stable IDs/time across independent state updates and old snapshots; mesh extrema do not imply a declared map domain")

    builders = (
        lambda value: Scene(mesh_scene.background, mesh_scene.vehicle, mesh_scene.target, t_ref=value),
        lambda value: Scene.from_obj(mesh_source, config, Pose((1, 2, 3)), (-2, 1, 2), t_ref=value),
    )
    for builder in builders:
        for value in (None, 0, -7, 2.5):
            timed = builder(value)
            assert timed.t_ref == value and timed.get_geometry().t_ref == value
            assert timed.with_target_position((1, 0, 0)).with_vehicle_pose(moved_pose).t_ref == value
        for bad in (True, False, "1.25", [], math.nan, math.inf, -math.inf):
            rejected(lambda bad=bad, builder=builder: builder(bad))
    assert Scene.from_obj(mesh_source, config, Pose((0, 0, 1)), (2, 3, 1)).t_ref is None
    checks.append("optional finite snapshot time propagates through constructors and updates; absent time stays unknown and invalid values are rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "devtools.json")
    assert config == SceneConfig(0.2, 0.3, config.sensors)
    assert config.sensors is not None
    with tempfile.TemporaryDirectory(prefix="scene-check-", dir=args.results.parent) as temp:
        folder = Path(temp)
        config_path = folder / "facts.json"
        config_path.write_text('{"vehicle":{"radius_m":0.2},"target":{"radius_m":0.3}}')
        first_config = load_config(config_path)
        config_path.write_text('{"vehicle":{"radius_m":0.7},"target":{"radius_m":0.8}}')
        assert first_config == SceneConfig(0.2, 0.3) and load_config(config_path) == SceneConfig(0.7, 0.8)
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

        source = folder / "physical.obj"
        source.write_text(mesh_obj(), encoding="ascii")
        scene = Scene.from_obj(source, config, Pose((1, -2, 3)), (4, 5, 6))
        background = scene.background

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

        mesh_source = source
        mesh_text = mesh_obj()
        mesh_source.write_text(mesh_text, encoding="ascii")
        mesh_before = hashlib.sha256(mesh_source.read_bytes()).hexdigest()
        mesh_scene = Scene.from_obj(mesh_source, config, Pose((1, 2, 3)), (-2, 1, 2), t_ref=-1.25)
        mesh = mesh_scene.background
        assert isinstance(mesh, MeshMap) and mesh.source == mesh_source
        assert mesh.comments == ("unit metre", "size_xyz 8 6 4")
        assert mesh.size_xyz_m == (8, 6, 4) and mesh.bounds_m == ((-4, 4), (-3, 3), (0, 3))
        assert mesh.vertex_count == 12 and mesh.triangle_count == 14
        assert tuple((obj.name, obj.kind) for obj in mesh.objects) == (("ground", "ground"), ("pillar_0", "obstacle"))
        ground, obstacle = mesh.objects
        assert ground.triangles == ((0, 1, 2), (0, 2, 3))
        assert obstacle.triangles[0] == (0, 2, 1) and obstacle.triangles[-1] == (3, 4, 7)
        assert set(obstacle.vertices_m) == {(x, y, z) for x in (-1, 1) for y in (-2, 0) for z in (0, 3)}
        # Analytic box geometry independently checks every face normal, closure and volume.
        incidences, volume6 = {}, 0.0
        for a_index, b_index, c_index in obstacle.triangles:
            a, b, c = (obstacle.vertices_m[index] for index in (a_index, b_index, c_index))
            u, v = tuple(b[k] - a[k] for k in range(3)), tuple(c[k] - a[k] for k in range(3))
            normal = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])
            face_center = tuple((a[k] + b[k] + c[k])/3 for k in range(3))
            assert sum(normal[k] * (face_center[k] - (0, -1, 1.5)[k]) for k in range(3)) > 0
            volume6 += (a[0]*(b[1]*c[2] - b[2]*c[1])
                        + a[1]*(b[2]*c[0] - b[0]*c[2]) + a[2]*(b[0]*c[1] - b[1]*c[0]))
            for first, second in ((a_index, b_index), (b_index, c_index), (c_index, a_index)):
                incidences.setdefault(tuple(sorted((first, second))), []).append((first, second))
        close(volume6/6, 12)
        assert all(len(edges) == 2 and edges[0] == edges[1][::-1] for edges in incidences.values())
        assert mesh_scene.vehicle.radius_m == config.vehicle_radius_m and mesh_scene.target.radius_m == config.target_radius_m
        updated_mesh_scene = mesh_scene.with_vehicle_pose(mixed).with_target_position((7, 8, 9))
        assert updated_mesh_scene.background is mesh and mesh_scene.vehicle.pose.position_m == (1, 2, 3)
        assert Scene.from_obj(mesh_source, config, Pose((0, 0, 1)), (2, 3, 1)).background == mesh
        rejected(lambda: setattr(mesh, "objects", ()), FrozenInstanceError)
        rejected(lambda: setattr(obstacle, "vertices_m", ()), FrozenInstanceError)
        assert hashlib.sha256(mesh_source.read_bytes()).hexdigest() == mesh_before
        checks.append("complete OBJ object-local indices, independent analytic box normals/closure/volume, exact bounds, configured spheres and shared background")
        verify_geometry_contract(mesh_scene, mesh_source, config, checks)

        ground_only = mesh_text.split("o pillar_0")[0]
        mesh_source.write_text(ground_only, encoding="ascii")
        assert MeshMap.from_obj(mesh_source).vertex_count == 4
        box_lines = mesh_text.split("o pillar_0\n")[1].splitlines()
        box_vertices = [line for line in box_lines if line.startswith("v ")]
        box_faces = [line for line in box_lines if line.startswith("f ")]
        shifted_faces = ["f " + " ".join(str(int(index) + 8) for index in line.split()[1:])
                         for line in box_faces]
        overlapping_text = mesh_text + "o pillar_1\n" + "\n".join(box_vertices + shifted_faces) + "\n"
        mesh_source.write_text(overlapping_text, encoding="ascii")
        overlapping_map = MeshMap.from_obj(mesh_source)
        assert overlapping_map.vertex_count == 20 and overlapping_map.triangle_count == 26
        assert overlapping_map.objects[1].vertices_m == overlapping_map.objects[2].vertices_m
        namespaced_text = overlapping_text.replace("o pillar_0", "o target").replace("o pillar_1", "o vehicle")
        mesh_source.write_text(namespaced_text, encoding="ascii")
        namespaced = Scene.from_obj(mesh_source, config, Pose((0, 0, 1)), (2, 3, 1)).get_geometry()
        assert tuple(obj.object_id for obj in namespaced.objects) == ("map/ground", "map/target", "map/vehicle", "vehicle", "target")
        assert namespaced.get_object("map/target").role == namespaced.get_object("map/vehicle").role == "obstacle"
        assert isinstance(namespaced.get_object("target").geometry, Sphere)
        assert isinstance(namespaced.get_object("vehicle").geometry, Sphere)
        checks.append("map object names target/vehicle remain independent from namespaced dynamic sphere IDs")
        disconnected_obstacle = (ground_only + "o pillar_0\n"
                                 + "\n".join(box_vertices * 2 + box_faces + shifted_faces) + "\n")
        mesh_source.write_text(mesh_text.replace("# unit metre\n", "").replace("# size_xyz 8 6 4\n", ""), encoding="ascii")
        assert MeshMap.from_obj(mesh_source).size_xyz_m is None
        reversed_obstacle = "\n".join(" ".join((fields[0], fields[1], fields[3], fields[2]))
                                      if fields[:1] == ["f"] else line
                                      for line in mesh_text.splitlines()
                                      for fields in [line.split()]) + "\n"
        # Leave the ground upward while reversing the complete obstacle.
        reversed_obstacle = ground_only + "o pillar_0" + reversed_obstacle.split("o pillar_0")[1]
        invalid_obj = [
            "", "v 0 0 0\n", "o ground\n", mesh_text + "mtllib unused.mtl\n",
            mesh_text + "f 8 9 12\n", disconnected_obstacle,
            mesh_text.replace("o pillar_0", "o ground"), mesh_text.replace("o ground", "g ground"),
            mesh_text.replace("v -1 -2 0", "v nan -2 0"), mesh_text.replace("v -1 -2 0", "v 0 inf 0"),
            mesh_text.replace("v -1 -2 0", "v -1 -2"), mesh_text.replace("f 5 7 6", "f 5 5 6"),
            mesh_text.replace("f 5 7 6", "f 5 7 1"), mesh_text.replace("f 5 7 6", "f 5 7 13"),
            mesh_text.replace("f 5 7 6", "f -5 7 6"), mesh_text.replace("f 5 7 6", "f 5/1 7/1 6/1"),
            mesh_text.replace("f 5 7 6", "f 5 7 6 8"), mesh_text.replace("f 5 7 6\n", ""),
            mesh_text.replace("f 5 7 6", "f 5 6 7"), reversed_obstacle,
            mesh_text.replace("f 5 7 6", "v 0 0 0\nf 5 7 6"),
            mesh_text.replace("f 5 7 6", "f 5 7 6\nv 0 0 0"),
            mesh_text.replace("v -1 -2 3", "v -1 -2 0"),
            mesh_text.replace("f 1 2 3", "f 1 3 2"),
            mesh_text.replace("unit metre", "unit centimetre"),
            mesh_text + "# unit metre\n", mesh_text + "# size_xyz 8 6 4\n",
            mesh_text.replace("size_xyz 8 6 4", "size_xyz 8 6 4 resolution 0.1"),
            mesh_text.replace("size_xyz 8 6 4", "size_xyz 8 6 2"),
            mesh_text.replace("v -4 -3 0", "v -3 -3 0"),
            mesh_text.replace("size_xyz 8 6 4", "size_xyz 8 0 4"),
        ]
        for text in invalid_obj:
            mesh_source.write_text(text, encoding="ascii")
            rejected(lambda: MeshMap.from_obj(mesh_source))
        mutable_vertices = [list(vertex) for vertex in obstacle.vertices_m]
        mutable_faces = [list(face) for face in obstacle.triangles]
        copied = MeshObject("pillar_copy", mutable_vertices, mutable_faces, "obstacle")
        mutable_vertices[0][0], mutable_faces[0][0] = 100, 7
        assert copied.vertices_m == obstacle.vertices_m and copied.triangles == obstacle.triangles
        rejected(lambda: MeshObject("pillar_bad", obstacle.vertices_m, ((True, 2, 1),), "obstacle"))
        rejected(lambda: MeshMap(mesh_source, (obstacle,)))
        rejected(lambda: MeshMap(mesh_source, (ground, obstacle, obstacle)))
        checks.append("ground-only and independent overlapping objects; optional metadata; strict OBJ features/units/indices/bounds; degenerate/open/inward/disconnected surfaces rejected; immutable copied mesh storage")

    result = {"status": "passed", "checks": checks, "check_groups": len(checks),
              "mesh_vertex_count": mesh.vertex_count, "mesh_triangle_count": mesh.triangle_count, "units": "metres",
              "quaternion": "unit xyzw body-to-world, norm tolerance 1e-6 with rounding correction",
              "scope": "physical scene loading/state/transforms and public filtered geometry snapshots; sensing, ESDF, collision and dynamics are not implemented"}
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"scene: {len(checks)} check groups passed; result: {args.results}")


if __name__ == "__main__":
    main()
