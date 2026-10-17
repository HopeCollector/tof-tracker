"""Adapt existing public scene geometry to the reference's 0.1 m occupancy input.

Run in Docker. This is a voxelized cost comparison, not geometry equivalence.
No rendering, hidden image resampling, ROS, or changes to reference sources.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from devtools.config import load_config
from devtools.scene import Scene, Sphere
from devtools.mesh import MeshObject
from experiments.sensor_zone_perf.benchmark import make_poses, TARGET_POSITION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, nargs="+", default=[20, 5000])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    poses = make_poses(64)
    pose_file = args.output / "poses-f32-xyz-xyzw.bin"
    np.asarray([(*p.position_m, *p.quaternion_xyzw) for p in poses], dtype="<f4").tofile(pose_file)
    manifest = {"resolution_m": .1, "pose_path": str(pose_file),
                "pose_sha256": digest(pose_file), "poses": 64,
                "pose_source": "Same prefix-stable varied poses as sensor_zone_perf; first is the 4m target review regression",
                "adaptation": "Occupy 0.1m cells whose center lies inside public closed pillar AABB or target sphere; ground layer occupied. One point at each occupied cell center plus the four original ground corners to fix XY origin/extent. Reference GridMap is unchanged; threshold=0. Mesh/continuous geometry is not preserved exactly.",
                "maps": []}
    for count in args.maps:
        path = ROOT / f"outputs/sensor_perf/maps/pillars-{count}.obj"
        scene = Scene.from_obj(path, load_config(ROOT / "configs/devtools.json"),
                               poses[0], TARGET_POSITION)
        geometry = scene.get_geometry(exclude_ids={"vehicle"})
        origin = np.array([bound[0] for bound in geometry.map_bounds_m])
        extent = np.array([bound[1] for bound in geometry.map_bounds_m]) - origin
        shape = np.rint(extent / .1).astype(int)
        occupied = np.zeros(tuple(shape), dtype=bool)
        occupied[:, :, 0] = True
        ground_vertices = None
        for item in geometry.objects:
            obj = item.geometry
            if isinstance(obj, MeshObject):
                vertices = np.asarray(obj.vertices_m)
                if obj.kind == "ground":
                    ground_vertices = vertices.astype("<f4")
                    continue
                low, high = vertices.min(axis=0), vertices.max(axis=0)
                # The performance maps are unrotated closed rectangular pillars.
                assert len(vertices) == 8 and len(obj.triangles) == 12
                assert set(map(tuple, vertices)) == {tuple(v) for v in
                    np.array(np.meshgrid(*zip(low, high), indexing="ij")).reshape(3, -1).T}
            elif isinstance(obj, Sphere):
                center = np.asarray(obj.pose.position_m)
                low, high = center - obj.radius_m, center + obj.radius_m
            else:
                raise TypeError(type(obj))
            first = np.maximum(0, np.ceil((low - origin) / .1 - .5).astype(int))
            last = np.minimum(shape, np.floor((high - origin) / .1 - .5).astype(int) + 1)
            if np.any(last <= first):
                continue
            region = tuple(slice(int(a), int(b)) for a, b in zip(first, last))
            if isinstance(obj, Sphere):
                coordinates = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(first, last))]
                squared = sum((origin[i] + (coordinates[i] + .5)*.1 - center[i])**2 for i in range(3))
                occupied[region] |= squared <= obj.radius_m**2
            else:
                occupied[region] = True
        assert ground_vertices is not None
        output = args.output / f"pillars-{count}-occupancy-f32-xyz.bin"
        with output.open("wb") as stream:
            ground_vertices.tofile(stream)
            # Streaming x slabs avoid a large temporary index list/cloud.
            for x in range(shape[0]):
                yz = np.argwhere(occupied[x])
                points = np.empty((len(yz), 3), dtype="<f4")
                points[:, 0] = origin[0] + (x + .5)*.1
                points[:, 1:] = origin[1:] + (yz + .5)*.1
                points.tofile(stream)
        manifest["maps"].append({"pillars": count, "mesh_path": str(path),
                                  "mesh_sha256": digest(path), "cloud_path": str(output),
                                  "cloud_sha256": digest(output), "cloud_points": output.stat().st_size//12,
                                  "declared_shape": shape.tolist(), "occupied_cells": int(occupied.sum()),
                                  "target_position_m": TARGET_POSITION, "target_radius_m": .3})
        print(json.dumps(manifest["maps"][-1]), flush=True)
    (args.output / "fixtures.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
