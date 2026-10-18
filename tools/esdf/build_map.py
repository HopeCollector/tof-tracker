"""Build a reusable uniform C1 distance grid from a complete mesh map, offline."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from devtools.esdf import build_esdf
from devtools.mesh import MeshMap
from devtools.mesh_distance import MeshDistanceQuery, PreparedMesh
from devtools.scene import GeometrySnapshot, SceneObject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=.2, help="shared-node spacing in metres")
    parser.add_argument("--bounds", nargs=6, type=float, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    parser.add_argument("--library", type=Path, default=Path("outputs/esdf/build/libmesh_distance.so"))
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")
    if not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("voxel-size must be positive and finite")
    start = time.perf_counter()
    asset = MeshMap.from_obj(args.map)
    if args.bounds is not None:
        bounds = np.asarray(args.bounds).reshape(2, 3)
    elif asset.size_xyz_m is not None:
        x, y, z = asset.size_xyz_m
        bounds = np.array([[-x/2, -y/2, 0.], [x/2, y/2, z]])
    else:
        parser.error("map has no declared size; supply --bounds")
    snapshot = GeometrySnapshot(tuple(SceneObject("map/" + obj.name, obj.kind, obj)
                                      for obj in asset.objects), None,
                                tuple(zip(bounds[0], bounds[1])))
    # Cover ceil-to-grid storage padding and the query interpolation halo.
    support = bounds + np.array([[-1.], [2.]]) * args.voxel_size
    mesh = PreparedMesh.from_geometry(snapshot, query_bounds_m=support)
    geometry_seconds = time.perf_counter() - start
    query_start = time.perf_counter()
    with MeshDistanceQuery(mesh.vertices_m, mesh.triangles, library_path=args.library) as query:
        bvh_seconds = time.perf_counter() - query_start
        sample_start = time.perf_counter()
        grid = build_esdf(query, bounds_m=bounds, voxel_size_m=args.voxel_size,
                          chunk_size=args.chunk_size,
                          metadata={"source_map": str(args.map),
                                    "source_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(),
                                    "geometry": mesh.metadata})
        sample_seconds = time.perf_counter() - sample_start
    save_start = time.perf_counter()
    grid.save(args.output)
    save_seconds = time.perf_counter() - save_start
    report = {"completed": True, "map": str(args.map), "output": str(args.output),
              "bounds_m": grid.bounds_m, "sample_origin_m": grid.sample_origin_m,
              "voxel_size_m": grid.spacing_m, "shape_xyz_with_halo": grid.shape,
              "node_count": int(np.prod(grid.shape)), "node_bytes": grid.storage_bytes,
              "geometry_preparation_seconds": geometry_seconds, "bvh_seconds": bvh_seconds,
              "node_sampling_seconds": sample_seconds, "save_seconds": save_seconds,
              "total_wall_seconds": time.perf_counter() - start,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "rss_scope": "Linux lifetime peak RSS of this Python builder, including imports, mesh preparation, BVH, node sampling and save; not container total or swap",
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "map_files": {p.name: p.stat().st_size for p in args.output.iterdir() if p.is_file()}}
    (args.output / "build-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
