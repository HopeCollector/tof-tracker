"""Same-geometry controlled PLY input for the fixed YOPO EDT algorithm.

This is a surface sampling adapter, not the original project's absent dataset
or original random map generator. No obstacle interior filling is applied.
"""
import argparse
import ast
import gc
import glob
import hashlib
import json
import os
from pathlib import Path
import resource
import platform
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from devtools.mesh import MeshMap


def surface_points(asset, spacing=.1):
    chunks = []
    for obj in asset.objects:
        xyz = np.asarray(obj.vertices_m)
        lo, hi = xyz.min(0), xyz.max(0)
        if obj.kind == "ground":
            x, y = [np.linspace(lo[a], hi[a], int(np.ceil((hi[a]-lo[a])/spacing))+1) for a in (0,1)]
            xx, yy = np.meshgrid(x, y, indexing="ij")
            chunks.append(np.stack((xx,yy,np.zeros_like(xx)),axis=-1).reshape(-1,3))
        else:
            for axis in range(3):
                other = [a for a in range(3) if a != axis]
                uv = [np.linspace(lo[a], hi[a], int(np.ceil((hi[a]-lo[a])/spacing))+1) for a in other]
                u,v = np.meshgrid(*uv,indexing="ij")
                for side in (0,1):
                    p = np.empty((u.size,3))
                    p[:,axis] = (lo if side == 0 else hi)[axis]
                    p[:,other[0]],p[:,other[1]] = u.ravel(),v.ravel()
                    chunks.append(p)
    return np.concatenate(chunks).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, help="read-only reference safety_loss.py for verbatim builder parity")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    asset = MeshMap.from_obj(args.map)
    pts32 = surface_points(asset)
    ply = args.output/"pointcloud-0.ply"
    with ply.open("wb") as stream:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts32)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        stream.write(header.encode())
        stream.write(pts32.astype("<f4").tobytes())
    # Open3D exposes the float32 PLY coordinates as float64; preserve that input.
    points = pts32.astype(np.float64)
    lo = points.min(0)-np.array([0,0,.2])
    hi = points.max(0)+np.array([0,0,6.])
    shape = np.ceil((hi-lo)/.2).astype(int)
    indices = ((points-lo)/.2).astype(int)
    valid = ((indices >= 0)&(indices < shape)).all(1)
    occ = np.zeros(tuple(shape), dtype=np.uint8)
    occ[tuple(indices[valid].T)] = 1
    sampling_seconds = time.perf_counter()-started
    start = time.perf_counter()
    outside = distance_transform_edt(occ == 0)*.2
    inside = distance_transform_edt(occ == 1)*.2
    outside[occ == 1] = -inside[occ == 1]
    edt_seconds = time.perf_counter()-start
    samples = outside.astype(np.float32)
    np.save(args.output/"reference_edt.npy",samples,allow_pickle=False)
    np.save(args.output/"occupancy.npy",occ,allow_pickle=False)
    del outside,inside,indices,points,pts32
    gc.collect()
    parity = None
    if args.original:
        # Execute the selected original methods, not the rewritten arithmetic.
        import open3d as o3d
        import torch as th
        src = args.original.read_text()
        cls = next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name == "SafetyLoss")
        chosen = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {"get_sdf_from_ply","read_sorted_ply_files"}]
        namespace = dict(np=np,th=th,os=os,glob=glob,o3d=o3d,distance_transform_edt=distance_transform_edt)
        code = "class OriginalBuilder:\n"+"\n\n".join("    "+ast.get_source_segment(src,n) for n in chosen)
        exec(compile(code,str(args.original),"exec"),namespace)
        original = namespace["OriginalBuilder"]()
        original.map_expand_min,original.map_expand_max = np.array([0,0,.2]),np.array([0,0,6.])
        original.voxel_size,original.device = .2,th.device("cpu")
        output = original.get_sdf_from_ply(str(args.output))[0][0,0].permute(2,1,0).numpy()
        assert np.array_equal(samples,output)
        parity = {"bitwise_equal_to_original_builder":True,"original_sha256":hashlib.sha256(src.encode()).hexdigest()}
    report = {"source_map_sha256":hashlib.sha256(args.map.read_bytes()).hexdigest(),
        "pointcloud_sha256":hashlib.sha256(ply.read_bytes()).hexdigest(),"spacing_m":.2,
        "point_spacing_max_m":.1,"point_count":int(valid.size),"dropped_upper_boundary_points":int((~valid).sum()),
        "origin_m":lo.tolist(),"max_bounds_m":hi.tolist(),"shape_xyz":shape.tolist(),
        "occupied_nodes":int(occ.sum()),"node_bytes":samples.nbytes,
        "surface_sampling_and_voxelization_seconds":sampling_seconds,"two_edt_seconds":edt_seconds,
        "process_peak_rss_bytes_including_original_parity":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "original_builder_parity":parity,
        "versions":{"python":platform.python_version(),"numpy":np.__version__,
                    "scipy":__import__("scipy").__version__},
        "limits":"Controlled same-OBJ raw box-face+ground sampling, float32 .1m-max spacing; not original dataset/generator/PCL VoxelGrid. Interior is not filled. Original floor-index, double EDT, no half-cell correction preserved."}
    (args.output/"metadata.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
