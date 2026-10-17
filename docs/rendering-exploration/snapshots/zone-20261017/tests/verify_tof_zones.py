"""Independent continuous ToF-zone geometry checks, executed only in Docker.

The CPU oracle enumerates active constraints of a convex quadratic program on
analytic box faces / a plane. It does not sample rays, render a finer image, or
reuse the GPU polygon-clipping implementation. Downloads below are assertion
boundaries only; the formal path returns CUDA tensors.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from functools import lru_cache
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from devtools.config import DepthConfig, load_config
from devtools.mesh import MeshMap, MeshObject
from devtools.rendering import BatchRenderer
from devtools.scene import GeometrySnapshot, Pose, Scene, SceneObject, Sphere
from devtools.sensors import poses_to_tensor, sample_camera, sample_depth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BOX_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
CONFIG = DepthConfig(8, 8, 90, 90, .1, 12)
ORIGIN = (0., 0., 10.)
ATOL = 3e-5


def zone_halfspaces(config, row, col):
    """Analytic homogeneous inequalities A @ sensor_point <= 0."""
    hy, hz = math.tan(math.radians(config.hfov_deg)/2), math.tan(math.radians(config.vfov_deg)/2)
    left, right = hy*(1-2*col/config.width), hy*(1-2*(col+1)/config.width)
    top, bottom = hz*(1-2*row/config.height), hz*(1-2*(row+1)/config.height)
    return np.asarray(((-1, 0, 0), (-left, 1, 0), (right, -1, 0),
                       (-top, 0, 1), (bottom, 0, -1)), dtype=float)


def _face_qp(inequalities, limits, face_normal, face_offset):
    """Nearest point on one bounded planar polygon, via active-set QP.

    In a two-dimensional face a minimizer has zero, one, or two additional
    active inequalities. Enumerated feasible vertices also independently prove
    that the intersection has area, rather than merely touching an edge.
    """
    # A plane through the camera projects to a line and has no angular area.
    if abs(face_offset) < 1e-12:
        return math.inf, None
    count = len(limits)
    candidates, vertices = [], []
    for extra_count in (0, 1, 2):
        if extra_count == 0:
            points = (face_normal*face_offset/(face_normal@face_normal))[None]
        else:
            active = _active_sets(count, extra_count)
            matrix = np.concatenate((np.broadcast_to(face_normal,(len(active),1,3)), inequalities[active]),axis=1)
            target = np.concatenate((np.full((len(active),1),face_offset),limits[active]),axis=1)
            if extra_count == 2:
                full_rank = np.abs(np.linalg.det(matrix)) > 1e-11
                points = np.linalg.solve(matrix[full_rank],target[full_rank,...,None])[...,0]
            else:
                gram = matrix @ matrix.transpose(0,2,1)
                full_rank = np.abs(np.linalg.det(gram)) > 1e-11
                multipliers = np.linalg.solve(gram[full_rank],target[full_rank,...,None])
                points = (matrix[full_rank].transpose(0,2,1) @ multipliers)[...,0]
        valid = np.max(points @ inequalities.T-limits,axis=1) <= 1e-8
        points = points[valid]
        candidates.extend(points)
        if extra_count == 2:
            vertices.extend(points)
    if len(vertices) < 3:
        return math.inf, None
    differences = np.asarray(vertices) - vertices[0]
    # Rank is measured in world metres, not a projected pixel epsilon. It
    # rejects exact line/point contacts while retaining our sub-millimetre box.
    if np.linalg.matrix_rank(differences, tol=1e-9) < 2:
        return math.inf, None
    point = min(candidates, key=lambda p: float(p @ p))
    return float(np.linalg.norm(point)), point


@lru_cache(maxsize=8)
def _active_sets(count, extra_count):
    return np.asarray(tuple(combinations(range(count),extra_count)),dtype=np.int64)


def nearest_box_in_zone(config, pose_matrix, bounds_xyz, row, col):
    """Return (nearest radial distance, sensor point) on an analytic closed box.

    pose_matrix is a CPU sensor-to-world 4x4 matrix. Infinity/None is a miss.
    No max/min range filtering is performed: a too-close occluder still wins.
    """
    pose_matrix = np.asarray(pose_matrix, dtype=float)
    rotation, origin = pose_matrix[:3, :3], pose_matrix[:3, 3]
    low, high = np.asarray(bounds_xyz, dtype=float).T
    # world p = rotation @ sensor p + origin; six closed-box inequalities.
    box_a = np.vstack((rotation, -rotation))
    box_b = np.concatenate((high-origin, origin-low))
    inequalities = np.vstack((box_a, zone_halfspaces(config, row, col)))
    limits = np.concatenate((box_b, np.zeros(5)))
    best = (math.inf, None)
    for normal, offset in zip(box_a, box_b):
        candidate = _face_qp(inequalities, limits, normal, offset)
        if candidate[0] < best[0]:
            best = candidate
    return best


def nearest_ground_in_zone(config, pose_matrix, bounds_xy, row, col):
    """Nearest point on a finite analytic world z=0 rectangle in one zone."""
    pose_matrix = np.asarray(pose_matrix, dtype=float)
    rotation, origin = pose_matrix[:3, :3], pose_matrix[:3, 3]
    low, high = np.asarray(bounds_xy, dtype=float).T
    inequalities = np.vstack((rotation[:2], -rotation[:2], zone_halfspaces(config, row, col)))
    limits = np.concatenate((high-origin[:2], origin[:2]-low, np.zeros(5)))
    return _face_qp(inequalities, limits, rotation[2], -origin[2])


def analytic_box_zones(config, pose_matrix, boxes, *, ground=None):
    """Public test helper: ranges and names from independent face QPs."""
    ranges = np.full((config.height, config.width), math.inf)
    names = np.full(ranges.shape, "", dtype=object)
    for row in range(config.height):
        for col in range(config.width):
            candidates = [(nearest_box_in_zone(config, pose_matrix, bounds, row, col)[0], name)
                          for name, bounds in boxes]
            if ground is not None:
                candidates.append((nearest_ground_in_zone(config, pose_matrix, ground, row, col)[0], "ground"))
            distance, name = min(candidates, key=lambda candidate:candidate[0], default=(math.inf, ""))
            if distance <= config.max_range_m:
                ranges[row, col] = distance if distance >= config.min_range_m else math.nan
                names[row, col] = name
    return ranges, names


def box(name, bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    vertices = ((x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1))
    return MeshObject(name, vertices, BOX_FACES, "obstacle")


def fixture(*boxes, bounds=None):
    return GeometrySnapshot(tuple(SceneObject(name, "obstacle", box(name, extent)) for name, extent in boxes),
                            7.5, bounds)


def host(value):
    return value.detach().cpu().numpy()


def raw_host(value):
    return {"distance": host(value.distance_m), "identity": host(value.object_index),
            "point": host(value.hit_point_sensor_m), "triangle": host(value.triangle_index)}


def check_equal(actual, expected, atol=ATOL):
    for key in actual:
        if key in ("identity", "triangle"):
            assert np.array_equal(actual[key], expected[key]), key
        else:
            assert np.allclose(actual[key], expected[key], rtol=0, atol=atol), key


def contract(raw, matrices, config, source):
    shape = (len(matrices), config.height, config.width)
    for tensor, expected_shape, dtype in ((raw.distance_m, shape, torch.float32),
                                          (raw.object_index, shape, torch.int64),
                                          (raw.hit_point_sensor_m, (*shape, 3), torch.float32),
                                          (raw.triangle_index, shape, torch.int64)):
        assert tensor.shape == expected_shape and tensor.dtype == dtype
        assert tensor.is_cuda and tensor.device == matrices.device
    assert raw.object_ids == tuple(obj.object_id for obj in source.objects)
    assert raw.t_ref == source.t_ref and raw.map_bounds_m == source.map_bounds_m
    value = raw_host(raw)
    missed = value["identity"] < 0
    assert np.all(value["distance"][missed] == 0) and np.all(value["triangle"][missed] == -1)
    assert np.all(value["point"][missed] == 0)
    hit = ~missed
    assert np.allclose(np.linalg.norm(value["point"][hit], axis=-1), value["distance"][hit], rtol=0, atol=ATOL)
    return value


def check_mesh_source(value, source, pose_matrix):
    """Returned ID, triangle and point must identify the same public surface."""
    offset, metadata = 0, []
    for object_index, obj in enumerate(source.objects):
        geometry = obj.geometry
        count = len(geometry.triangles) if isinstance(geometry, MeshObject) else 16128
        metadata.append((offset, count, object_index, geometry))
        offset += count
    rotation, origin = pose_matrix[:3, :3], pose_matrix[:3, 3]
    for row, col in np.argwhere(value["identity"][0] >= 0):
        object_index = int(value["identity"][0,row,col])
        triangle_index = int(value["triangle"][0,row,col])
        start, count, expected_id, geometry = metadata[object_index]
        assert expected_id == object_index and start <= triangle_index < start+count
        if not isinstance(geometry, MeshObject):
            continue
        vertices = np.asarray(geometry.vertices_m)[np.asarray(geometry.triangles)[triangle_index-start]]
        triangle = (vertices-origin) @ rotation
        point = value["point"][0,row,col].astype(float)
        edge1, edge2 = triangle[1]-triangle[0], triangle[2]-triangle[0]
        basis = np.column_stack((edge1,edge2))
        uv = np.linalg.lstsq(basis, point-triangle[0], rcond=None)[0]
        residual = point-triangle[0]-basis@uv
        assert np.linalg.norm(residual) < ATOL
        assert uv.min() >= -2e-5 and uv.sum() <= 1+2e-5, (row,col,uv)


def expected_hits(value, expected):
    actual = {tuple(index) for index in np.argwhere(value["identity"][0] >= 0)}
    assert actual == set(expected), (actual, expected)


def verify_small_objects_and_boundaries():
    records = []
    cases = (
        ("tiny_inside_zone", ((4,4.002),(.2,.202),(10.2,10.202)), {(3,3)}, (4,.2,.2)),
        ("cross_column", ((4,4.002),(-.05,.05),(10.2,10.202)), {(3,3),(3,4)}, (4,0,.2)),
        ("cross_row_column", ((4,4.002),(-.05,.05),(9.95,10.05)), {(3,3),(3,4),(4,3),(4,4)}, (4,0,0)),
        ("edge_contact_has_no_area", ((4,4.002),(0,.1),(10.2,10.202)), {(3,3)}, (4,0,.2)),
        ("corner_contact_has_no_area", ((4,4.002),(0,.1),(10,10.1)), {(3,3)}, (4,0,0)),
    )
    for name, bounds, expected, closest in cases:
        source = fixture(("probe", bounds))
        with BatchRenderer(source) as renderer:
            matrices = poses_to_tensor((Pose(ORIGIN),), device=renderer.device)
            pixel = renderer.render(matrices, CONFIG)
            raw = renderer.render(matrices, CONFIG, sampling="zone")
            value = contract(raw, matrices, CONFIG, source)
            expected_hits(value, expected)
            for row,col in expected:
                assert abs(float(value["distance"][0,row,col])-np.linalg.norm(closest)) < ATOL
                assert np.allclose(value["point"][0,row,col], closest, rtol=0, atol=ATOL)
                analytic, _ = nearest_box_in_zone(CONFIG, host(matrices)[0], bounds, row, col)
                assert abs(analytic-np.linalg.norm(closest)) < 1e-8
            if name == "tiny_inside_zone":
                assert not np.any(host(pixel.object_index) >= 0), "tiny object unexpectedly covered a pixel center"
            check_mesh_source(value, source, host(matrices)[0])
            records.append({"case": name, "hit_zones": sorted([list(index) for index in expected]),
                            "nearest_m": float(np.linalg.norm(closest))})
    return records


def verify_plane_and_occlusion():
    records = {}
    plane = ("plane", ((4,4.01),(-10,10),(0,20)))
    source = fixture(plane)
    with BatchRenderer(source) as renderer:
        matrices = poses_to_tensor((Pose(ORIGIN),), device=renderer.device)
        value = contract(renderer.render(matrices, CONFIG, sampling="zone"), matrices, CONFIG, source)
        pixel = host(renderer.render(matrices, CONFIG).distance_m)
        assert np.all(value["identity"] == 0)
        for row in range(8):
            for col in range(8):
                y = min((1-2*col/8,1-2*(col+1)/8), key=abs)
                z = min((1-2*row/8,1-2*(row+1)/8), key=abs)
                expected = np.asarray((4,4*y,4*z))
                assert np.allclose(value["point"][0,row,col], expected, rtol=0, atol=ATOL)
                assert abs(value["distance"][0,row,col]-np.linalg.norm(expected)) < ATOL
        assert np.all(value["distance"] < pixel), "zone result was still center-ray depth"
        check_mesh_source(value, source, host(matrices)[0])
        records["plane_foot_edge_corner"] = "64 analytic distances and points matched"
    cases = (
        ("full_occlusion", (("near",((3,3.1),(-10,10),(0,20))),
                             ("back",((4,4.01),(.2,.24),(10.2,10.24)))), "near", (3,0,0)),
        ("partial_occlusion", (("near",((3,3.01),(0,.165),(10,10.3))),
                                ("back",((4,4.01),(.2,.24),(10.2,10.24)))), "near", (3,0,0)),
        ("radial_not_forward_depth", (("earlier_x",((3,3.01),(.7,.72),(10.7,10.72))),
                                      ("radial_nearest",((3.05,3.06),(.02,.04),(10.02,10.04)))), "radial_nearest", (3.05,.02,.02)),
    )
    for name, boxes, winner, closest in cases:
        source = fixture(*boxes)
        with BatchRenderer(source) as renderer:
            matrices = poses_to_tensor((Pose(ORIGIN),), device=renderer.device)
            value = contract(renderer.render(matrices, CONFIG, sampling="zone"), matrices, CONFIG, source)
            row,col=3,3
            assert value["identity"][0,row,col] == raw_id(source,winner)
            assert abs(value["distance"][0,row,col]-np.linalg.norm(closest)) < ATOL
            check_mesh_source(value,source,host(matrices)[0])
            records[name] = {"winner":winner,"range_m":float(value["distance"][0,row,col])}
            selected = contract(renderer.render(matrices,CONFIG,only_object=boxes[-1][0],sampling="zone"),
                                matrices,CONFIG,source)
            assert np.all(selected["identity"][selected["identity"]>=0] == len(boxes)-1)
            assert np.all(selected["triangle"][selected["triangle"]>=0] >= 12*(len(boxes)-1))
            check_mesh_source(selected,source,host(matrices)[0])
    return records


def raw_id(source, name):
    return tuple(obj.object_id for obj in source.objects).index(name)


def verify_near_and_coverage():
    records = {}
    crossing = ("crossing", ((-1,.05),(-1,1),(9,11)))
    far = ("far", ((4,4.1),(-10,10),(0,20)))
    source = fixture(crossing,far,bounds=((-20,20),(-20,20),(0,20)))
    with BatchRenderer(source) as renderer:
        matrices = poses_to_tensor((Pose(ORIGIN),), device=renderer.device)
        value = contract(renderer.render(matrices, CONFIG, sampling="zone"), matrices, CONFIG, source)
        assert np.all(value["identity"] == 0)
        assert np.max(value["distance"]) < CONFIG.min_range_m
        processed = sample_depth(renderer,CONFIG,matrices,sampling="zone")
        assert np.isnan(host(processed.range_m)).all(), "too-close occluder exposed the far wall"
        records["eye_plane_and_too_close"] = {"zones":64,"max_raw_m":float(value["distance"].max())}
    # Although the hit is inside this narrow domain, other directions in the
    # zone have an unknown shorter prefix and cannot establish a nearest return.
    bounds = ((-1,10),(-.1,.1),(9.9,10.1))
    source = fixture(("off_center",((4,4.01),(.01,.02),(10.01,10.02))),bounds=bounds)
    with BatchRenderer(source) as renderer:
        matrices = poses_to_tensor((Pose(ORIGIN),), device=renderer.device)
        value = sample_depth(renderer,CONFIG,matrices,sampling="zone")
        assert np.isnan(host(value.range_m)[0,3,3])
        center_y = 4 * .125 / math.sqrt(1+2*.125**2)
        assert center_y > bounds[1][1]
        records["known_hit_unknown_zone_prefix"] = "NaN"
    source = fixture(("off_center",((4,4.01),(.01,.02),(10.01,10.02))),
                     bounds=((-1,10),(-5,5),(5,15)))
    with BatchRenderer(source) as renderer:
        matrices=poses_to_tensor((Pose(ORIGIN),),device=renderer.device)
        value=sample_depth(renderer,CONFIG,matrices,sampling="zone")
        assert np.isfinite(host(value.range_m)[0,3,3])
        records["known_hit_known_zone_prefix"] = float(host(value.range_m)[0,3,3])
    source = fixture(("outside_hit",((4,4.01),(.9,.92),(10.01,10.02))),
                     bounds=((-1,10),(-.6,.6),(5,15)))
    with BatchRenderer(source) as renderer:
        matrices=poses_to_tensor((Pose(ORIGIN),),device=renderer.device)
        value=sample_depth(renderer,CONFIG,matrices,sampling="zone")
        assert np.isnan(host(value.range_m)[0,3,3])
        records["center_reprojection_known_real_hit_outside"] = "NaN"
    one = DepthConfig(1,1,90,90,0,8)
    hidden = ("behind",((-2,-1),(-.1,.1),(9.9,10.1)))
    for name,bounds,classification in (
        ("center_known_edges_unknown",((-1,9),(-1,1),(9,11)),np.isnan),
        ("center_and_corners_known_edge_extremum_unknown",((-1,9),(-5,5),(1,19)),np.isnan),
        ("whole_cone_known_empty",((-1,9),(-9,9),(1,19)),np.isposinf),
    ):
        source=fixture(hidden,bounds=bounds)
        with BatchRenderer(source) as renderer:
            matrices=poses_to_tensor((Pose(ORIGIN),),device=renderer.device)
            value=sample_depth(renderer,one,matrices,sampling="zone")
            assert classification(host(value.range_m)).all(),name
            records[name]="NaN" if classification is np.isnan else "+inf"
    assert 8/math.sqrt(3) < 5 < 8/math.sqrt(2)
    return records


def verify_batch_ownership_and_updates():
    source=fixture(("probe",((4,4.01),(.2,.24),(10.2,10.24))))
    poses=tuple(Pose((.125*(i%3),0,10)) for i in range(8))
    with BatchRenderer(source) as renderer:
        matrices=poses_to_tensor(poses,device=renderer.device)
        raw=renderer.render(matrices,CONFIG,sampling="zone")
        value=contract(raw,matrices,CONFIG,source)
        saved={key:array.copy() for key,array in value.items()}
        reverse=raw_host(renderer.render(matrices.flip(0),CONFIG,sampling="zone"))
        check_equal({key:array[::-1] for key,array in reverse.items()},saved,0)
        for i in range(len(poses)):
            single=raw_host(renderer.render(matrices[i:i+1],CONFIG,sampling="zone"))
            check_equal(single,{key:array[i:i+1] for key,array in saved.items()},0)
        check_equal(raw_host(raw),saved,0)
        first=renderer.geometry_info
        renderer.render(poses_to_tensor((poses[0],)*256,device=renderer.device),CONFIG,sampling="zone")
        after=renderer.geometry_info
        for field in ("scene_geometry_buffer_name","scene_geometry_buffer_bytes","geometry_uploads"):
            assert first[field]==after[field],field
    check_equal(raw_host(raw),saved,0)
    ground=MeshObject("ground",((-50,-50,0),(50,-50,0),(50,50,0),(-50,50,0)),((0,1,2),(0,2,3)),"ground")
    background=MeshMap(Path("zone-dynamic-fixture.obj"),(ground,),size_xyz_m=(100,100,100))
    first=Scene(background,Sphere(Pose(ORIGIN),.2),Sphere(Pose((4,.4,10.4)),.1),7.5)
    moved=first.with_target_position((4,-.4,10.4))
    scenes=(first,moved,moved.with_target_position(first.target.pose.position_m))
    cold=[]
    for scene in scenes:
        with BatchRenderer(scene.get_geometry(exclude_ids={"vehicle"})) as renderer:
            matrices=poses_to_tensor((Pose(ORIGIN),),device=renderer.device)
            cold.append(raw_host(renderer.render(matrices,CONFIG,sampling="zone")))
    with BatchRenderer(first.get_geometry(exclude_ids={"vehicle"})) as renderer:
        matrices=poses_to_tensor((Pose(ORIGIN),),device=renderer.device)
        raw=renderer.render(matrices,CONFIG,sampling="zone")
        saved=raw_host(raw)
        initial=renderer.geometry_info
        for index,scene in enumerate(scenes[1:],1):
            snap=scene.get_geometry(exclude_ids={"vehicle"})
            assert snap.get_object("target").geometry.pose.position_m==scene.target.pose.position_m
            renderer.sync(snap)
            updated=renderer.render(matrices,CONFIG,sampling="zone")
            check_equal(raw_host(updated),cold[index],0)
            info=renderer.geometry_info
            assert info["geometry_uploads"]==initial["geometry_uploads"]==1
            assert info["scene_geometry_buffer_name"]==initial["scene_geometry_buffer_name"]
            assert info["object_transform_updates"]==initial["object_transform_updates"]+index
            assert info["object_transform_upload_bytes"]==initial["object_transform_upload_bytes"]+64*index
            check_equal(raw_host(raw),saved,0)
        for index,(value,scene) in enumerate(zip(cold,scenes)):
            cell=(3,3) if index!=1 else (3,4)
            target_index=raw_id(scene.get_geometry(exclude_ids={"vehicle"}),"target")
            target_cells={tuple(at) for at in np.argwhere(value["identity"][0]==target_index)}
            assert target_cells=={cell},target_cells
            center=np.asarray(scene.target.pose.position_m)-ORIGIN
            assert abs(value["distance"][(0,*cell)]-(np.linalg.norm(center)-.1))<2e-4
    return {"batch_views":8,"large_batch_views":256,"transform_updates":2,"transform_bytes":128,
            "target_cells_A_B_A":[[3,3],[3,4],[3,3]]}


def verify_screenshot_case():
    scene_config=load_config(ROOT/"configs"/"devtools.json")
    # Exact project-generated seed-3 map used by the original sensor review;
    # tracked here so the regression does not depend on ignored output files.
    path=ROOT/"tests"/"fixtures"/"sensor-review-pillars-20.obj"
    assert path.is_file(),f"existing scene missing: {path}"
    body=Pose((.004000000189989805,-17.604000091552734,1.5),
              (0,0,math.sin(math.radians(78.68399810791016)/2),math.cos(math.radians(78.68399810791016)/2)))
    target=(.7369999885559082,-13.317999839782717,1.5)
    scene=Scene.from_obj(path,scene_config,body,target,t_ref=1)
    source=scene.get_geometry(exclude_ids={"vehicle"})
    with BatchRenderer(source) as renderer:
        matrices=poses_to_tensor((body,),device=renderer.device)
        pixel=renderer.render(matrices,scene_config.sensors.tof)
        zone=renderer.render(matrices,scene_config.sensors.tof,sampling="zone")
        camera=sample_camera(renderer,scene_config.sensors.camera,matrices,scene_config.sensors.min_visible_fraction)
        target_index=raw_id(source,"target")
        pixel_count=int(np.count_nonzero(host(pixel.object_index)==target_index))
        target_cells={tuple(index) for index in np.argwhere(host(zone.object_index)[0]==target_index)}
        camera_count=int(np.count_nonzero(host(camera.target_mask)))
        assert pixel_count==0,pixel_count
        assert target_cells=={(3,3),(3,4),(4,3),(4,4)},target_cells
        assert camera_count==148,camera_count
        # Independent angular containment: the sphere crosses y=0 and z=0,
        # while remaining strictly inside the central two columns/rows.
        matrix=host(matrices)[0].astype(float)
        center=(np.asarray(target)-matrix[:3,3])@matrix[:3,:3]
        radius=scene_config.target_radius_m
        assert abs(center[1])<radius and abs(center[2])<radius
        boundary=math.tan(math.radians(scene_config.sensors.tof.hfov_deg)/2)/4
        assert abs(center[1])+radius < boundary*(center[0]-radius)
        assert abs(center[2])+radius < boundary*(center[0]-radius)
    return {"map_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"pixel_target_cells":pixel_count,
            "zone_target_cells":sorted([[int(value) for value in cell] for cell in target_cells]),
            "camera_visible_target_pixels":camera_count,"target_sensor_center_m":center.tolist(),
            "oracle":"analytic sphere angular support and original 8x8/160x120 outputs; no supersampling"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"outputs"/"sensor_review"/"tof-zones-verification.json")
    args=parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    records=[]
    for name,action in (("small_objects_and_boundaries",verify_small_objects_and_boundaries),
                        ("plane_and_occlusion",verify_plane_and_occlusion),
                        ("near_range_and_whole_cone_coverage",verify_near_and_coverage),
                        ("batch_ownership_transform_updates",verify_batch_ownership_and_updates),
                        ("exact_review_screenshot",verify_screenshot_case)):
        try:
            records.append({"name":name,"status":"passed","detail":action()})
            print(f"{name}: passed",flush=True)
        except Exception as error:
            records.append({"name":name,"status":"failed","error_type":type(error).__name__,
                            "error":str(error),"traceback":traceback.format_exc()})
            print(f"{name}: failed: {type(error).__name__}: {error}",flush=True)
    result={"status":"passed" if all(record["status"]=="passed" for record in records) else "failed",
            "device":torch.cuda.get_device_name(),"checks":records,
            "oracle":"closed-form analytic witnesses and independent active-constraint QP; no finer rasterization"}
    args.results.parent.mkdir(parents=True,exist_ok=True)
    args.results.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    raise SystemExit(0 if result["status"]=="passed" else 1)


if __name__=="__main__":
    main()
