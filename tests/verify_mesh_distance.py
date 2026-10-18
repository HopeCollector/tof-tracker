"""Independent analytic checks for offline mesh distances; run inside Docker.

No SdfLib, Torch, trimesh proximity, or production rectangle-distance helper is
used as the geometry reference. The public ESDF builder also gets analytic node,
query and sign checks on resolved box fixtures, plus a small end-to-end plane
check. Optional --map exercises the complete public map input, including a
free-space ESDF accuracy check.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from itertools import product
import json
from pathlib import Path
import tempfile

import numpy as np

from devtools.config import load_config
from devtools.mesh import MeshObject
from devtools.esdf.mesh_distance import MeshDistanceQuery, PreparedMesh
from devtools.scene import GeometrySnapshot, Pose, Scene, SceneObject, Sphere


def box(name, lo, hi):
    vertices = list(product(*zip(lo, hi)))
    indices = {point: index for index, point in enumerate(vertices)}
    faces = []
    for axis in range(3):
        u, v = (axis+1) % 3, (axis+2) % 3
        for high in (False, True):
            corners = []
            for pu,pv in ((0,0),(1,0),(1,1),(0,1)):
                point = list(lo)
                point[axis] = hi[axis] if high else lo[axis]
                point[u], point[v] = (lo[u],hi[u])[pu], (lo[v],hi[v])[pv]
                corners.append(indices[tuple(point)])
            if not high:
                corners.reverse()
            faces.extend(((corners[0],corners[1],corners[2]), (corners[0],corners[2],corners[3])))
    return MeshObject(name, tuple(vertices), tuple(faces), "obstacle")


def ground():
    return MeshObject("ground", ((-3.,-3.,0.),(3.,-3.,0.),(3.,3.,0.),(-3.,3.,0.)),
                      ((0,1,2),(0,2,3)), "ground")


def snapshot(*objects):
    return GeometrySnapshot(tuple(SceneObject(o.name, o.kind, o) for o in objects), None, None)


def box_distance(points, lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    q = np.abs(points-(lo+hi)*.5) - (hi-lo)*.5
    return np.linalg.norm(np.maximum(q,0), axis=1) + np.minimum(q.max(axis=1),0)


def cross_distance(points):
    polygon = np.array([[-1,-.2],[-.2,-.2],[-.2,-1],[.2,-1],[.2,-.2],[1,-.2],
                        [1,.2],[.2,.2],[.2,1],[-.2,1],[-.2,.2],[-1,.2]])
    distance = np.full(len(points), np.inf)
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge = b-a
        t = np.clip((points[:,:2]-a)@edge/(edge@edge),0,1)
        distance = np.minimum(distance,np.linalg.norm(points[:,:2]-a-t[:,None]*edge,axis=1))
    xy = np.abs(points[:,:2])
    inside = ((xy[:,0] <= 1) & (xy[:,1] <= .2)) | ((xy[:,0] <= .2) & (xy[:,1] <= 1))
    dxy = distance*np.where(inside,-1,1)
    q = np.stack((dxy,np.abs(points[:,2])-1),axis=1)
    return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)


def builder_check(library_path):
    """Exercise the moved callable composition with an independent z plane."""
    import torch
    from devtools.esdf import ESDFGrid, build_esdf_from_mesh

    geometry = ground()
    text = "# unit metre\n# size_xyz 6 6 2\no ground\n"
    text += "".join("v " + " ".join(map(str,p)) + "\n" for p in geometry.vertices_m)
    text += "".join("f " + " ".join(str(i+1) for i in face) + "\n" for face in geometry.triangles)
    declared = ((-3.,3.),(-3.,3.),(0.,2.))
    snap = GeometrySnapshot(snapshot(geometry).objects,None,declared)
    with tempfile.TemporaryDirectory(prefix="esdf-component-") as directory:
        root = Path(directory)
        obj = root/"plane.obj"; obj.write_text(text)
        field = build_esdf_from_mesh(obj,library_path=library_path,voxel_size_m=.5,chunk_size=17)
        other = build_esdf_from_mesh(snap,library_path=library_path,voxel_size_m=.5,chunk_size=4096)
        assert field.bounds_m == ((-3.,-3.,0.),(3.,3.,2.))
        assert torch.equal(field.samples,other.samples)
        # Every node, including the halo, has exact signed distance z.
        z = field.sample_origin_m[2]+np.arange(field.shape[2])*field.spacing_m
        expected = np.broadcast_to(z,field.shape).astype(np.float32)
        np.testing.assert_array_equal(field.samples.numpy(),expected)
        nonaligned = build_esdf_from_mesh(snap,library_path=library_path,
            bounds_m=[[-.9,-.8,.1],[1.13,1.27,1.38]],voxel_size_m=.3,chunk_size=19)
        z = nonaligned.sample_origin_m[2]+np.arange(nonaligned.shape[2])*nonaligned.spacing_m
        np.testing.assert_allclose(nonaligned.samples.numpy(),np.broadcast_to(z,nonaligned.shape),atol=1e-7,rtol=0)
        for source in (snapshot(geometry),root/"no-domain.obj"):
            if isinstance(source,Path): source.write_text(text.replace("# size_xyz 6 6 2\n",""))
            try:
                build_esdf_from_mesh(source,library_path=library_path)
            except ValueError as error:
                assert "bounds_m" in str(error)
            else: raise AssertionError("missing declared domain accepted")
        explicit = build_esdf_from_mesh(snapshot(geometry),library_path=library_path,
            bounds_m=[[-1.,-1.,0.],[1.,1.,1.]],voxel_size_m=.25)
        assert explicit.spacing_m == .25
        sphere = GeometrySnapshot((SceneObject("target","target",Sphere(Pose((0.,0.,0.)),.2)),),None,declared)
        try: build_esdf_from_mesh(sphere,library_path=library_path)
        except ValueError: pass
        else: raise AssertionError("sphere accepted into static mesh builder")
        field.save(root/"saved")
        loaded = ESDFGrid.load(root/"saved")
        p = torch.tensor([[.12,-.3,.31],[1.,1.,1.]],dtype=torch.float64,requires_grad=True)
        distance = loaded.query(p)
        gradient, = torch.autograd.grad(distance.sum(),p)
        torch.testing.assert_close(distance,p[:,2],atol=1e-13,rtol=0)
        torch.testing.assert_close(gradient,p.new_tensor([[0,0,1],[0,0,1]]),atol=1e-13,rtol=0)
        absent = root/"not-compiled.so"
        try: build_esdf_from_mesh(obj,library_path=absent,voxel_size_m=.5)
        except FileNotFoundError: pass
        else: raise AssertionError("missing native library accepted")
        assert not absent.exists()  # No implicit compilation or cache creation.
    return {"passed":True,"obj_snapshot_nodes_identical":True,"all_halo_nodes_analytic":True,
            "nonaligned_bounds_and_configurable_spacing":True,"save_load_position_gradient":True,
            "explicit_native_library":True,"shape":field.shape}


def esdf_map_accuracy(library_path):
    """Compare built grids with analytic distances on resolved box fixtures.

    These checks use a single box and two boxes separated by two voxels: every
    node must match the analytic distance, and trilinear query values must stay
    within their eight corner values. The half-voxel distance and sign
    thresholds below are regression checks for these fixtures, not a general
    geometric error guarantee. Thin features can remain unresolved.
    """
    import torch
    from devtools.esdf import build_esdf_from_mesh

    def bounded(objects, bounds):
        return GeometrySnapshot(tuple(SceneObject(o.name,o.kind,o) for o in objects),None,bounds)

    def measure(name, objects, bounds, spacing):
        field = build_esdf_from_mesh(bounded(objects,bounds),library_path=library_path,
            voxel_size_m=spacing)
        indices = np.stack(np.meshgrid(*(np.arange(n) for n in field.shape),indexing="ij"),axis=-1)
        grid = np.asarray(field.sample_origin_m)+indices.reshape(-1,3)*spacing
        extents = [(np.min(o.vertices_m,axis=0),np.max(o.vertices_m,axis=0)) for o in objects]
        node_expected = np.minimum.reduce([box_distance(grid,*b) for b in extents])
        node_error = float(np.max(np.abs(field.samples.numpy().ravel()-node_expected),initial=0))
        assert node_error <= 2e-6, (name,"node",node_error)

        rng = np.random.default_rng(20261018)
        lower, upper = np.asarray(bounds)[:,0],np.asarray(bounds)[:,1]
        points = [rng.uniform(lower,upper,(50000,3))]
        for lo, hi in extents:
            near = rng.uniform(lo-spacing,hi+spacing,(10000,3))
            for axis in range(3):
                near[:,axis] = np.where(rng.random(len(near)) < .5,
                    np.where(rng.random(len(near)) < .5,lo[axis],hi[axis]),near[:,axis])
            points.append(near)
        points = np.vstack(points)
        points = points[np.all((points >= lower) & (points <= upper),axis=1)]
        expected = np.minimum.reduce([box_distance(points,*b) for b in extents])
        actual, valid = field.query(torch.tensor(points,dtype=torch.float64),return_valid=True)
        actual, valid = actual.numpy(), valid.numpy()
        assert valid.all()
        error = np.abs(actual-expected)
        maximum = float(error.max())
        assert maximum <= .5*spacing+2e-6, (name,"query",maximum,spacing)
        mismatch = (np.sign(actual) != np.sign(expected)) & (np.abs(expected) > .5*spacing)
        assert not mismatch.any(), (name,"sign",int(mismatch.sum()))
        samples = field.samples.numpy()
        coordinate = (points-np.asarray(field.sample_origin_m))/spacing
        base = np.maximum(1,np.minimum(np.floor(coordinate).astype(int),
            np.asarray(samples.shape)-3))
        nearby = np.stack([samples[base[:,0]+dx,base[:,1]+dy,base[:,2]+dz]
            for dx,dy,dz in product(range(2),repeat=3)],axis=1)
        # The eight nonnegative trilinear weights preserve the corner range.
        tolerance = 64*np.finfo(actual.dtype).eps*np.maximum(1,np.abs(nearby).max(axis=1))
        assert np.all(actual >= nearby.min(axis=1)-tolerance)
        assert np.all(actual <= nearby.max(axis=1)+tolerance)
        all_positive, all_negative = nearby.min(axis=1) > 0, nearby.max(axis=1) < 0
        negative_overshoot = int(((actual < -tolerance) & all_positive).sum())
        positive_overshoot = int(((actual > tolerance) & all_negative).sum())
        assert not negative_overshoot and not positive_overshoot, \
            (name,"overshoot",negative_overshoot,positive_overshoot)
        return {"spacing_m":spacing,"node_count":len(grid),"query_points":len(points),
            "node_max_abs_error_m":node_error,"query_max_abs_error_m":maximum,
            "query_max_error_in_voxels":maximum/spacing,
            "sign_mismatches_outside_half_voxel":int(mismatch.sum()),
            "near_surface_sign_mismatches":int(((np.sign(actual) != np.sign(expected))
                & (np.abs(expected) <= .5*spacing)).sum()),
            "all_positive_node_negative_values":negative_overshoot,
            "all_negative_node_positive_values":positive_overshoot}

    unit = box("unit",(0.,0.,0.),(1.,1.,1.))
    gap = box("gap",(1.4,0.,0.),(2.4,1.,1.))
    return {"single_box":measure("single_box",(unit,),((-2.,3.),(-2.,3.),(-2.,3.)),.2),
            "two_box_gap":measure("two_box_gap",(unit,gap),((-2.,4.),(-2.,3.),(-2.,3.)),.2)}


def run(library_path, map_path=None, esdf_voxel=.5):
    rng = np.random.default_rng(1843)
    results = {}
    bounds = np.array([[-2.,-2.,-.5],[2.,2.,2.]])

    def checked(name, prepared, points, expected, tolerance=2e-12):
        with MeshDistanceQuery(prepared.vertices_m, prepared.triangles, library_path=library_path) as query:
            actual = query(points)
        error = float(np.max(np.abs(actual-expected),initial=0))
        assert np.isfinite(actual).all() and error <= tolerance, (name,error)
        results[name] = {"count": len(points), "max_abs_error_m": error}
        return actual

    unit = box("box",(-1.,-1.,-1.),(1.,1.,1.))
    prepared = PreparedMesh.from_geometry(snapshot(unit),query_bounds_m=bounds)
    p = rng.uniform(-2,2,(1024,3))
    p = np.vstack((p, [[0,0,0],[1,0,0],[1,1,0],[1,1,1],[1+1e-10,0,0],[1-1e-10,0,0]]))
    checked("analytic_box_faces_edges_vertices",prepared,p,box_distance(p,(-1,-1,-1),(1,1,1)))

    angle = .371
    rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    rotated = MeshObject("rotated",tuple(map(tuple,np.asarray(unit.vertices_m)@rotation.T)),unit.triangles,"obstacle")
    arbitrary = PreparedMesh.from_geometry(snapshot(rotated),query_bounds_m=bounds)
    checked("single_arbitrary_closed_mesh",arbitrary,p,box_distance(p@rotation,(-1,-1,-1),(1,1,1)))

    cross = PreparedMesh.from_geometry(snapshot(box("x",(-1.,-.2,-1.),(1.,.2,1.)),
        box("y",(-.2,-1.,-1.),(.2,1.,1.))), query_bounds_m=bounds)
    p = np.vstack((rng.uniform(-1.5,1.5,(1536,3)),[[0,0,0],[.1,.05,0],[.2,.2,0]]))
    checked("overlap_union_extruded_cross",cross,p,cross_distance(p))
    assert abs(cross_distance(np.zeros((1,3)))[0]+np.sqrt(.08)) < 1e-15

    touching = PreparedMesh.from_geometry(snapshot(box("left",(-1.,-1.,-1.),(0.,1.,1.)),
        box("right",(0.,-1.,-1.),(1.,1.,1.))),query_bounds_m=bounds)
    duplicate = PreparedMesh.from_geometry(snapshot(unit,box("copy",(-1.,-1.,-1.),(1.,1.,1.)),
        box("nested",(-.2,-.2,-.2),(.2,.2,.2))),query_bounds_m=bounds)
    for label, mesh in (("touching_face_union",touching),("duplicate_nested_union",duplicate)):
        checked(label,mesh,p,box_distance(p,(-1,-1,-1),(1,1,1)))

    ground_only = PreparedMesh.from_geometry(snapshot(ground()),query_bounds_m=bounds)
    p = rng.uniform(bounds[0],bounds[1],(1024,3))
    xy = rng.uniform(-2,2,(100,2))
    near = np.concatenate([np.column_stack((xy,np.full(len(xy),z))) for z in (0.,1e-10,-1e-10,1e-7,-1e-7)])
    p = np.vstack((p,near))
    checked("ground_halfspace_including_near_zero",ground_only,p,p[:,2])
    pillar_ground = PreparedMesh.from_geometry(snapshot(box("pillar",(-1.,-1.,0.),(1.,1.,2.)),ground()),query_bounds_m=bounds)
    p = np.array([[0.,0.,0.],[0.,0.,.1],[0.,0.,1.],[0.,0.,2.],[1.,0.,0.],[1.2,0.,.2],[0.,0.,-.1]])
    expected = np.array([-1.,-1.,-1.,0.,0.,.2,-np.sqrt(1.01)])
    checked("ground_union_removes_internal_bottom",pillar_ground,p,expected)

    # Long, obtuse triangle fans on the ground plus negative/near-zero queries.
    # These stress the stable selected-triangle residual used in the v4 fix.
    near_pillar = box("small",(2.,1.1680832136,0.),(2.3780330085230643,2.0390345914,3.))
    fan_mesh = PreparedMesh.from_geometry(snapshot(near_pillar,ground()),query_bounds_m=np.array([[-30.,-30.,-.1],[30.,30.,15.]]))
    p = np.array([[2.32500005,1.22500002,0.],[2.2943529276689634,1.2560906879737153,.004218114497394696],
                  [-10.1875,14.462499618530273,0.],[-10.1875,14.462499618530273,1e-7],[-10.1875,14.462499618530273,-1e-7]])
    expected = p[:,2].copy()
    expected[:2] = p[:2,0]-2.3780330085230643
    checked("obtuse_fans_near_ground_regression",fan_mesh,p,expected)

    def rejects(name, function, exception=(ValueError,RuntimeError)):
        try:
            function()
        except exception:
            results[name] = {"rejected": True}
        else:
            raise AssertionError(f"{name}: accepted invalid input")

    vertices, triangles = np.array(unit.vertices_m), np.array(unit.triangles)
    rejects("reject_open_mesh",lambda:MeshDistanceQuery(vertices,triangles[:-1],library_path=library_path))
    rejects("reject_inward_mesh",lambda:MeshDistanceQuery(vertices,triangles[:,::-1],library_path=library_path))
    broken = vertices.copy(); broken[1] = broken[0]
    rejects("reject_degenerate_mesh",lambda:MeshDistanceQuery(broken,triangles,library_path=library_path))
    rejects("reject_out_of_range_index",lambda:MeshDistanceQuery(vertices,triangles+99,library_path=library_path))
    rejects("reject_float_indices",lambda:MeshDistanceQuery(vertices,triangles.astype(float),library_path=library_path))
    rejects("reject_nonfinite_mesh",lambda:MeshDistanceQuery(vertices*np.nan,triangles,library_path=library_path))
    rejects("reject_duplicate_face",lambda:MeshDistanceQuery(vertices,np.vstack((triangles,triangles[:1])),library_path=library_path))
    rejects("reject_arbitrary_mesh_mixture",lambda:PreparedMesh.from_geometry(snapshot(rotated,ground()),query_bounds_m=bounds))
    sphere = GeometrySnapshot((SceneObject("target","target",Sphere(Pose((0.,0.,0.)),.2)),),None,None)
    rejects("reject_sphere_snapshot",lambda:PreparedMesh.from_geometry(sphere,query_bounds_m=bounds))
    query = MeshDistanceQuery(vertices,triangles,library_path=library_path)
    assert query(np.empty((0,3))).shape == (0,)
    rejects("reject_nonfinite_query",lambda:query(np.array([[np.nan,0,0]])))
    rejects("reject_finite_overflow_query",lambda:query(np.array([[1e200,1e200,1e200]])))
    # Bypass Python validation to verify that native errors stay inside the ABI.
    raw_point = np.array([[np.inf,0.,0.]])
    raw_result = np.empty(1)
    pointer = ctypes.POINTER(ctypes.c_double)
    error = ctypes.create_string_buffer(256)
    status = query._library.mdq_query(query._handle,raw_point.ctypes.data_as(pointer),1,
        raw_result.ctypes.data_as(pointer),error,len(error))
    assert status != 0 and b"finite" in error.value
    results["native_nonfinite_error_contained"] = {"passed": True}
    with ThreadPoolExecutor(max_workers=2) as executor:
        values = list(executor.map(query,[np.array([[0.,0.,0.]]),np.array([[2.,0.,0.]])]))
    assert values[0][0] == -1 and values[1][0] == 1
    results["serialized_thread_queries"] = {"passed": True}
    vertices[:] = 9
    assert query(np.zeros((1,3)))[0] == -1  # Native BVH owns its input.
    with MeshDistanceQuery(ground_only.vertices_m,ground_only.triangles,library_path=library_path) as other:
        assert other(np.array([[0.,0.,.5]]))[0] == .5 and query(np.zeros((1,3)))[0] == -1
    query.close(); query.close()
    rejects("reject_closed_query",lambda:query(np.zeros((1,3))))
    results["lifetime_empty_multiple_meshes"] = {"passed": True}

    if map_path:
        scene = Scene.from_obj(map_path,load_config("configs/devtools.json"),Pose((0.,0.,1.)),(0.,0.,1.))
        snap = scene.get_geometry(exclude_ids={"vehicle","target"})
        logical = np.asarray(snap.map_bounds_m).T
        mesh = PreparedMesh.from_geometry(snap,query_bounds_m=logical+np.array([[-.2,-.2,-.2],[.2,.2,.2]]))
        boxes = np.asarray(mesh.metadata["boxes_m"])
        p = rng.uniform(logical[0],logical[1],(2048,3))
        expected = np.minimum.reduce([box_distance(p,*b) for b in boxes]+[p[:,2]])
        free = expected >= 0
        checked("complete_map_free_space_vs_analytic_boxes_and_ground",mesh,p[free],expected[free],2e-10)
        results["complete_map_geometry"] = {key:mesh.metadata[key] for key in
            ("obstacle_count","overlap_pair_count","rectangle_count","vertex_count","triangle_count","closed_oriented_edge_check")}

        import torch
        from devtools.esdf import build_esdf_from_mesh

        def union_distance(points):
            expected = np.asarray(points)[:,2].copy()
            for box_bounds in boxes:
                np.minimum(expected,box_distance(points,*box_bounds),out=expected)
            return expected

        esdf = build_esdf_from_mesh(snap,library_path=library_path,voxel_size_m=esdf_voxel)
        indices = np.stack(np.meshgrid(*(np.arange(n) for n in esdf.shape),indexing="ij"),axis=-1)
        grid = np.asarray(esdf.sample_origin_m)+indices.reshape(-1,3)*esdf.spacing_m
        node_expected = union_distance(grid)
        node_free = node_expected > 0
        node_error = float(np.max(np.abs(esdf.samples.numpy().ravel()[node_free]
            -node_expected[node_free]),initial=0))
        assert node_error <= 2e-6, ("map node",node_error)
        q = rng.uniform(logical[0],logical[1],(20000,3))
        query_expected = union_distance(q)
        query_free = query_expected > .5*esdf.spacing_m
        actual, valid = esdf.query(torch.tensor(q[query_free],dtype=torch.float64),return_valid=True)
        actual, valid = actual.numpy(), valid.numpy()
        assert valid.all()
        error = np.abs(actual-query_expected[query_free])
        maximum = float(error.max())
        # This map-specific threshold does not bound every ESDF geometry.
        assert maximum <= .5*esdf.spacing_m+2e-6, ("map query",maximum,esdf.spacing_m)
        mismatch = np.sign(actual) != np.sign(query_expected[query_free])
        assert not mismatch.any(), ("map sign",int(mismatch.sum()))
        samples = esdf.samples.numpy()
        coordinate = (q[query_free]-np.asarray(esdf.sample_origin_m))/esdf.spacing_m
        base = np.maximum(1,np.minimum(np.floor(coordinate).astype(int),
            np.asarray(samples.shape)-3))
        nearby = np.stack([samples[base[:,0]+dx,base[:,1]+dy,base[:,2]+dz]
            for dx,dy,dz in product(range(2),repeat=3)],axis=1)
        # The eight nonnegative trilinear weights preserve the corner range.
        tolerance = 64*np.finfo(actual.dtype).eps*np.maximum(1,np.abs(nearby).max(axis=1))
        assert np.all(actual >= nearby.min(axis=1)-tolerance)
        assert np.all(actual <= nearby.max(axis=1)+tolerance)
        all_positive, all_negative = nearby.min(axis=1) > 0, nearby.max(axis=1) < 0
        negative_overshoot = int(((actual < -tolerance) & all_positive).sum())
        positive_overshoot = int(((actual > tolerance) & all_negative).sum())
        assert not negative_overshoot and not positive_overshoot, \
            ("map overshoot",negative_overshoot,positive_overshoot)
        results["complete_map_esdf_accuracy"] = {"voxel_size_m":esdf.spacing_m,
            "shape":list(esdf.shape),"free_nodes":int(node_free.sum()),
            "free_node_max_abs_error_m":node_error,"query_points":len(actual),
            "query_max_abs_error_m":maximum,
            "query_max_error_in_voxels":maximum/esdf.spacing_m,
            "sign_mismatches_outside_half_voxel":int(mismatch.sum()),
            "all_positive_node_negative_values":negative_overshoot,
            "all_negative_node_positive_values":positive_overshoot}
    results["public_esdf_builder"] = builder_check(library_path)
    results["esdf_map_accuracy"] = esdf_map_accuracy(library_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--map",type=Path)
    parser.add_argument("--esdf-voxel",type=float,default=.5)
    args = parser.parse_args()
    print(json.dumps({"checks":run(args.library,args.map,args.esdf_voxel)},indent=2,allow_nan=False))
