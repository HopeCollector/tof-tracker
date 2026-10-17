import json
from dataclasses import replace
import torch
from devtools.config import DepthConfig
from devtools.scene import Pose, Sphere, SceneObject, GeometrySnapshot
from devtools.rendering import BatchRenderer

config = DepthConfig(8,8,60,60,.05,12)
snapshot = GeometrySnapshot((SceneObject('custom-target', 'target', Sphere(Pose((3,0,0)), .1)),),12.5,None)
poses = torch.eye(4, device='cuda').repeat(2,1,1)
poses[1,1,3] = .01
with BatchRenderer(snapshot) as renderer:
    pixel = renderer.render(poses, config)
    zone = renderer.render(poses, config, sampling='zone')
    hits = zone.object_index >= 0
    counts = hits.sum((1,2))
    assert counts.tolist() == [4,4], counts
    assert (pixel.object_index >= 0).sum().item() == 0
    assert zone.hit_point_sensor_m.shape == (2,8,8,3)
    assert zone.triangle_index[hits].min().item() >= 0
    assert torch.allclose(zone.distance_m[hits], torch.linalg.vector_norm(zone.hit_point_sensor_m[hits], dim=-1), atol=1e-6,rtol=0)
    expected = zone.distance_m.clone()
    sphere = snapshot.objects[0].geometry
    moved = replace(snapshot, objects=(replace(snapshot.objects[0], geometry=replace(sphere, pose=Pose((3.1,.01,.02)))),))
    renderer.sync(moved)
    updated = renderer.render(poses, config, only_object='custom-target', sampling='zone')
    with BatchRenderer(moved) as cold:
        fresh = cold.render(poses, config, sampling='zone')
        assert torch.equal(updated.distance_m,fresh.distance_m)
        assert torch.equal(updated.hit_point_sensor_m,fresh.hit_point_sensor_m)
        assert torch.equal(updated.triangle_index,fresh.triangle_index)
    assert torch.equal(zone.distance_m, expected)
    assert renderer.geometry_info['geometry_uploads'] == 1
    assert renderer.geometry_info['object_transform_updates'] == 1
    assert renderer.memory_info['texture_array_count'] == 2
    print(json.dumps({'status':'passed','zone_counts':counts.tolist(),'pixel_count':0,'zone_ranges':zone.distance_m[hits].tolist(),'triangles':zone.triangle_index[hits].tolist(),'memory':renderer.memory_info}))
assert torch.equal(zone.distance_m, expected)
