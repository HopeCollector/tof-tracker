import json
import torch
from devtools.config import SamplingRuntimeConfig
from devtools.scene import GeometrySnapshot, SceneObject, Sphere, Pose
from devtools.ray_rendering import RayRenderer
from devtools._ray_budget import plan

snapshot = GeometrySnapshot((SceneObject('target', 'target', Sphere(Pose((3., 0., 0.)), .5)),),
                            0., ((-10., 10.),) * 3)
with RayRenderer(snapshot) as renderer:
    origins = torch.tensor([[0., 0., 0.], [0., 1., 0.]], device='cuda')
    directions = torch.tensor([[1., 0., 0.]] * 4, device='cuda')
    for dtype in (torch.int32, torch.int64):
        index = torch.tensor([0, 1, -1, 5], dtype=dtype, device='cuda')
        hits = renderer.trace(origins, directions, 4., origin_indices=index)
        assert hits.distance_m.cpu().tolist() == [2.5, 0., 0., 0.]
        assert hits.object_index.cpu().tolist() == [0, -1, -1, -1]
    shared = renderer.trace(origins[:1], directions, torch.full((4,), 4., device='cuda'))
    assert shared.distance_m.cpu().tolist() == [2.5] * 4
    count, info = plan(renderer, 10, 1024, 10000, SamplingRuntimeConfig())
    assert count == 10 and info['chunk_count'] == 1
    try:
        plan(renderer, 10, 1024, 10000, SamplingRuntimeConfig(1e-8))
    except MemoryError:
        pass
    else:
        raise AssertionError('tiny budget accepted')
    print(json.dumps({'passed': True, 'compact_dtypes': ['int32', 'int64'],
                      'shared_origin': True, 'budget': info, 'core': renderer.geometry_info}))
