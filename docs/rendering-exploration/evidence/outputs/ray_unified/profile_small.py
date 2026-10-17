import cProfile
import pstats
import torch
from experiments.unified_ray_perf.benchmark import make_poses
from devtools.config import load_config
from devtools.scene import Scene
from devtools.ray_rendering import RayRenderer
from devtools.sensors import poses_to_tensor, sample_depth

torch.set_num_threads(4)
torch.set_grad_enabled(False)
config = load_config('configs/devtools.json')
poses_cpu = make_poses(1)
scene = Scene.from_obj('outputs/sensor_perf/maps/pillars-20.obj', config, poses_cpu[0],
                       poses_cpu[0].body_to_world((3.,0.,0.)))
poses = poses_to_tensor(poses_cpu)
with RayRenderer(scene.get_geometry(exclude_ids={'vehicle'})) as renderer:
    def action():
        return sample_depth(renderer, config.sensors.tof, poses, reduction='weighted_quantile')
    for _ in range(5):
        action()
        torch.cuda.synchronize()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        action()
        torch.cuda.synchronize()
    profiler.disable()
    with open('outputs/ray_unified/small-cpu-profile.txt','w') as stream:
        pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(60)
