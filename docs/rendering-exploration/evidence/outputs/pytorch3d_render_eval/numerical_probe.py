"""Scratch diagnostic: standard API settings, no PyTorch3D source changes."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, '/workspace')
import experiments.pytorch3d_render_eval.backend as backend
from experiments.pytorch3d_render_eval.fixtures import CONFIG, load_scene, make_poses, poses_tensor, depth_comparison
from experiments.pytorch3d_render_eval.postprocess import postprocess_depth
from devtools.sensors import MeshSensor

torch.set_grad_enabled(False)
scene, _ = load_scene(20)
bounds = scene.get_geometry().map_bounds_m
report = []
with MeshSensor(scene) as baseline:
    for mode, size in [('tof',8),('camera',4)]:
        config = getattr(CONFIG.sensors,mode)
        poses = make_poses(size,mode)
        matrices = poses_tensor(poses)
        frames = baseline.sample_many(config,poses)
        expected = np.stack([f.range_m for f in frames])
        expected_id = np.stack([f.object_index for f in frames])
        for scale, clip, blur in [(1,False,0),(1,True,0),(1,True,1e-12),(1000,True,0),(1000,True,1e-12)]:
            backend.Z_CLIP_M = 1e-5 * scale
            renderer = backend.P3DRenderer(scene)
            if scale!=1:
                renderer._base_full.verts_list()[0].mul_(scale)
                renderer._base_target.verts_list()[0].mul_(scale)
            renderer.prepare(size,include_target=False)
            local_config=replace(config,min_range_m=config.min_range_m*scale,max_range_m=config.max_range_m*scale)
            raster=renderer._view(local_config,False)[3]
            raster.raster_settings=replace(raster.raster_settings,clip_barycentric_coords=clip,blur_radius=blur)
            local_poses=matrices.clone()
            local_poses[:,:3,3]*=scale
            distance,identity=renderer.render(local_poses,local_config)
            distance=distance/scale
            ranges=postprocess_depth(distance,identity,matrices,config,bounds)
            comparison=depth_comparison(ranges.cpu().numpy(),expected,identity.cpu().numpy(),expected_id)
            item=dict(mode=mode,scale=scale,clip_barycentric=clip,blur=blur,**comparison)
            report.append(item)
            print(json.dumps(item),flush=True)
            renderer.release_batch()
            del renderer,distance,identity,ranges
Path('/workspace/outputs/pytorch3d_render_eval/numerical_probe.json').write_text(json.dumps(report,indent=2)+'\n')
