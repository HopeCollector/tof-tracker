"""Scratch: clip view-space triangles with stock Torch3D helper, then raster."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,'/workspace')
from pytorch3d import _C
from pytorch3d.renderer.mesh.clip import clip_faces,ClipFrustum
from experiments.pytorch3d_render_eval.backend import P3DRenderer
from experiments.pytorch3d_render_eval.fixtures import CONFIG,load_scene,make_poses,poses_tensor,depth_comparison
from experiments.pytorch3d_render_eval.postprocess import postprocess_depth
from devtools.sensors import MeshSensor

torch.set_grad_enabled(False)
scene,_=load_scene(20)
bounds=scene.get_geometry().map_bounds_m
report=[]
with MeshSensor(scene) as baseline:
 for mode,batch in [('tof',8),('camera',4)]:
  config=getattr(CONFIG.sensors,mode)
  poses=make_poses(batch,mode);matrices=poses_tensor(poses)
  expected=baseline.sample_many(config,poses)
  expected_range=np.stack([x.range_m for x in expected]);expected_id=np.stack([x.object_index for x in expected])
  renderer=P3DRenderer(scene);renderer.prepare(batch,include_target=False)
  mesh=renderer._batch_full
  sensor=torch.bmm(mesh.verts_padded().double()-matrices[:,:3,3].double()[:,None],matrices[:,:3,:3].double())
  view=sensor[:,:,[1,2,0]]
  faceverts=view.reshape(-1,3)[mesh.faces_packed()]
  previous_dtype=torch.get_default_dtype()
  torch.set_default_dtype(torch.float64)
  try:
   clipped=clip_faces(faceverts,mesh.mesh_to_faces_packed_first_idx(),mesh.num_faces_per_mesh(),ClipFrustum(cull=False,z_clip_value=1e-5,perspective_correct=False))
  finally: torch.set_default_dtype(previous_dtype)
  points=clipped.face_verts
  focal=renderer._view(config,False)[0][0].double()
  ndc=torch.stack((points[...,0]/points[...,2]*focal[0],points[...,1]/points[...,2]*focal[1],points[...,2]*1000),-1).float().contiguous()
  neighbor=clipped.clipped_faces_neighbor_idx
  if neighbor is None: neighbor=torch.full((ndc.shape[0],),-1,device='cuda',dtype=torch.int64)
  for blur in [0.,1e-12]:
   faces,zbuf,bary,dists=_C.rasterize_meshes(ndc,clipped.mesh_to_face_first_idx,clipped.num_faces_per_mesh,neighbor,(config.height,config.width),blur,1,8 if max(config.width,config.height)<=64 else 16,32768,True,True,False)
   face=faces[...,0]
   orig=face
   if clipped.faces_clipped_to_unclipped_idx is not None:
    orig=clipped.faces_clipped_to_unclipped_idx[face.clamp_min(0)]
   orig=orig.remainder(renderer.triangle_count)
   hit=(face>=0)
   ids=torch.where(hit,renderer._full_face_ids[orig],-1)
   norm=renderer._view(config,False)[2]
   depth=zbuf[...,0]/1000*norm
   for plane in [False,True]:
    distance=depth
    if plane:
     v=renderer._base_full.verts_packed().double();f=renderer._base_full.faces_packed()
     triangles=v[f][orig]
     n=torch.linalg.cross(triangles[...,1,:]-triangles[...,0,:],triangles[...,2,:]-triangles[...,0,:])
     from experiments.pytorch3d_render_eval.postprocess import _pixels
     rays=_pixels(config,matrices.device)[0]
     directions=torch.matmul(rays.reshape(-1,3),matrices[:,:3,:3].double().transpose(-1,-2)).reshape(batch,config.height,config.width,3)
     distance=((n*(triangles[...,0,:]-matrices[:,:3,3].double()[:,None,None])).sum(-1)/(n*directions).sum(-1)).float()
    shell=config.max_range_m+max(config.max_range_m*.001,4*32*torch.finfo(torch.float32).eps*max(1,config.max_range_m))
    valid=hit&torch.isfinite(distance)&(distance>=0)&(distance<=shell)
    raw=torch.where(valid,distance,0);identity=torch.where(valid,ids,-1)
    result=postprocess_depth(raw,identity,matrices,config,bounds)
    item=dict(mode=mode,blur=blur,plane_reconstruct=plane,**depth_comparison(result.cpu().numpy(),expected_range,identity.cpu().numpy(),expected_id))
    report.append(item);print(json.dumps(item),flush=True)
  renderer.release_batch()
Path('/workspace/outputs/pytorch3d_render_eval/view_clip_probe.json').write_text(json.dumps(report,indent=2)+'\n')
