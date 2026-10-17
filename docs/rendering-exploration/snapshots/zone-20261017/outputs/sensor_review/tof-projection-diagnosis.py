"""Isolated reproduction of browser sample #6; never calls the online server."""
from pathlib import Path
import json
import math
import sys
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from devtools.config import load_config
from devtools.scene import Pose, Scene
from devtools.rendering import BatchRenderer, _sphere_mesh
from devtools.sensors import poses_to_tensor, sample_rig_diagnostics

root = Path(__file__).resolve().parents[2]
out = Path(__file__).parent
config = load_config(root / 'configs/devtools.json')
yaw = math.radians(78.68399810791016)
body = Pose((0.004000000189989805, -17.604000091552734, 1.5),
            (0., 0., math.sin(yaw / 2), math.cos(yaw / 2)))
target = (0.7369999885559082, -13.317999839782717, 1.5)
scene = Scene.from_obj(root / 'outputs/sensor_perf/maps/pillars-20.obj',
                       config, body, target)
snapshot = scene.get_geometry(exclude_ids={'vehicle'})
poses = poses_to_tensor((body,))
with BatchRenderer(snapshot) as renderer, torch.no_grad():
    diag = sample_rig_diagnostics(renderer, config.sensors, poses)
    target_index = diag.tof_raw.object_ids.index('target')
    camera_mask = diag.camera_raw.object_index[0] == target_index
    rows, cols = torch.where(camera_mask)
    result = {
        'source': 'Current UI pose values; separate Docker renderer, no online API calls',
        'body_position_m': body.position_m, 'body_yaw_deg': math.degrees(yaw),
        'target_position_m': target, 'target_radius_m': config.target_radius_m,
        'center_distance_m': float(np.linalg.norm(np.array(target)-body.position_m)),
        'target_center_sensor_m': body.world_to_body(target),
        'target_id': target_index,
        'tof_target_counts': [(view == target_index).sum().item()
                              for view in diag.tof_raw.object_index],
        'front_tof_raw_ids': diag.tof_raw.object_index[1].cpu().tolist(),
        'camera_target_pixels': int(camera_mask.sum()),
        'camera_target_bbox_xyxy_exclusive': [int(cols.min()), int(rows.min()),
                                             int(cols.max())+1, int(rows.max())+1],
        'camera_raw_at_row60_col78_m': diag.camera_raw.distance_m[0,60,78].item(),
        'camera_visible_fraction': diag.rig.camera.visible_fraction.item(),
        'resolution_controls': [],
    }
    for size in (8, 16, 32, 64, 128, 256):
        depth = replace(config.sensors.tof, width=size, height=size)
        full = renderer.render(poses, depth)
        alone = renderer.render(poses, depth, only_object='target')
        full_mask = full.object_index[0] == target_index
        alone_mask = alone.object_index[0] == target_index
        case = {'width': size, 'height': size, 'full_target_pixels': int(full_mask.sum()),
                'target_only_pixels': int(alone_mask.sum())}
        if size == 256:
            zones = full_mask.reshape(8, 32, 8, 32).sum((1,3))
            case['visible_target_pixels_per_8x8_zone'] = zones.cpu().tolist()
        result['resolution_controls'].append(case)
    # Change only Z in this independent renderer, to test sampling alignment.
    shifted = scene.with_target_position((target[0], target[1], target[2]+0.32))
    renderer.sync(shifted.get_geometry(exclude_ids={'vehicle'}))
    shifted_raw = renderer.render(poses, config.sensors.tof)
    result['isolated_z_plus_0_32m_target_pixels'] = int((shifted_raw.object_index == target_index).sum())

# Project the exact formal sphere vertices; analytical sphere bounds are an
# independent upper bound since the actual mesh is inscribed in the sphere.
vertices, _ = _sphere_mesh(config.target_radius_m)
world = vertices + target
matrix = np.asarray(body.as_transform().matrix)
sensor = (world-matrix[:3,3]) @ matrix[:3,:3]
tangent = math.tan(math.radians(config.sensors.tof.vfov_deg)/2)
uv = np.column_stack((4*(1-sensor[:,1]/sensor[:,0]/tangent),
                      4*(1-sensor[:,2]/sensor[:,0]/tangent)))
result['projected_mesh_bbox_8x8_xyxy'] = [*uv.min(0).tolist(), *uv.max(0).tolist()]
center = np.asarray(body.world_to_body(target))
half_height = 4*config.target_radius_m/(tangent*math.sqrt(center[0]**2-config.target_radius_m**2))
result['analytic_sphere_vertical_span_8x8'] = [4-half_height, 4+half_height]
result['nearest_two_pixel_center_rows'] = [3.5, 4.5]
assert result['tof_target_counts'] == [0,0,0,0]
assert result['camera_target_pixels'] == 148
assert abs(result['camera_raw_at_row60_col78_m'] - 4.05362) < 0.0001
assert result['resolution_controls'][0]['target_only_pixels'] == 0
assert result['resolution_controls'][-1]['full_target_pixels'] > 0
assert result['isolated_z_plus_0_32m_target_pixels'] > 0
(out / 'tof-projection-diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')

# A small standalone diagram of the central 4x4 portion of the 8x8 image.
# Convex hull gives the actual projected silhouette of this convex mesh.
points = sorted(set(map(tuple, uv)))
def cross(o,a,b):
    return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
def half(seq):
    hull=[]
    for p in seq:
        while len(hull)>1 and cross(hull[-2],hull[-1],p)<=0: hull.pop()
        hull.append(p)
    return hull
hull = half(points)[:-1]+half(points[::-1])[:-1]
def px(v): return 60+(v-2)*105
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="540" height="550" viewBox="0 0 540 550">',
     '<rect width="540" height="550" fill="#10212a"/>',
     '<g font-family="sans-serif" fill="#e7f3f6"><text x="30" y="28" font-size="19">Sample #6: front ToF projection</text>',
     '<text x="30" y="49" font-size="13">Central 4 x 4 cells of the actual 8 x 8 image</text>']
for k in range(2,7):
    p=px(k)
    svg += [f'<path d="M {p},60 V 480 M 60,{p} H 480" stroke="#506773" fill="none"/>',
            f'<text x="{p-4}" y="501" font-size="12">{k}</text>',
            f'<text x="37" y="{p+4}" font-size="12">{k}</text>']
coords=' '.join(f'{px(u):.3f},{px(v):.3f}' for u,v in hull)
svg += [f'<polygon points="{coords}" fill="#ffb55d" fill-opacity=".75" stroke="#ffd99e" stroke-width="2"/>']
for y in np.arange(2.5,6,1):
    for x in np.arange(2.5,6,1):
        svg.append(f'<circle cx="{px(x)}" cy="{px(y)}" r="4" fill="#86e7e3"/>')
svg += ['<circle cx="40" cy="525" r="4" fill="#86e7e3"/><text x="54" y="530" font-size="13">Pixel centers</text>',
        '<rect x="235" y="516" width="14" height="14" fill="#ffb55d"/><text x="257" y="530" font-size="13">Projected target: no center covered</text></g></svg>']
(out/'tof-projection-grid.svg').write_text('\n'.join(svg))
print(json.dumps(result, indent=2))
