"""Render one scene to four ToF images and a visible-target camera example.

Run in the project Docker container. All sensor settings are development
assumptions, not hardware specifications. The output directory must be new.
"""

import argparse
from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from devtools.config import load_config
from devtools.scene import Pose, Scene
from devtools.rendering import BatchRenderer
from devtools.sensors import TOF_ORDER, mounted_pose, poses_to_tensor, sample_depth, sample_rig
from tof_tracker.geometry import quaternion_from_rpy_degrees


def _statistics(values):
    finite = np.isfinite(values)
    valid = values[finite]
    return {"shape": list(values.shape), "finite": int(finite.sum()),
            "positive_infinity": int(np.isposinf(values).sum()),
            "nan": int(np.isnan(values).sum()),
            "finite_min_m": float(valid.min()) if valid.size else None,
            "finite_max_m": float(valid.max()) if valid.size else None}


def _depth_plot(figure, values, row, col, coloraxis, label):
    import plotly.graph_objects as go

    height, width = values.shape
    hover = [[f"{label}<br>row={v}, col={u}<br>"
              + (f"视线距离 {float(value):.5f} m" if math.isfinite(float(value)) else
                 "+inf：至最大量程的线段完整覆盖且无返回" if math.isinf(float(value)) else
                 "NaN：未知覆盖、过近表面或无有效返回")
              for u, value in enumerate(line)] for v, line in enumerate(values)]
    figure.add_trace(go.Heatmap(
        x=list(range(width)), y=list(range(height)),
        z=np.where(np.isfinite(values), values, np.nan), coloraxis=coloraxis,
        text=hover, hovertemplate="%{text}<extra></extra>", hoverongaps=False,
        name=label), row=row, col=col)
    special = np.full(values.shape, np.nan)
    special[np.isposinf(values)] = 1
    special[np.isnan(values)] = 2
    figure.add_trace(go.Heatmap(
        x=list(range(width)), y=list(range(height)), z=special, zmin=1, zmax=2,
        colorscale=[[0, "#cbd5e1"], [0.499, "#cbd5e1"],
                    [0.5, "#e8b7da"], [1, "#e8b7da"]],
        showscale=False, text=hover, hovertemplate="%{text}<extra></extra>",
        hoverongaps=False, name="状态"), row=row, col=col)


def _html(data, config, summary):
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    labels = ("右 ToF", "前 ToF", "左 ToF", "后 ToF")
    figure = make_subplots(
        rows=3, cols=2, row_heights=[0.32, 0.32, 0.36],
        vertical_spacing=0.08, horizontal_spacing=0.12,
        subplot_titles=(*labels, "前向相机 · 视线距离", "前向相机 · 最近表面可见目标 mask"))
    for index, values in enumerate(data["tof_m"]):
        _depth_plot(figure, values, index // 2 + 1, index % 2 + 1,
                    "coloraxis", labels[index])
    _depth_plot(figure, data["camera_range_m"], 3, 1, "coloraxis2", "相机")
    mask = data["target_mask"].astype("u1")
    figure.add_trace(go.Heatmap(
        x=list(range(config.camera.width)), y=list(range(config.camera.height)), z=mask,
        zmin=0, zmax=1, colorscale=[[0, "#f1f5f9"], [1, "#f59e0b"]], showscale=False,
        hovertemplate="row=%{y}, col=%{x}<br>最近表面为目标：%{z}<extra></extra>",
        name="可见目标"), row=3, col=2)
    hint = data["target_hint"]
    bounds = None
    if hint[0] > 0:
        bounds = (round(float(hint[1]) * config.camera.width),
                  round(float(hint[2]) * config.camera.height),
                  round(float(hint[3]) * config.camera.width),
                  round(float(hint[4]) * config.camera.height))
        u0, v0, u1, v1 = bounds
        # Plotly uses integer pixel centers; image-cell edges are half integers.
        figure.add_shape(type="rect", x0=u0 - 0.5, y0=v0 - 0.5,
                         x1=u1 - 0.5, y1=v1 - 0.5,
                         line={"color": "#dc2626", "width": 2}, row=3, col=2)
    for index in range(6):
        row, col = index // 2 + 1, index % 2 + 1
        dimensions = config.tof if index < 4 else config.camera
        figure.update_xaxes(range=[-0.5, dimensions.width - 0.5],
                            title_text="列（向右）", constrain="domain", row=row, col=col)
        figure.update_yaxes(range=[dimensions.height - 0.5, -0.5],
                            title_text="行（向下）", scaleanchor="x" if index == 0 else f"x{index+1}",
                            scaleratio=1, constrain="domain", row=row, col=col)
    figure.update_layout(
        template="plotly_white", height=1280, margin={"l": 55, "r": 105, "t": 45, "b": 45},
        coloraxis={"colorscale": "Viridis", "cmin": config.tof.min_range_m,
                   "cmax": config.tof.max_range_m,
                   "colorbar": {"title": "ToF m", "len": 0.40, "y": 0.78}},
        coloraxis2={"colorscale": "Viridis", "cmin": config.camera.min_range_m,
                    "cmax": config.camera.max_range_m,
                    "colorbar": {"title": "相机 m", "len": 0.37, "y": 0.21}},
        showlegend=False)
    plot = pio.to_html(figure, include_plotlyjs=True, full_html=False,
                       div_id="sensor-observations", config={"responsive": True,
                       "displaylogo": False, "modeBarButtonsToRemove": ["sendChartToCloud"]})
    detection = (f"已检出；半开像素框 [{bounds[0]}, {bounds[2]}) × [{bounds[1]}, {bounds[3]})"
                 if bounds else "未检出；target_hint 为五个零")
    escaped_summary = html.escape(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    title = "同一场景 · 四路 ToF 与前向目标观测"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{margin:0;background:#f8fafc;color:#213047;font-family:system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:22px}}h1{{font-size:23px;margin:0 0 12px}}
p{{line-height:1.65;margin:8px 0;font-size:14px}}.note{{padding:12px 16px;background:#fff;
border:1px solid #dce3eb;border-radius:10px}}.key{{display:inline-block;padding:0 7px;border-radius:3px}}
details{{margin:14px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
@media(max-width:700px){{main{{padding:12px}}h1{{font-size:20px}}}}</style></head>
<body><main><h1>{title}</h1><div class="note">
<p>输入：{html.escape(summary['map'])}。位姿保留完整三维姿态；所有距离均为沿像素视线的欧氏距离（米）。
悬停查看行、列、距离与无效原因。四路顺序为右、前、左、后。</p>
<p><span class="key" style="background:#cbd5e1">灰色 +inf</span>：至最大量程的线段完整覆盖且没有有效表面；
<span class="key" style="background:#e8b7da">粉色 NaN</span>：未知覆盖、过近表面或无有效返回。
两者在原始 NPZ 中保持不同值。最大填充值不代表已知空闲。</p>
<p>目标可见比例 {summary['camera']['visible_fraction']:.3f}，无遮挡有效目标像素 {summary['camera']['unoccluded_target_pixels']}，
检测阈值 {config.min_visible_fraction:g}。{detection}。
橙色 mask 来自全场景最近表面测试；红色框仅在通过阈值时包住可见像素。</p>
<p>这是无噪声的理想针孔开发观测。配置中的视场、量程、安装位姿及阈值均为开发假设，
不代表硬件规格；不模拟真实 ToF 分区响应或 RGB 外观。球体使用内接三角网格近似。</p></div>
{plot}<details><summary>输入、GPU 后端、统计与同场景多位姿计时</summary><pre>{escaped_summary}</pre></details>
<p>本页已内嵌 Plotly 与图像数据，可单文件离线打开。原始数据见同目录 observations.npz；统计见 summary.json。</p>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path, help="已有 OBJ 背景地图")
    parser.add_argument("--config", type=Path, default=Path("configs/devtools.json"))
    parser.add_argument("--vehicle-position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--vehicle-rpy-deg", type=float, nargs=3, default=(0, 0, 0),
                        metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensors/one-frame"), help="必须不存在的新输出目录")
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in args.vehicle_rpy_deg):
        parser.error("--vehicle-rpy-deg must contain finite angles")
    source, configuration, output = args.map.resolve(), args.config.resolve(), args.output.resolve()
    if output.exists() or output in (source, configuration):
        parser.error("--output must be a new directory; inputs and existing results cannot be overwritten")
    config = load_config(configuration)
    if config.sensors is None:
        parser.error("--config must include explicit sensors configuration")
    sensor_config = config.sensors
    body_pose = Pose(tuple(args.vehicle_position), tuple(quaternion_from_rpy_degrees(args.vehicle_rpy_deg).tolist()))
    scene = Scene.from_obj(source, config, body_pose, tuple(args.target_position), t_ref=0.0)
    # One scene upload; the public full-pose API also accepts alternate sensor
    # poses without changing scene facts or advancing scene time.
    yawed = list(args.vehicle_rpy_deg)
    yawed[2] += 15
    body_samples = (body_pose,
                    Pose(body_pose.body_to_world((0.25, 0, 0)), body_pose.quaternion_xyzw),
                    Pose(body_pose.position_m, tuple(quaternion_from_rpy_degrees(yawed).tolist())))
    front_mount = sensor_config.tof_mounts[TOF_ORDER.index("front")]
    sensor_poses = tuple(mounted_pose(pose, front_mount) for pose in body_samples)
    begin = time.perf_counter()
    with BatchRenderer(scene.get_geometry(exclude_ids={"vehicle"})) as renderer:
        body_tensor = poses_to_tensor([body_pose], device=renderer.device)
        sensor_tensor = poses_to_tensor(sensor_poses, device=renderer.device)
        torch.cuda.synchronize(renderer.device)
        prepared = time.perf_counter()
        rig = sample_rig(renderer, sensor_config, body_tensor)
        torch.cuda.synchronize(renderer.device)
        sampled = time.perf_counter()
        many = sample_depth(renderer, sensor_config.tof, sensor_tensor)
        torch.cuda.synchronize(renderer.device)
        finished = time.perf_counter()
        backend = renderer.geometry_info
        triangle_count, uploads = renderer.triangle_count, backend["geometry_uploads"]
    # Explicit export boundary: normal sensor outputs and postprocessing remain
    # on CUDA. Only the offline NPZ/HTML example downloads these arrays.
    data = {
        "tof_m": rig.tof_m[0].cpu().numpy(),
        "target_hint": rig.target_hint[0].cpu().numpy(),
        "camera_range_m": rig.camera.depth.range_m[0].cpu().numpy(),
        "target_mask": rig.camera.target_mask[0].cpu().numpy(),
        "camera_object_index": rig.camera.depth.object_index[0].cpu().numpy(),
        "object_ids": np.asarray(rig.camera.depth.object_ids),
        "t_ref": np.asarray(scene.t_ref),
        "scene_version": np.asarray(rig.scene_version),
        "sample_many_range_m": many.range_m.cpu().numpy(),
        "sample_many_positions_m": np.asarray([pose.position_m for pose in sensor_poses]),
        "sample_many_quaternions_xyzw": np.asarray([pose.quaternion_xyzw for pose in sensor_poses]),
    }
    summary = {
        "map": str(source), "config": str(configuration), "t_ref": scene.t_ref,
        "assumptions": "Noiseless ideal pinhole development configuration; not hardware specifications.",
        "vehicle_pose": asdict(body_pose),
        "target_position_m": list(scene.get_geometry().get_object("target").pose.position_m),
        "sensor_config": asdict(sensor_config), "renderer": backend,
        "triangles": triangle_count, "geometry_uploads": uploads,
        "map_bounds_m": scene.get_geometry().map_bounds_m,
        "tof_order": list(TOF_ORDER),
        "tof": {name: _statistics(values) for name, values in zip(TOF_ORDER, data["tof_m"])},
        "camera": {**_statistics(data["camera_range_m"]),
                   "visible_target_pixels": int(np.count_nonzero(data["target_mask"])),
                   "unoccluded_target_pixels": int(rig.camera.unoccluded_pixels[0].cpu()),
                   "visible_fraction": float(rig.camera.visible_fraction[0].cpu()),
                   "target_hint": data["target_hint"].tolist()},
        "sample_many": {"view": "front ToF", "mode": "one instanced draw; CUDA output and batched postprocessing",
                        "body_pose_changes": ["reference", "body +X 0.25 m", "world yaw +15 degrees"],
                        "sensor_poses": [asdict(pose) for pose in sensor_poses],
                        "frames": [_statistics(frame) for frame in data["sample_many_range_m"]]},
        "timing_seconds": {"prepare_scene_gpu": prepared - begin,
                           "first_rig_sample": sampled - prepared,
                           "sample_many_three_poses": finished - sampled,
                           "sample_many_mean_per_pose": (finished - sampled) / len(sensor_poses)},
    }
    # Serialize before creating output, so invalid summary floats never leave a
    # misleading partially written result. A fresh directory protects all inputs.
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    document = _html(data, sensor_config, summary)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "observations.npz", **data)
    (output / "summary.json").write_text(serialized, encoding="utf-8")
    (output / "one_frame.html").write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(output), "html": str(output / "one_frame.html"),
                      "tof_shape": list(data["tof_m"].shape),
                      "camera_shape": list(data["camera_range_m"].shape),
                      "target_hint": data["target_hint"].tolist(), "renderer": backend},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
