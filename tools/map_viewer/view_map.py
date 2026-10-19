"""View an OBJ mesh, optionally with physical spheres and full FLU attitude axes."""

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def sphere_mesh(center, radius, latitude_steps=16, longitude_steps=32):
    """Closed display mesh of a physical sphere, including both poles."""
    vertices = [(center[0], center[1], center[2] + radius)]
    for row in range(1, latitude_steps):
        theta = math.pi * row / latitude_steps
        for column in range(longitude_steps):
            phi = 2 * math.pi * column / longitude_steps
            vertices.append((center[0] + radius * math.sin(theta) * math.cos(phi),
                             center[1] + radius * math.sin(theta) * math.sin(phi),
                             center[2] + radius * math.cos(theta)))
    south = len(vertices)
    vertices.append((center[0], center[1], center[2] - radius))
    triangles = []
    for column in range(longitude_steps):
        next_column = (column + 1) % longitude_steps
        triangles.append((0, 1 + column, 1 + next_column))
        for row in range(latitude_steps - 2):
            a = 1 + row * longitude_steps + column
            b = 1 + row * longitude_steps + next_column
            c, d = a + longitude_steps, b + longitude_steps
            triangles.extend(((a, c, b), (b, c, d)))
        last = 1 + (latitude_steps - 2) * longitude_steps
        triangles.append((last + column, south, last + next_column))
    return vertices, triangles


def add_scene_geometry(figure, ranges, scene, padding):
    """Overlay physical spheres and full attitude axes without modifying the map."""
    import plotly.graph_objects as go

    scene_info = {}
    for name, sphere, color in (("vehicle", scene.vehicle, "#2679dd"),
                                ("target", scene.target, "#ed8b23")):
        center = sphere.pose.position_m
        vertices, triangles = sphere_mesh(center, sphere.radius_m)
        figure.add_trace(go.Mesh3d(
            x=[p[0] for p in vertices], y=[p[1] for p in vertices],
            z=[p[2] for p in vertices], i=[t[0] for t in triangles],
            j=[t[1] for t in triangles], k=[t[2] for t in triangles],
            name="飞机" if name == "vehicle" else "目标", color=color, opacity=0.9,
            hovertemplate=("飞机球" if name == "vehicle" else "目标球") +
                f" · 半径 {sphere.radius_m:g} m<extra></extra>"))
        scene_info[name] = {"position_m": center, "radius_m": sphere.radius_m,
                            "quaternion_xyzw": sphere.pose.quaternion_xyzw}
        for axis in range(3):
            ranges[axis][0] = min(ranges[axis][0], center[axis] - sphere.radius_m - padding)
            ranges[axis][1] = max(ranges[axis][1], center[axis] + sphere.radius_m + padding)
    pose = scene.vehicle.pose
    axis_length = scene.vehicle.radius_m * 2.5
    for axis, label, color in ((0, "X 前", "#e04444"), (1, "Y 左", "#2aab56"),
                               (2, "Z 上", "#2465d4")):
        body_endpoint = tuple(axis_length if i == axis else 0.0 for i in range(3))
        endpoint = pose.body_to_world(body_endpoint)
        start = pose.position_m
        figure.add_trace(go.Scatter3d(
            x=[start[0], endpoint[0]], y=[start[1], endpoint[1]], z=[start[2], endpoint[2]],
            mode="lines+text", text=["", label], textposition="top center",
            line={"color": color, "width": 6}, name=label, hoverinfo="skip"))
        for i in range(3):
            ranges[i][0] = min(ranges[i][0], endpoint[i] - padding)
            ranges[i][1] = max(ranges[i][1], endpoint[i] + padding)
    note = (f"<p>蓝色飞机球：中心 {scene.vehicle.pose.position_m} m，半径 {scene.vehicle.radius_m:g} m；"
            f"橙色目标球：中心 {scene.target.pose.position_m} m，半径 {scene.target.radius_m:g} m。"
            "红 / 绿 / 蓝线表示机体前 / 左 / 上轴，保留实际三维姿态。</p>")
    return scene_info, note


def mesh_edges(mesh):
    """Return boundary/crease edges; hide triangulation diagonals on flat faces."""
    normals = {}
    for face in mesh.triangles:
        a, b, c = (mesh.vertices_m[index] for index in face)
        u, v = (tuple(point[i] - a[i] for i in range(3)) for point in (b, c))
        normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                  u[0] * v[1] - u[1] * v[0])
        length = math.hypot(*normal)
        if length == 0:
            raise ValueError("mesh 含退化三角形")
        normal = tuple(value / length for value in normal)
        for index in range(3):
            edge = tuple(sorted((face[index], face[(index + 1) % 3])))
            normals.setdefault(edge, []).append(normal)
    return [edge for edge, directions in normals.items() if len(directions) != 2 or
            sum(a * b for a, b in zip(*directions)) < 1 - 1e-10]


def build_mesh_html(data, scene=None):
    """Render every original OBJ face at its exact metric coordinates."""
    import plotly.graph_objects as go
    import plotly.io as pio

    figure = go.Figure()
    for kind, name, color in (("obstacle", "障碍物", "#91a9c5"),
                               ("ground", "地面", "#d8dfe7")):
        vertices, triangles, names = [], [], []
        for mesh in data.objects:
            if mesh.kind != kind:
                continue
            offset = len(vertices)
            vertices.extend(mesh.vertices_m)
            triangles.extend(tuple(offset + index for index in face) for face in mesh.triangles)
            names.extend([html.escape(mesh.name)] * len(mesh.vertices_m))
        figure.add_trace(go.Mesh3d(
            x=[p[0] for p in vertices], y=[p[1] for p in vertices], z=[p[2] for p in vertices],
            i=[t[0] for t in triangles], j=[t[1] for t in triangles], k=[t[2] for t in triangles],
            customdata=names, color=color, name=name, flatshading=True,
            lighting={"ambient": 0.6, "diffuse": 0.8, "specular": 0.08, "roughness": 0.9},
            lightposition={"x": -100, "y": -200, "z": 300},
            hovertemplate="%{customdata}<br>X %{x:.4f} m<br>Y %{y:.4f} m<br>Z %{z:.4f} m<extra></extra>"))
    edge_points = []
    for mesh in data.objects:
        if mesh.kind == "obstacle":
            for start, end in mesh_edges(mesh):
                edge_points.extend((mesh.vertices_m[start], mesh.vertices_m[end], (None, None, None)))
    figure.add_trace(go.Scatter3d(
        x=[p[0] for p in edge_points], y=[p[1] for p in edge_points], z=[p[2] for p in edge_points],
        mode="lines", name="障碍物边线", line={"color": "#38516e", "width": 2}, hoverinfo="skip"))
    padding = max(max(hi - lo for lo, hi in data.bounds_m) * 0.025, 0.1)
    ranges = [[lo - padding, hi + padding] for lo, hi in data.bounds_m]
    scene_info, scene_note = None, ""
    if scene is not None:
        if (scene.background.source.resolve() != data.source.resolve() or
                scene.background.vertex_count != data.vertex_count or
                scene.background.triangle_count != data.triangle_count):
            raise ValueError("场景背景与显示地图必须是同一份完整 OBJ")
        scene_info, scene_note = add_scene_geometry(figure, ranges, scene, padding)
        scene_info.update(background_vertices=data.vertex_count, background_triangles=data.triangle_count)
    camera = {"eye": {"x": 1.75, "y": -2, "z": 1.625}, "up": {"x": 0, "y": 0, "z": 1}}
    figure.update_layout(
        template="plotly_white", margin={"l": 0, "r": 0, "t": 5, "b": 0}, showlegend=False,
        uirevision="mesh-camera", scene={"aspectmode": "data", "dragmode": "orbit",
            "xaxis": {"title": "X (m)", "range": ranges[0]},
            "yaxis": {"title": "Y (m)", "range": ranges[1]},
            "zaxis": {"title": "Z (m)", "range": ranges[2]}, "camera": camera})
    script = """
const chart = document.getElementById('map');
const modeControl = document.getElementById('mesh-mode');
const groundControl = document.getElementById('ground-visible');
function updateBackground() {
  const mode = modeControl.value;
  Plotly.restyle(chart, {visible: [mode !== 'wire', groundControl.checked, mode !== 'solid']}, [0, 1, 2]);
}
modeControl.addEventListener('change', updateBackground);
groundControl.addEventListener('change', updateBackground);
document.getElementById('reset-camera').addEventListener('click', () => {
  Plotly.relayout(chart, {'scene.camera': CAMERA});
});
""".replace("CAMERA", json.dumps(camera))
    plot = pio.to_html(figure, include_plotlyjs=True, full_html=False, div_id="map",
                       default_width="100%", default_height="100%", post_script=script,
                       config={"responsive": True, "displaylogo": False, "scrollZoom": True,
                               "modeBarButtonsToRemove": ["sendChartToCloud"]})
    obstacles = sum(mesh.kind == "obstacle" for mesh in data.objects)
    summary = {"source": data.source.name, "format": "obj", "bounds_m": data.bounds_m,
               "size_xyz_m": data.size_xyz_m, "object_count": len(data.objects),
               "obstacle_count": obstacles, "input_vertices": data.vertex_count,
               "display_vertices": data.vertex_count, "input_triangles": data.triangle_count,
               "display_triangles": data.triangle_count,
               "objects": [{"name": mesh.name, "kind": mesh.kind,
                            "vertices": len(mesh.vertices_m), "triangles": len(mesh.triangles)}
                           for mesh in data.objects]}
    if scene_info is not None:
        summary["scene"] = scene_info
    safe_summary = json.dumps(summary, ensure_ascii=True).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    comments = html.escape("\n".join(data.comments))
    title = ("场景查看" if scene is not None else "地图查看") + " · " + html.escape(data.source.name)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;color:#243247;background:#f7f9fc;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{padding:16px 24px 10px;background:white;border-bottom:1px solid #dce3ec;flex:none;max-height:50vh;overflow:auto}}
h1{{font-size:20px;margin:0 0 8px}}p{{font-size:13px;margin:7px 0;line-height:1.5}}
.controls{{display:flex;align-items:center;gap:22px;flex-wrap:wrap;margin:12px 0}}
label{{font-size:14px;display:flex;gap:8px;align-items:center}}select,button{{padding:5px 8px}}
#plot{{flex:1;min-height:0;overflow:hidden}}#plot>div{{height:100%}}details{{font-size:12px}}
pre{{white-space:pre-wrap;margin:6px 0}}@media(max-width:700px){{header{{padding:12px}}}}</style>
</head><body><header><h1>{title}</h1>{scene_note}
<p id="summary">{obstacles:,} 个障碍物 · 独立地面 · {data.vertex_count:,} 顶点 · {data.triangle_count:,} 三角面</p>
<div class="controls"><label>障碍物 <select id="mesh-mode"><option value="edges">实心 + 边线</option><option value="solid">实心</option><option value="wire">线框</option></select></label>
<label><input id="ground-visible" type="checkbox" checked>显示地面</label><button id="reset-camera" type="button">重置视角</button></div>
<p>拖动旋转，滚轮缩放。XYZ 单位为米，三轴等比例；完整显示原始 mesh，悬停查看障碍名称和坐标。</p>
<details><summary>地图元数据</summary><pre>{comments}</pre></details>
</header><div id="plot">{plot}</div><script id="viewer-summary" type="application/json">{safe_summary}</script></body></html>"""


def serve_html(path, host="127.0.0.1", port=8080):
    """Serve this HTML alone: no directory listing, workspace paths, or symlinks."""
    path = Path(path)
    content = path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if unquote(urlsplit(self.path).path) not in {"/", "/" + path.name}:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

    with ThreadingHTTPServer((host, port), Handler) as server:
        if host == "0.0.0.0":
            print(f"容器内监听 {host}:{port}；宿主访问链接由容器入口打印", flush=True)
        else:
            print(f"浏览器打开 http://{host}:{port}/{quote(path.name)} （Ctrl+C 结束）", flush=True)
        server.serve_forever()


def validate_args(parser, args):
    """Resolve the requested mode and reject incomplete scenes before any output."""
    if args.input.suffix.lower() != ".obj":
        parser.error("输入须为 .obj mesh 地图")
    scene_requested = any(value is not None for value in (
        args.vehicle_position, args.target_position, args.vehicle_rpy_deg, args.config))
    if scene_requested and (args.vehicle_position is None or args.target_position is None):
        parser.error("场景查看须同时提供 --vehicle-position X Y Z 和 --target-position X Y Z；"
                     "姿态或配置参数也不能省略这两个位置")
    args.config = args.config or Path("configs/devtools.json")
    args.vehicle_rpy_deg = args.vehicle_rpy_deg or (0.0, 0.0, 0.0)
    output = args.output or Path("outputs/map_viewer") / (args.input.stem + ".html")
    protected = (args.input, args.config)
    if output.suffix.lower() != ".html" or any(
            output.resolve() == path.resolve() or
            (output.exists() and path.exists() and output.samefile(path)) for path in protected):
        parser.error("输出必须是独立的 .html 文件，不能覆盖输入地图或配置")
    if not 1 <= args.port <= 65535:
        parser.error("port 须在 1..65535 之间")
    return scene_requested, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OBJ mesh 地图，坐标按米解释")
    parser.add_argument("--output", type=Path, help="HTML 路径；默认 outputs/map_viewer/<输入名>.html")
    parser.add_argument("--vehicle-position", type=float, nargs=3,
                        metavar=("X", "Y", "Z"), help="飞机世界位置，米；须与目标位置成对提供")
    parser.add_argument("--target-position", type=float, nargs=3,
                        metavar=("X", "Y", "Z"), help="目标世界位置，米；须与飞机位置成对提供")
    parser.add_argument("--vehicle-rpy-deg", type=float, nargs=3,
                        metavar=("ROLL", "PITCH", "YAW"), help="机体姿态，度；Rz Ry Rx，默认 0 0 0")
    parser.add_argument("--config", type=Path, help="场景统一配置；默认 configs/devtools.json")
    parser.add_argument("--serve", action="store_true", help="生成后持续提供仅此 HTML 的 HTTP 服务")
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    scene_requested, output = validate_args(parser, args)
    try:
        scene = None
        if scene_requested:
            from tof_tracker.geometry import quaternion_from_rpy_degrees
            from devtools.config import load_config
            from devtools.scene import Pose, Scene

            config = load_config(args.config)
            quaternion = tuple(float(value) for value in quaternion_from_rpy_degrees(args.vehicle_rpy_deg))
            pose = Pose(args.vehicle_position, quaternion)
            scene = Scene.from_obj(args.input, config, pose, args.target_position)
        from devtools.mesh import MeshMap

        data = scene.background if scene is not None else MeshMap.from_obj(args.input)
        page = build_mesh_html(data, scene=scene)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(page, encoding="utf-8")
        if scene is not None:
            background = {"background_vertices": data.vertex_count, "background_triangles": data.triangle_count}
            print(json.dumps({**background,
                              "vehicle": {"position_m": scene.vehicle.pose.position_m,
                                          "radius_m": scene.vehicle.radius_m,
                                          "quaternion_xyzw": scene.vehicle.pose.quaternion_xyzw},
                              "target": {"position_m": scene.target.pose.position_m,
                                         "radius_m": scene.target.radius_m}}, ensure_ascii=False), flush=True)
        print(f"已生成 {output}：{len(data.objects)} 个独立 mesh，"
              f"{data.vertex_count} 顶点，{data.triangle_count} 三角面（完整显示）", flush=True)
        if args.serve:
            serve_html(output, args.host, args.port)
    except KeyboardInterrupt:
        pass
    except (OSError, UnicodeError, ValueError, ImportError) as error:
        mode = "场景" if scene_requested else "地图"
        parser.exit(1, f"{mode}查看失败：{error}\n")


if __name__ == "__main__":
    main()
