"""View an ASCII XYZ PLY, optionally with two physical spheres and body FLU axes."""

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


@dataclass(frozen=True)
class MapData:
    source: Path
    count: int
    comments: tuple[str, ...]
    resolution: float | None
    voxel_size: float
    origin: tuple[float, float, float]
    bounds: tuple[tuple[float, float], ...]
    points: tuple[tuple[float, float, float], ...]
    cells: tuple[tuple[int, int, int], ...]
    voxels_limited: bool


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("数值必须为有限正数")
    return number


def voxel_index(point, size, origin=(0.0, 0.0, 0.0)):
    """Cell centers are origin + index*size; cells are half-open around centers."""
    scaled = [(v - base) / size + 0.5 for v, base in zip(point, origin)]
    if any(not math.isfinite(v) or abs(v) >= 2**52 for v in scaled):
        raise ValueError("坐标相对体素尺寸过大，无法可靠表示格子")
    return tuple(math.floor(v) for v in scaled)


def read_map(path, voxel_size=None, max_points=100_000, max_voxels=12_000):
    """Validate every input point; keep bounded display data without changing the PLY.

    Point selection is uniform by input index. When occupied cells exceed the
    limit, keep cells with the smallest stable hash, independent of input order.
    Only selected cells are retained, so large input does not require a full set.
    """
    if max_points < 1 or max_voxels < 1:
        raise ValueError("显示点数与体素数上限必须为正整数")
    path = Path(path)
    with path.open(encoding="ascii") as stream:
        if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
            raise ValueError("仅支持 format ascii 1.0 的 PLY")
        count, properties, comments = None, [], []
        for line_number in range(3, 259):
            line = stream.readline()
            if not line or len(line) > 16_384:
                raise ValueError("PLY 头缺少 end_header 或行过长")
            fields = line.strip().split()
            if not fields:
                raise ValueError("PLY 头存在空行")
            if fields == ["end_header"]:
                break
            if fields[0] == "comment":
                comments.append(line.strip()[8:])
            elif fields[:2] == ["element", "vertex"] and len(fields) == 3 and count is None:
                count = int(fields[2])
            elif fields[0] == "property" and count is not None and len(fields) == 3:
                if fields[1] not in {"float", "double", "float32", "float64"}:
                    raise ValueError("XYZ 必须使用 float 或 double 标量属性")
                properties.append(fields[2])
            else:
                raise ValueError(f"不支持的 PLY 头字段：第 {line_number} 行")
        else:
            raise ValueError("PLY 头过长或缺少 end_header")
        if count is None or count <= 0 or sorted(properties) != ["x", "y", "z"]:
            raise ValueError("PLY 必须包含非空 vertex 及且仅有 x/y/z 属性")

        resolution, sizes = None, None
        for comment in comments:
            fields = comment.split()
            if fields[:1] == ["unit"] and fields != ["unit", "metre"]:
                raise ValueError("PLY 单位必须为 metre；此工具不转换单位")
            if fields[:1] == ["size_xyz"]:
                if len(fields) != 6 or fields[4] != "resolution" or sizes is not None:
                    raise ValueError("size_xyz / resolution 元数据格式错误或重复")
                sizes = tuple(positive_number(v) for v in fields[1:4])
                resolution = positive_number(fields[5])
        size = positive_number(voxel_size) if voxel_size is not None else resolution
        if size is None:
            raise ValueError("PLY 没有 resolution 元数据；请提供 --voxel-size（米）")
        origin = (-sizes[0] / 2, -sizes[1] / 2, 0.0) if sizes else (0.0, 0.0, 0.0)
        order = [properties.index(axis) for axis in "xyz"]
        sample_count = min(count, max_points)
        indices = iter([count // 2] if sample_count == 1 else
                       (i * (count - 1) // (sample_count - 1) for i in range(sample_count)))
        next_index = next(indices)
        points, cells, ranked_cells = [], set(), []
        bounds = [[math.inf, -math.inf] for _ in range(3)]
        limited = False
        for index in range(count):
            fields = stream.readline().split()
            if len(fields) != 3:
                raise ValueError(f"顶点 {index + 1} 缺失或不是三个数值")
            values = tuple(float(v) for v in fields)
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"顶点 {index + 1} 包含非有限坐标")
            point = tuple(values[k] for k in order)
            for axis, value in enumerate(point):
                bounds[axis][0] = min(bounds[axis][0], value)
                bounds[axis][1] = max(bounds[axis][1], value)
            if index == next_index:
                points.append(point)
                next_index = next(indices, None)
            cell = voxel_index(point, size, origin)
            if cell in cells:
                continue
            rank = int.from_bytes(hashlib.blake2b(str(cell).encode("ascii"), digest_size=8).digest(), "big")
            entry = (-rank, *(-v for v in cell))
            if len(cells) < max_voxels:
                cells.add(cell)
                heapq.heappush(ranked_cells, entry)
            else:
                limited = True
                if entry > ranked_cells[0]:
                    removed = heapq.heapreplace(ranked_cells, entry)
                    cells.remove(tuple(-v for v in removed[1:]))
                    cells.add(cell)
        if any(line.strip() for line in stream):
            raise ValueError("顶点数据超出 PLY 声明数量")
    return MapData(path, count, tuple(comments), resolution, size, origin,
                   tuple(tuple(v) for v in bounds), tuple(points), tuple(sorted(cells)), limited)


# Eight corner signs and twelve outward triangles of one closed cube.
CORNERS = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
           (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
TRIANGLES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))


def cube_mesh(centers, size, fill):
    """Closed cubes of edge size*fill; no surface interpolation between cells."""
    if not math.isfinite(fill) or not 0.1 <= fill <= 1:
        raise ValueError("体素填充比例须在 0.1..1 之间")
    half = size * fill / 2
    vertices = [tuple(c[k] + sign[k] * half for k in range(3))
                for c in centers for sign in CORNERS]
    triangles = [tuple(8 * index + v for v in triangle)
                 for index in range(len(centers)) for triangle in TRIANGLES]
    return vertices, triangles


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


def build_html(data, point_size=3.0, voxel_fill=0.9, scene=None):
    import plotly.graph_objects as go
    import plotly.io as pio

    if not math.isfinite(point_size) or not 1 <= point_size <= 12:
        raise ValueError("点像素尺寸须在 1..12 之间")
    centers = [tuple(base + index * data.voxel_size for base, index in zip(data.origin, cell))
               for cell in data.cells]
    vertices, triangles = cube_mesh(centers, data.voxel_size, voxel_fill)
    figure = go.Figure([
        go.Scatter3d(x=[p[0] for p in data.points], y=[p[1] for p in data.points],
                     z=[p[2] for p in data.points], mode="markers", name="原始点云", visible=False,
                     marker={"size": point_size, "color": [p[2] for p in data.points],
                             "coloraxis": "coloraxis", "opacity": 1},
                     hovertemplate="原始点 (m)<br>X %{x:.4f}<br>Y %{y:.4f}<br>Z %{z:.4f}<extra></extra>"),
        go.Mesh3d(x=[v[0] for v in vertices], y=[v[1] for v in vertices],
                  z=[v[2] for v in vertices], i=[t[0] for t in triangles],
                  j=[t[1] for t in triangles], k=[t[2] for t in triangles],
                  intensity=[c[2] for c in centers for _ in CORNERS], coloraxis="coloraxis",
                  flatshading=True, name="显示占用格", lighting={"ambient": 0.65, "diffuse": 0.7},
                  hovertemplate="显示体素表面 (m)<br>X %{x:.4f}<br>Y %{y:.4f}<br>Z %{z:.4f}<extra></extra>")])
    ranges = [[lo - data.voxel_size, hi + data.voxel_size] for lo, hi in data.bounds]
    scene_info = None
    scene_note = ""
    if scene is not None:
        if scene.background.source.resolve() != data.source.resolve() or scene.background.count != data.count:
            raise ValueError("场景背景与显示地图必须是同一份完整 PLY")
        scene_info = {"background_points": scene.background.count}
        for name, sphere, color in (("vehicle", scene.vehicle, "#2679dd"),
                                     ("target", scene.target, "#ed8b23")):
            center = sphere.pose.position_m
            sphere_vertices, sphere_triangles = sphere_mesh(center, sphere.radius_m)
            figure.add_trace(go.Mesh3d(
                x=[p[0] for p in sphere_vertices], y=[p[1] for p in sphere_vertices],
                z=[p[2] for p in sphere_vertices], i=[t[0] for t in sphere_triangles],
                j=[t[1] for t in sphere_triangles], k=[t[2] for t in sphere_triangles],
                name="飞机" if name == "vehicle" else "目标", color=color, opacity=0.9,
                hovertemplate=("飞机球" if name == "vehicle" else "目标球") +
                    f" · 半径 {sphere.radius_m:g} m<extra></extra>"))
            scene_info[name] = {"position_m": center, "radius_m": sphere.radius_m,
                                "quaternion_xyzw": sphere.pose.quaternion_xyzw}
            for axis in range(3):
                ranges[axis][0] = min(ranges[axis][0], center[axis] - sphere.radius_m - data.voxel_size)
                ranges[axis][1] = max(ranges[axis][1], center[axis] + sphere.radius_m + data.voxel_size)
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
                ranges[i][0] = min(ranges[i][0], endpoint[i] - data.voxel_size)
                ranges[i][1] = max(ranges[i][1], endpoint[i] + data.voxel_size)
        scene_note = (f"<p>蓝色飞机球：中心 {scene.vehicle.pose.position_m} m，半径 {scene.vehicle.radius_m:g} m；"
                      f"橙色目标球：中心 {scene.target.pose.position_m} m，半径 {scene.target.radius_m:g} m。"
                      "红 / 绿 / 蓝线表示机体前 / 左 / 上轴，保留实际三维姿态。</p>")
    zlo, zhi = data.bounds[2]
    figure.update_layout(
        template="plotly_white", margin={"l": 0, "r": 15, "t": 5, "b": 0}, showlegend=False,
        uirevision="map-camera", scene={"aspectmode": "data", "dragmode": "orbit",
            "xaxis": {"title": "X (m)", "range": ranges[0]},
            "yaxis": {"title": "Y (m)", "range": ranges[1]},
            "zaxis": {"title": "Z (m)", "range": ranges[2]},
            "camera": {"eye": {"x": 1.5, "y": -1.7, "z": 1.2}}},
        coloraxis={"colorscale": "Viridis", "cmin": zlo, "cmax": max(zhi, zlo + data.voxel_size),
                   "colorbar": {"title": "Z (m)", "thickness": 14}})
    # Only fill needs custom geometry updates; camera and WebGL are Plotly's.
    script = """
const chart = document.getElementById('map');
const centers = CENTERS;
const signs = SIGNS;
document.getElementById('mode').addEventListener('change', event => {
  const mode = event.target.value;
  Plotly.restyle(chart, {visible: [mode !== 'voxels', mode !== 'points']}, [0, 1]);
});
document.getElementById('point-size').addEventListener('input', event => {
  document.getElementById('point-value').textContent = event.target.value + ' px';
  Plotly.restyle(chart, {'marker.size': Number(event.target.value)}, [0]);
});
const fillControl = document.getElementById('voxel-fill');
fillControl.addEventListener('input', event => {
  document.getElementById('fill-value').textContent = event.target.value;
});
fillControl.addEventListener('change', event => {
  const half = VOXEL_SIZE * Number(event.target.value) / 2;
  const x = [], y = [], z = [];
  for (const center of centers) for (const sign of signs) {
    x.push(center[0] + sign[0] * half);
    y.push(center[1] + sign[1] * half);
    z.push(center[2] + sign[2] * half);
  }
  Plotly.restyle(chart, {x: [x], y: [y], z: [z]}, [1]);
});
""".replace("CENTERS", json.dumps(centers)).replace("SIGNS", json.dumps(CORNERS)).replace(
        "VOXEL_SIZE", json.dumps(data.voxel_size))
    plot = pio.to_html(figure, include_plotlyjs=True, full_html=False, div_id="map",
                       default_width="100%", default_height="100%", post_script=script,
                       config={"responsive": True, "displaylogo": False, "scrollZoom": True,
                               "modeBarButtonsToRemove": ["sendChartToCloud"]})
    summary = {"source": data.source.name, "input_points": data.count,
               "display_points": len(data.points), "points_limited": len(data.points) < data.count,
               "display_voxels": len(data.cells), "voxels_limited": data.voxels_limited,
               "voxel_size_m": data.voxel_size, "origin_m": data.origin, "bounds_m": data.bounds}
    if scene_info is not None:
        summary["scene"] = scene_info
    point_note = f"点云 {len(data.points):,} / {data.count:,} 点"
    if len(data.points) < data.count:
        point_note += "（按输入序号均匀抽样）"
    voxel_note = f"占用格展示 {len(data.cells):,} 个"
    voxel_note += "（已达显示上限；按稳定哈希抽样，非完整占用）" if data.voxels_limited else "（完整）"
    comments = html.escape("\n".join(data.comments))
    safe_summary = json.dumps(summary, ensure_ascii=True).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    title = ("场景查看" if scene is not None else "地图查看") + " · " + html.escape(data.source.name)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;color:#243247;background:#f7f9fc}}
header{{padding:16px 24px 10px;background:white;border-bottom:1px solid #dce3ec}}
h1{{font-size:20px;margin:0 0 8px}}p{{font-size:13px;margin:7px 0;line-height:1.5}}
.controls{{display:flex;align-items:center;gap:22px;flex-wrap:wrap;margin:12px 0}}
label{{font-size:14px;display:flex;gap:8px;align-items:center}}select{{padding:5px}}
output{{min-width:38px;font-variant-numeric:tabular-nums}}#plot{{height:calc(100vh - 225px);min-height:450px}}
details{{font-size:12px}}pre{{white-space:pre-wrap;margin:6px 0}}@media(max-width:700px){{header{{padding:12px}}#plot{{height:65vh}}}}</style>
</head><body><header><h1>{title}</h1>
{scene_note}
<p id="summary">{point_note} · {voxel_note} · 网格 {data.voxel_size:g} m · 原点 {data.origin} m</p>
<div class="controls"><label>显示 <select id="mode"><option value="voxels">体素</option><option value="points">原始点云</option><option value="both">叠加</option></select></label>
<label>点大小 <input id="point-size" type="range" min="1" max="12" step="0.5" value="{point_size}"><output id="point-value">{point_size:g} px</output></label>
<label>体素填充 <input id="voxel-fill" type="range" min="0.1" max="1" step="0.05" value="{voxel_fill}"><output id="fill-value">{voxel_fill:g}</output></label></div>
<p>拖动旋转，滚轮缩放。XYZ 单位为米，按 Z 高度着色，三轴等比例。体素边长 = 网格 × 填充比例（松开滑块更新）。</p>
<p>体素由全部点重新落格并去重，仅表示点采样触及的格子；不填充障碍内部，也不推断空闲空间。原始 PLY 不变。</p>
<details><summary>PLY 元数据 / 网格约定</summary><pre>{comments}</pre>
<p>格中心 = 原点 + 整数索引 × 网格；各轴区间为 [中心 − 网格/2, 中心 + 网格/2)。有 size_xyz 时原点为 (−size_x/2, −size_y/2, 0)，否则为 (0, 0, 0)。</p></details>
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
        address = "127.0.0.1" if host == "0.0.0.0" else host
        print(f"浏览器打开 http://{address}:{port}/{quote(path.name)} （Ctrl+C 结束）", flush=True)
        server.serve_forever()


def quaternion_from_rpy_degrees(angles):
    """Body FLU to world rotation Rz(yaw) Ry(pitch) Rx(roll), stored xyzw."""
    roll, pitch, yaw = (math.radians(value) / 2 for value in angles)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def validate_args(parser, args):
    """Resolve the requested mode and reject incomplete scenes before any output."""
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
    parser.add_argument("input", type=Path, help="ASCII XYZ PLY，坐标按米解释")
    parser.add_argument("--output", type=Path, help="HTML 路径；默认 outputs/map_viewer/<输入名>.html")
    parser.add_argument("--voxel-size", type=float, help="显示网格边长（米）；默认 PLY resolution")
    parser.add_argument("--point-size", type=float, default=3, help="初始点大小 1..12 px")
    parser.add_argument("--voxel-fill", type=float, default=0.9, help="初始体素边长比例 0.1..1")
    parser.add_argument("--max-points", type=int, default=100_000, help="点显示上限，默认 100000")
    parser.add_argument("--max-voxels", type=int, default=12_000, help="体素显示上限，默认 12000")
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
            from devtools.config import load_config
            from devtools.scene import Pose, Scene

            config = load_config(args.config)
            pose = Pose(args.vehicle_position, quaternion_from_rpy_degrees(args.vehicle_rpy_deg))
            scene = Scene.from_ply(args.input, config, pose, args.target_position)
        data = read_map(args.input, args.voxel_size, args.max_points, args.max_voxels)
        page = build_html(data, args.point_size, args.voxel_fill, scene=scene)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(page, encoding="utf-8")
        if scene is not None:
            print(json.dumps({"background_points": scene.background.count,
                              "vehicle": {"position_m": scene.vehicle.pose.position_m,
                                          "radius_m": scene.vehicle.radius_m,
                                          "quaternion_xyzw": scene.vehicle.pose.quaternion_xyzw},
                              "target": {"position_m": scene.target.pose.position_m,
                                         "radius_m": scene.target.radius_m}}, ensure_ascii=False), flush=True)
        print(f"已生成 {output}：点 {len(data.points)}/{data.count}，占用格 {len(data.cells)}"
              f"{'（已达显示上限）' if data.voxels_limited else ''}，网格 {data.voxel_size:g} m", flush=True)
        if args.serve:
            serve_html(output, args.host, args.port)
    except KeyboardInterrupt:
        pass
    except (OSError, UnicodeError, ValueError, ImportError) as error:
        mode = "场景" if scene_requested else "地图"
        parser.exit(1, f"{mode}查看失败：{error}\n")


if __name__ == "__main__":
    main()
