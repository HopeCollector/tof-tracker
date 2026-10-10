"""Verify map/scene geometry, unified viewing CLI, offline HTML and HTTP scope."""

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import urlopen

from devtools.config import SceneConfig
from devtools.scene import Pose, Scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "map_viewer"))
import view_map


def ply(points, comments=("unit metre", "size_xyz 8 6 3 resolution 0.2"), properties="xyz"):
    header = ["ply", "format ascii 1.0", *("comment " + value for value in comments),
              f"element vertex {len(points)}", *("property float " + axis for axis in properties),
              "end_header"]
    rows = [" ".join(str(point["xyz".index(axis)]) for axis in properties) for point in points]
    return "\n".join(header + rows) + "\n"


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def vector_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def page_summary(page):
    return json.loads(re.search(
        r'<script id="viewer-summary" type="application/json">(.*?)</script>', page).group(1))


def page_figure(page):
    match = re.search(r'Plotly\.newPlot\(\s*"map",\s*', page)
    payload = page[match.end():]
    decoder = json.JSONDecoder()
    traces, end = decoder.raw_decode(payload)
    layout, _ = decoder.raw_decode(payload[end:].lstrip(", \n\t"))
    return traces, layout


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []
        self.controls = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if (tag == "script" and "src" in attrs) or (tag == "link" and "href" in attrs):
            self.external.append(attrs)
        if "id" in attrs:
            self.controls[attrs["id"]] = attrs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    entry = Path(view_map.__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="viewer-check-", dir=args.results.parent) as temp:
        folder = Path(temp)
        source = folder / "input & <map>.ply"
        points = [(-0.101, 0, 0), (-0.1, 0, 0), (0, 0, 0), (0.1, 0, 0),
                  (0.23, -0.37, 0.41), (0.23, -0.37, 0.41)]
        comments = ("unit metre", "</script><script>window.invalid = 1</script>")
        source.write_text(ply(points, comments), encoding="ascii")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        data = view_map.read_map(source, voxel_size=0.2)
        assert data.points == tuple(points), "changed original point coordinates/order"
        assert data.origin == (0, 0, 0)
        assert data.cells == ((-1, 0, 0), (0, 0, 0), (1, -2, 2), (1, 0, 0))
        assert view_map.voxel_index((-0.1000001, -0.3, 0.1), 0.2) == (-1, -1, 1)
        assert view_map.voxel_index((-0.1, 0, 0), 0.2) == (0, 0, 0)
        assert view_map.voxel_index((-0.1, 0, 0), 0.2, (-0.2, 0, 0)) == (1, 0, 0)
        assert len(data.cells) < data.count and not data.voxels_limited
        checks.append("original points, negative half-open bins, shifted origin, duplicate cells")

        reordered = folder / "reordered.ply"
        reordered.write_text(ply(points, properties="zyx"))
        ordered_data = view_map.read_map(reordered)
        assert ordered_data.points == tuple(points) and ordered_data.origin == (-4, -3, 0)
        near(ordered_data.resolution, 0.2)
        assert "size_xyz 8 6 3 resolution 0.2" in ordered_data.comments
        override = view_map.read_map(reordered, voxel_size=0.4)
        near(override.voxel_size, 0.4)
        checks.append("XYZ property order, metadata resolution/anchor, explicit grid override")

        # Independent plane fixtures include float32-style serialization jitter,
        # and odd extents whose lower bound does not lie on an origin-zero grid.
        for sx, sy in ((8, 6), (1.4, 1)):
            nx, ny = round(sx / 0.2), round(sy / 0.2)
            import struct
            f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
            ground = [(f32(-sx / 2 + i * 0.2), f32(-sy / 2 + j * 0.2), 0)
                      for i in range(nx) for j in range(ny)]
            plane = folder / f"ground-{sx}.ply"
            plane.write_text(ply(ground, ("unit metre", f"size_xyz {sx} {sy} 1.2 resolution 0.2")))
            result = view_map.read_map(plane)
            assert set(result.cells) == {(i, j, 0) for i in range(nx) for j in range(ny)}
        checks.append("float32 even/odd ground grids preserved: 1200 and 35 cells")

        centers = [(-1, 0.5, 2), (1.7, -2, 0)]
        for fill in (0.1, 0.9, 1):
            vertices, triangles = view_map.cube_mesh(centers, 0.2, fill)
            assert len(vertices) == 16 and len(triangles) == 24
            for cube, center in enumerate(centers):
                faces = triangles[12 * cube:12 * (cube + 1)]
                corners = vertices[8 * cube:8 * (cube + 1)]
                for axis in range(3):
                    near(min(v[axis] for v in corners), center[axis] - 0.1 * fill)
                    near(max(v[axis] for v in corners), center[axis] + 0.1 * fill)
                edges = Counter(tuple(sorted((face[i], face[(i + 1) % 3])))
                                for face in faces for i in range(3))
                assert len(edges) == 18 and set(edges.values()) == {2}, "cube is not closed"
                volume = 0
                for face in faces:
                    a, b, c = [vertices[index] for index in face]
                    normal = cross(vector_sub(b, a), vector_sub(c, a))
                    face_center = tuple((a[k] + b[k] + c[k]) / 3 for k in range(3))
                    assert dot(normal, vector_sub(face_center, center)) > 0, "face winding inward"
                    volume += dot(a, cross(b, c)) / 6
                near(volume, (0.2 * fill)**3)
        checks.append("closed outward cube triangles, exact bounds/volume, independent fill scaling")

        many = folder / "many.ply"
        samples = [(i * 0.2, 0, 0) for i in range(1000)]
        many.write_text(ply(samples, ("unit metre",)))
        limited = view_map.read_map(many, voxel_size=0.2, max_points=7, max_voxels=9)
        assert len(limited.points) == 7 and len(limited.cells) == 9 and limited.voxels_limited
        assert limited.points == tuple(samples[i * 999 // 6] for i in range(7))
        assert limited.bounds == ((0, 199.8), (0, 0), (0, 0))
        reverse = folder / "reverse.ply"
        reverse.write_text(ply(list(reversed(samples)), ("unit metre",)))
        assert view_map.read_map(reverse, 0.2, 7, 9).cells == limited.cells
        checks.append("bounded display, uniform raw sampling, order-independent unique cell selection")

        page = view_map.build_html(data)
        parsed = PageParser()
        parsed.feed(page)
        assert not parsed.external, "HTML requires external scripts/styles"
        assert {"mode", "point-size", "voxel-fill", "map"} <= parsed.controls.keys()
        assert parsed.controls["point-size"]["max"] == "12"
        assert parsed.controls["voxel-fill"]["max"] == "1"
        assert "window.invalid = 1</script>" not in page and "&lt;/script&gt;" in page
        assert "input &amp; &lt;map&gt;.ply" in page
        summary = page_summary(page)
        assert summary["input_points"] == 6 and summary["display_voxels"] == 4
        traces, layout = page_figure(page)
        assert traces[0]["type"] == "scatter3d" and traces[0]["visible"] is False
        assert traces[0]["x"] == [p[0] for p in points]
        assert traces[1]["type"] == "mesh3d" and len(traces[1]["i"]) == 48
        assert layout["scene"]["aspectmode"] == "data"
        assert layout["scene"]["xaxis"]["title"]["text"] == "X (m)"
        assert "已达显示上限" in view_map.build_html(limited)
        assert '"modeBarButtonsToRemove": ["sendChartToCloud"]' in page
        checks.append("embedded Plotly JS, default true cubes, meter/aspect/color UI, HTML escaping/limits")

        invalid = [
            "not a ply\n", ply([]), ply(points).replace("ascii", "binary_little_endian"),
            ply(points).replace("element vertex 6", "element vertex -1"),
            ply(points).replace("element vertex 6", "element vertex 5"),
            ply(points).replace("element vertex 6", "element vertex 7"),
            ply(points).replace("property float z", "property float x"),
            ply(points).replace("property float z", "property list uchar float z"),
            ply(points).replace("resolution 0.2", "resolution 0"),
            ply(points).replace("unit metre", "unit centimetre"),
            ply(points).replace("end_header\n", ""),
            ply([(float("nan"), 0, 0)]), ply([(0, float("inf"), 0)]),
            ply([(0, 0, float("-inf"))]), ply([(1e300, 0, 0)]),
            ply([(0, 0, 0)]).replace("0 0 0\n", "0 bad 0\n"),
            ply(points).replace("size_xyz 8 6 3 resolution 0.2", "size_xyz 8 6 3 resolution nan"),
        ]
        for index, text in enumerate(invalid):
            bad = folder / f"bad-{index}.ply"
            bad.write_text(text)
            try:
                view_map.read_map(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted malformed fixture {index}")
        for options in ({"max_points": 0}, {"max_voxels": -1}, {"voxel_size": float("nan")}):
            try:
                view_map.read_map(source, **options)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid display arguments {options}")
        try:
            view_map.read_map(source)
        except ValueError:
            pass
        else:
            raise AssertionError("missing resolution did not require explicit voxel size")
        checks.append(f"{len(invalid)} malformed PLY cases, invalid limits/grid, missing resolution refused")

        output = folder / "public" / "map.html"
        command = [sys.executable, str(entry), str(source), "--voxel-size", "0.2", "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and output.is_file(), result.stderr
        map_page = output.read_text()
        assert "scene" not in page_summary(map_page)
        assert len(page_figure(map_page)[0]) == 2, "map-only CLI added scene objects"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        failed = subprocess.run([*command, "--point-size", "0"], capture_output=True, timeout=30)
        assert failed.returncode != 0
        # Bind a free port in the child itself (port 0 is a direct API test), then
        # report it so HTTP scope can be checked without assumptions about 8080.
        server_script = """
import sys
sys.path.insert(0, sys.argv[1])
from http.server import ThreadingHTTPServer
import view_map
real = view_map.ThreadingHTTPServer
def report_server(*args, **kwargs):
    server = real(*args, **kwargs)
    print(server.server_address[1], flush=True)
    return server
view_map.ThreadingHTTPServer = report_server
view_map.serve_html(sys.argv[2], port=0)
"""
        server = subprocess.Popen([sys.executable, "-c", server_script, str(entry.parent), str(output)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            port = int(server.stdout.readline())
            url = f"http://127.0.0.1:{port}"
            assert urlopen(url + "/map.html", timeout=5).read() == output.read_bytes()
            assert urlopen(url + "/", timeout=5).read() == output.read_bytes()
            for path in ("/../input.ply", "/%2e%2e/input.ply", "/AGENTS.md", "/map.html/extra"):
                try:
                    urlopen(url + path, timeout=5)
                except HTTPError as error:
                    assert error.code == 404
                else:
                    raise AssertionError(f"server exposed unexpected path {path}")
        finally:
            server.terminate()
            server.communicate(timeout=5)
        checks.append("CLI generation/source unchanged, loopback server serves only generated HTML")

        center, radius = (1.3, -2.1, 0.7), 0.4
        vertices, triangles = view_map.sphere_mesh(center, radius)
        for point in vertices:
            near(math.dist(point, center), radius)
        for axis in range(3):
            near(min(p[axis] for p in vertices), center[axis] - radius)
            near(max(p[axis] for p in vertices), center[axis] + radius)
        edges = Counter()
        volume = 0
        for triangle in triangles:
            a, b, c = (vertices[i] for i in triangle)
            normal = cross(vector_sub(b, a), vector_sub(c, a))
            centroid = tuple((a[i] + b[i] + c[i]) / 3 for i in range(3))
            assert dot(normal, vector_sub(centroid, center)) > 0, "sphere winding inward"
            volume += dot(a, cross(b, c)) / 6
            for i in range(3):
                edges[tuple(sorted((triangle[i], triangle[(i + 1) % 3])))] += 1
        assert set(edges.values()) == {2}, "sphere mesh must be closed"
        exact_volume = 4 * math.pi * radius ** 3 / 3
        assert 0.97 * exact_volume < volume < exact_volume
        checks.append("closed outward sphere geometry, physical radius/bounds and independent volume")

        background = folder / "background.ply"
        background.write_text(ply([(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 2)],
                                  ("unit metre", "size_xyz 6 6 3 resolution 0.25")))
        center = (0.2, -0.3, 1.0)
        config = SceneConfig(vehicle_radius_m=0.25, target_radius_m=0.5)
        scene = Scene.from_ply(background, config,
                               Pose(center, (0, math.sqrt(0.5), 0, math.sqrt(0.5))), (2, 1, 1.5))
        page = view_map.build_html(view_map.read_map(background), scene=scene)
        traces, layout = page_figure(page)
        assert len(traces) == 7
        for trace, sphere in zip(traces[2:4], (scene.vehicle, scene.target)):
            assert trace["type"] == "mesh3d"
            for point in zip(trace["x"], trace["y"], trace["z"]):
                near(math.dist(point, sphere.pose.position_m), sphere.radius_m)
        length = config.vehicle_radius_m * 2.5
        # Independent +90 degree pitch: forward points down, left stays left, up points forward.
        endpoints = ((center[0], center[1], center[2] - length),
                     (center[0], center[1] + length, center[2]),
                     (center[0] + length, center[1], center[2]))
        for trace, endpoint in zip(traces[4:], endpoints):
            for axis, start, end in zip("xyz", center, endpoint):
                near(trace[axis][0], start)
                near(trace[axis][1], end)
        assert layout["scene"]["aspectmode"] == "data"
        assert "{visible: [mode !== 'voxels', mode !== 'points']}, [0, 1]" in page
        assert "场景查看" in page and "机体前 / 左 / 上轴" in page
        parsed = PageParser()
        parsed.feed(page)
        assert not parsed.external, "scene HTML requires external scripts/styles"
        summary = page_summary(page)["scene"]
        assert summary["background_points"] == 4 and summary["target"]["radius_m"] == 0.5
        checks.append("actual Plotly spheres, full 3D FLU axes and persistent visibility")

        before = hashlib.sha256(background.read_bytes()).hexdigest()
        config_path, scene_output = folder / "physical.json", folder / "scene.html"
        for vehicle_radius, target_radius in ((0.2, 0.3), (0.45, 0.7)):
            config_path.write_text(json.dumps({"vehicle": {"radius_m": vehicle_radius},
                                              "target": {"radius_m": target_radius}}))
            config_before = hashlib.sha256(config_path.read_bytes()).hexdigest()
            result = subprocess.run([sys.executable, str(entry),
                                     str(background), "--config", str(config_path),
                                     "--output", str(scene_output),
                                     "--vehicle-position", "-1", "0", "1",
                                     "--vehicle-rpy-deg", "90", "0", "0",
                                     "--target-position", "2", "1", "1.5"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0 and scene_output.is_file(), result.stderr
            scene_page = scene_output.read_text()
            state = page_summary(scene_page)["scene"]
            assert state["background_points"] == 4
            assert state["vehicle"]["radius_m"] == vehicle_radius
            assert state["target"]["radius_m"] == target_radius
            assert state["vehicle"]["position_m"] == [-1, 0, 1]
            assert state["target"]["position_m"] == [2, 1, 1.5]
            assert len(state["vehicle"]["quaternion_xyzw"]) == 4
            for value, expected in zip(state["vehicle"]["quaternion_xyzw"],
                                       (math.sqrt(0.5), 0, 0, math.sqrt(0.5))):
                near(value, expected)
            rendered = page_figure(scene_page)[0]
            assert len(rendered) == 7
            for trace, name in zip(rendered[2:4], ("vehicle", "target")):
                sphere = state[name]
                for point in zip(trace["x"], trace["y"], trace["z"]):
                    near(math.dist(point, sphere["position_m"]), sphere["radius_m"])
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_before
        assert hashlib.sha256(background.read_bytes()).hexdigest() == before
        checks.append("unified scene CLI with explicit positions, rotated pose, changed rendered radii and unchanged inputs")

        default_output = folder / "default-scene.html"
        default_config = ROOT / "configs" / "devtools.json"
        default_before = hashlib.sha256(default_config.read_bytes()).hexdigest()
        result = subprocess.run([sys.executable, str(entry), str(background),
                                 "--output", str(default_output),
                                 "--vehicle-position", "0.2", "-0.3", "1",
                                 "--target-position", "2", "1", "1.5"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and default_output.is_file(), result.stderr
        default_page = default_output.read_text()
        state = page_summary(default_page)["scene"]
        expected_config = json.loads(default_config.read_text())
        assert state["vehicle"]["position_m"] == [0.2, -0.3, 1]
        assert state["target"]["position_m"] == [2, 1, 1.5]
        assert state["vehicle"]["quaternion_xyzw"] == [0, 0, 0, 1]
        for name in ("vehicle", "target"):
            near(state[name]["radius_m"], expected_config[name]["radius_m"])
        length = expected_config["vehicle"]["radius_m"] * 2.5
        endpoints = ((center[0] + length, center[1], center[2]),
                     (center[0], center[1] + length, center[2]),
                     (center[0], center[1], center[2] + length))
        traces = page_figure(default_page)[0]
        assert len(traces) == 7
        for trace, endpoint in zip(traces[4:], endpoints):
            for axis, start, end in zip("xyz", center, endpoint):
                near(trace[axis][0], start)
                near(trace[axis][1], end)
        assert hashlib.sha256(default_config.read_bytes()).hexdigest() == default_before
        assert hashlib.sha256(background.read_bytes()).hexdigest() == before
        checks.append("paired CLI positions use default physical configuration and identity FLU attitude")

        incomplete = [
            ["--vehicle-position", "0", "0", "1"],
            ["--target-position", "1", "0", "1"],
            ["--vehicle-rpy-deg", "0", "0", "0"],
            ["--config", str(config_path)],
            ["--vehicle-position", "0", "0", "1", "--config", str(config_path)],
            ["--target-position", "1", "0", "1", "--vehicle-rpy-deg", "0", "0", "0"],
        ]
        for index, options in enumerate(incomplete):
            rejected_output = folder / f"incomplete-scene-{index}.html"
            rejected = subprocess.run([sys.executable, str(entry), str(background),
                                       "--output", str(rejected_output), *options],
                                      cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert rejected.returncode != 0, f"accepted incomplete scene arguments: {options}"
            assert "--vehicle-position" in rejected.stderr
            assert "--target-position" in rejected.stderr
            assert not rejected_output.exists(), f"wrote HTML for incomplete scene: {options}"
        assert hashlib.sha256(background.read_bytes()).hexdigest() == before
        checks.append("single position or explicit attitude/config without paired positions refused before HTML generation")

    args.results.write_text(json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2) + "\n")
    print(f"passed: {len(checks)} groups; {args.results}")


if __name__ == "__main__":
    main()
