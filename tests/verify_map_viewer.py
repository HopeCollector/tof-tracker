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
from devtools.mesh import MeshMap
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


def obj_boxes():
    """Two exact rectangular solids and a separate plane, independently of mapgen."""
    faces = ((1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
             (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8))
    objects = []
    for name, lower, upper in (("box_<one>", (0.4, -0.7, 0), (1, 0.1, 1.4)),
                                ("box_two", (-1.4, 0.75, 0), (-1, 1.25, 2))):
        x, y, z = lower
        X, Y, Z = upper
        objects.append((name, ((x, y, z), (X, y, z), (X, Y, z), (x, Y, z),
                               (x, y, Z), (X, y, Z), (X, Y, Z), (x, Y, Z)), faces))
    objects.append(("ground", ((-3, -2, 0), (3, -2, 0), (3, 2, 0), (-3, 2, 0)),
                    ((1, 2, 3), (1, 3, 4))))
    rows = ["# unit metre", "# size_xyz 6 4 3", "# </script><script>window.invalid = 1</script>"]
    offset = 0
    for name, vertices, triangles in objects:
        rows.append("o " + name)
        rows.extend("v " + " ".join(str(v) for v in vertex) for vertex in vertices)
        rows.extend("f " + " ".join(str(index + offset) for index in triangle) for triangle in triangles)
        offset += len(vertices)
    return "\n".join(rows) + "\n", objects


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def vector_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def interval_cells(points, size, origin):
    """Find occupancy by physical intervals, independently of viewer bin/merge code.

    Continuous-coverage fixtures use dyadic coordinates away from half-cell
    boundaries. Decimal boundary cases below instead specify their initial bins
    explicitly, since a second floating-point division can classify them again.
    """
    intervals = []
    for axis, base in enumerate(origin):
        lo = math.floor((min(point[axis] for point in points) - base) / size) - 1
        hi = math.ceil((max(point[axis] for point in points) - base) / size) + 1
        intervals.append([(index, base + index * size - size / 2,
                           base + index * size + size / 2) for index in range(lo, hi + 1)])
    occupied = set()
    for point in points:
        cell = []
        for axis in range(3):
            matches = [index for index, lower, upper in intervals[axis]
                       if lower <= point[axis] < upper]
            assert len(matches) == 1, (point, axis, matches)
            cell.append(matches[0])
        occupied.add(tuple(cell))
    return occupied


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
        assert not limited.auto_voxel_size and not limited.voxels_coarsened
        near(limited.voxel_size, 0.2)
        assert limited.origin == (0, 0, 0)
        checks.append("bounded display, uniform raw sampling, order-independent unique cell selection")

        # The unchanged default 12,000-cell budget must preserve a complete
        # ground and tall pillar by merging whole neighboring cells.
        perimeter = [(x, y) for x in (-0.5, -0.25, 0, 0.25, 0.5)
                     for y in (-0.5, -0.25, 0, 0.25, 0.5)
                     if abs(x) == 0.5 or abs(y) == 0.5]
        shell = [(x, y, k * 0.25) for x, y in perimeter for k in range(49)]
        default_ground = [(-16 + i * 0.25, -16 + j * 0.25, 0)
                          for i in range(128) for j in range(128)]
        default_map = folder / "default-large-ground-and-pillar.ply"
        default_points = default_ground + shell
        default_map.write_text(ply(default_points, ("unit metre", "size_xyz 32 32 12 resolution 0.25")))
        default_data = view_map.read_map(default_map, max_points=5)
        assert default_data.auto_voxel_size and default_data.voxels_coarsened
        assert not default_data.voxels_limited and len(default_data.cells) <= 12_000
        assert default_data.coarsening_factor == 2 and default_data.coarsening_shift == 0
        near(default_data.voxel_size, 0.5)
        assert set(default_data.cells) == interval_cells(default_points, 0.5, (-15.875, -15.875, 0.125))
        assert interval_cells(default_ground, 0.5, default_data.origin) == {
            (i, j, 0) for i in range(64) for j in range(64)}
        checks.append("default 12000-voxel budget merges all large ground/pillar regions without sampling")

        ground = [(-4 + i * 0.25, -4 + j * 0.25, 0)
                  for i in range(32) for j in range(32)]
        analytic_points = ground + shell
        analytic = folder / "ground-and-pillar.ply"
        analytic.write_text(ply(analytic_points, ("unit metre", "size_xyz 8 8 12 resolution 0.25")))
        analytic_before = hashlib.sha256(analytic.read_bytes()).hexdigest()
        overview = view_map.read_map(analytic, max_points=5, max_voxels=120)
        assert overview.auto_voxel_size and overview.voxels_coarsened
        assert not overview.voxels_limited and 0 < len(overview.cells) <= 120
        near(overview.initial_voxel_size, 0.25)
        assert overview.initial_origin == (-4, -4, 0)
        near(overview.resolution, 0.25)
        expected_cells = interval_cells(analytic_points, overview.voxel_size, overview.origin)
        assert set(overview.cells) == expected_cells, "overview discarded occupied regions"
        assert len(overview.points) == 5
        assert interval_cells(overview.points, overview.voxel_size, overview.origin) != expected_cells
        ground_cells = interval_cells(ground, overview.voxel_size, overview.origin)
        ground_axes = [{cell[axis] for cell in ground_cells} for axis in range(3)]
        assert ground_cells == {(x, y, z) for x in ground_axes[0]
                                for y in ground_axes[1] for z in ground_axes[2]}
        shell_cells = interval_cells(shell, overview.voxel_size, overview.origin)
        for x, y in {(x, y) for x, y, _ in shell_cells}:
            layers = {z for cx, cy, z in shell_cells if (cx, cy) == (x, y)}
            assert layers == set(range(min(layers), max(layers) + 1)), "fixture has an untested gap"
            assert layers <= {z for cx, cy, z in overview.cells if (cx, cy) == (x, y)}, \
                "a vertical pillar was broken into sparse fragments"
        shuffled = folder / "ground-and-pillar-reverse-duplicate.ply"
        shuffled.write_text(ply(list(reversed(analytic_points)) + shell + ground,
                                 ("unit metre", "size_xyz 8 8 12 resolution 0.25")))
        shuffled_before = hashlib.sha256(shuffled.read_bytes()).hexdigest()
        reordered_overview = view_map.read_map(shuffled, max_points=5, max_voxels=120)
        assert reordered_overview.cells == overview.cells
        assert reordered_overview.voxel_size == overview.voxel_size
        assert reordered_overview.origin == overview.origin
        assert reordered_overview.coarsening_factor == overview.coarsening_factor
        assert reordered_overview.coarsening_shift == overview.coarsening_shift
        assert reordered_overview.initial_voxel_size == overview.initial_voxel_size
        assert reordered_overview.initial_origin == overview.initial_origin
        assert hashlib.sha256(analytic.read_bytes()).hexdigest() == analytic_before
        assert hashlib.sha256(shuffled.read_bytes()).hexdigest() == shuffled_before
        checks.append("independent interval coverage, continuous ground/pillar, order/duplicate invariance")

        # Finite points on both sides of every axis must eventually fit one
        # cell; a grid whose parent boundary remains at zero cannot do this.
        tiny_points = [(-64, -32, -16), (-0.25, 0.25, -0.25), (0.25, -0.25, 0.25), (64, 32, 16)]
        tiny = folder / "one-cell-budget.ply"
        tiny.write_text(ply(tiny_points, ("unit metre", "size_xyz 8 8 12 resolution 0.25")))
        tiny_before = hashlib.sha256(tiny.read_bytes()).hexdigest()
        tiny_overview = view_map.read_map(tiny, max_points=1, max_voxels=1)
        assert len(tiny_overview.cells) == 1 and tiny_overview.voxels_coarsened
        assert not tiny_overview.voxels_limited
        assert all(math.isfinite(value) for value in (*tiny_overview.origin, tiny_overview.voxel_size))
        assert interval_cells(tiny_points, tiny_overview.voxel_size, tiny_overview.origin) == set(tiny_overview.cells)
        assert hashlib.sha256(tiny.read_bytes()).hexdigest() == tiny_before
        checks.append("one-cell automatic budget terminates across positive/negative XYZ without discarding input")

        # These decimal half-grid points initially fall in bins 2 and 3 under
        # the public floating-point rule. Re-dividing against the final grid
        # would move the half-grid point to another parent. Hand-specified bins
        # and physical box bounds ensure coarsening preserves whole children.
        for resolution, extent, inside, boundary in ((0.1, 1, -0.3, -0.15),
                                                      (0.2, 2, -0.6, -0.3)):
            decimal_map = folder / f"decimal-half-grid-{resolution}.ply"
            decimal_points = [(inside, -extent / 2, 0), (boundary, -extent / 2, 0)]
            decimal_map.write_text(ply(decimal_points, ("unit metre",
                                   f"size_xyz {extent} {extent} 1 resolution {resolution}")))
            initial = view_map.read_map(decimal_map, voxel_size=resolution)
            assert initial.cells == ((2, 0, 0), (3, 0, 0))
            merged = view_map.read_map(decimal_map, max_voxels=1)
            assert merged.cells == ((1, 0, 0),)
            assert merged.coarsening_factor == 2 and merged.coarsening_shift == 0
            near(merged.voxel_size, 2 * resolution)
            expected_origin = (-4.5 * resolution, -4.5 * resolution, 0.5 * resolution)
            for actual, expected in zip(merged.origin, expected_origin):
                near(actual, expected)
            expected_center = (-2.5 * resolution, -4.5 * resolution, 0.5 * resolution)
            rendered = page_figure(view_map.build_html(merged, voxel_fill=1))[0]
            assert len(rendered[1]["x"]) == 8
            for axis, center in zip("xyz", expected_center):
                near(min(rendered[1][axis]), center - resolution)
                near(max(rendered[1][axis]), center + resolution)
            assert merged.points == tuple(decimal_points), "coarsening changed decimal coordinates"
        checks.append("decimal resolutions and half-grid boundaries retain initial bins and whole parent geometry")

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
        auto_page = view_map.build_html(overview)
        auto_summary = page_summary(auto_page)
        assert auto_summary["source_resolution_m"] == 0.25
        assert auto_summary["initial_voxel_size_m"] == 0.25
        assert auto_summary["initial_origin_m"] == [-4, -4, 0]
        assert auto_summary["auto_voxel_size"] and auto_summary["voxels_coarsened"]
        assert not auto_summary["voxels_limited"]
        assert auto_summary["voxel_size_m"] == overview.voxel_size
        assert auto_summary["origin_m"] == list(overview.origin)
        assert auto_summary["coarsening_factor"] == overview.coarsening_factor
        assert auto_summary["coarsening_shift"] == overview.coarsening_shift
        assert auto_summary["display_voxels"] == len(expected_cells)
        assert "源地图分辨率 0.25 m" in auto_page
        assert f"显示网格 {overview.voxel_size:g} m" in auto_page
        assert "自动粗化" in auto_page and "完整概览含点格" in auto_page
        assert "概览格不能用于判断源地图的柱宽" in auto_page
        assert "哈希抽样" not in auto_page and "非完整占用" not in auto_page
        fixed_summary = page_summary(view_map.build_html(limited))
        assert not fixed_summary["auto_voxel_size"] and not fixed_summary["voxels_coarsened"]
        assert fixed_summary["voxels_limited"] and fixed_summary["voxel_size_m"] == 0.2
        assert "非完整占用" in view_map.build_html(limited)
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

        coarse_scene = Scene.from_ply(analytic, config,
                                      Pose(center, (0, math.sqrt(0.5), 0, math.sqrt(0.5))), (2, 1, 1.5))
        original_background = coarse_scene.background
        original_spheres = (coarse_scene.vehicle, coarse_scene.target)
        coarse_page = view_map.build_html(overview, voxel_fill=1, scene=coarse_scene)
        coarse_summary = page_summary(coarse_page)
        assert coarse_summary["voxels_coarsened"] and not coarse_summary["voxels_limited"]
        assert coarse_summary["scene"]["background_points"] == len(analytic_points) > len(overview.points)
        assert coarse_scene.background is original_background
        assert coarse_scene.background.points_m == tuple(analytic_points)
        assert coarse_scene.background.resolution_m == 0.25
        assert coarse_scene.background.size_xyz_m == (8, 8, 12)
        assert (coarse_scene.vehicle, coarse_scene.target) == original_spheres
        traces, _ = page_figure(coarse_page)
        assert len(traces) == 7 and len(traces[0]["x"]) == 5
        assert list(zip(traces[0]["x"], traces[0]["y"], traces[0]["z"])) == list(overview.points)
        assert len(traces[1]["x"]) == 8 * len(expected_cells)
        for trace, position, radius in zip(traces[2:4], (center, (2, 1, 1.5)), (0.25, 0.5)):
            for point in zip(trace["x"], trace["y"], trace["z"]):
                near(math.dist(point, position), radius)
        for trace, endpoint in zip(traces[4:], endpoints):
            for axis, start, end in zip("xyz", center, endpoint):
                near(trace[axis][0], start)
                near(trace[axis][1], end)
        for name, position, radius in zip(("vehicle", "target"), (center, (2, 1, 1.5)), (0.25, 0.5)):
            assert coarse_summary["scene"][name]["position_m"] == list(position)
            near(coarse_summary["scene"][name]["radius_m"], radius)
        assert hashlib.sha256(analytic.read_bytes()).hexdigest() == analytic_before
        checks.append("coarsened display plus Scene preserves full original background, physical radii and pitched FLU pose")

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

        mesh_source = folder / "exact & <map>.obj"
        mesh_text, expected_objects = obj_boxes()
        mesh_source.write_text(mesh_text, encoding="ascii")
        mesh_before = hashlib.sha256(mesh_source.read_bytes()).hexdigest()
        mesh_map = MeshMap.from_obj(mesh_source)
        mesh_page = view_map.build_mesh_html(mesh_map)
        mesh_traces, mesh_layout = page_figure(mesh_page)
        assert len(mesh_traces) == 3, "each obstacle must not create a separate draw call"
        assert mesh_traces[0]["type"] == mesh_traces[1]["type"] == "mesh3d"
        expected_vertices = [point for _, vertices, _ in expected_objects[:2] for point in vertices]
        assert list(zip(*(mesh_traces[0][axis] for axis in "xyz"))) == expected_vertices
        expected_faces = [tuple(index - 1 + offset for index in face)
                          for offset, (_, _, faces) in zip((0, 8), expected_objects[:2]) for face in faces]
        assert list(zip(*(mesh_traces[0][axis] for axis in "ijk"))) == expected_faces
        assert list(zip(*(mesh_traces[1][axis] for axis in "xyz"))) == list(expected_objects[2][1])
        assert list(zip(*(mesh_traces[1][axis] for axis in "ijk"))) == [(0, 1, 2), (0, 2, 3)]
        assert mesh_traces[0]["customdata"] == ["box_&lt;one&gt;"] * 8 + ["box_two"] * 8
        assert len(mesh_traces[2]["x"]) == 24 * 3, "box wireframe must contain 12 edges, no face diagonals"
        for mesh in mesh_map.objects[:2]:
            assert len(view_map.mesh_edges(mesh)) == 12
        assert mesh_layout["scene"]["aspectmode"] == "data"
        parsed = PageParser()
        parsed.feed(mesh_page)
        assert not parsed.external
        assert {"mesh-mode", "ground-visible", "reset-camera", "map"} <= parsed.controls.keys()
        assert not {"point-size", "voxel-fill", "mode"} & parsed.controls.keys()
        assert "box_<one>" not in mesh_page and "window.invalid = 1</script>" not in mesh_page
        mesh_summary = page_summary(mesh_page)
        assert mesh_summary["format"] == "obj" and mesh_summary["obstacle_count"] == 2
        assert mesh_summary["input_vertices"] == mesh_summary["display_vertices"] == 20
        assert mesh_summary["input_triangles"] == mesh_summary["display_triangles"] == 26
        assert mesh_summary["bounds_m"] == [[-3, 3], [-2, 2], [0, 2]]
        assert "完整显示原始 mesh" in mesh_page and "[0, 1, 2]" in mesh_page
        checks.append("exact independent OBJ vertices/faces, all objects retained, crease-only edges and mesh-specific offline controls")

        mesh_scene = Scene.from_obj(mesh_source, config,
                                    Pose(center, (0, math.sqrt(0.5), 0, math.sqrt(0.5))), (2, 1, 1.5))
        scene_mesh_page = view_map.build_mesh_html(mesh_map, scene=mesh_scene)
        rendered = page_figure(scene_mesh_page)[0]
        assert len(rendered) == 8
        for trace, sphere in zip(rendered[3:5], (mesh_scene.vehicle, mesh_scene.target)):
            for point in zip(*(trace[axis] for axis in "xyz")):
                near(math.dist(point, sphere.pose.position_m), sphere.radius_m)
        length = config.vehicle_radius_m * 2.5
        pitched_endpoints = ((center[0], center[1], center[2] - length),
                             (center[0], center[1] + length, center[2]),
                             (center[0] + length, center[1], center[2]))
        for trace, endpoint in zip(rendered[5:], pitched_endpoints):
            for axis, end in zip("xyz", endpoint):
                near(trace[axis][1], end)
        state = page_summary(scene_mesh_page)["scene"]
        assert state["background_vertices"] == 20 and state["background_triangles"] == 26
        assert mesh_scene.background.objects == mesh_map.objects
        assert "{visible: [mode !== 'wire', groundControl.checked, mode !== 'solid']}, [0, 1, 2]" in scene_mesh_page
        checks.append("mesh scene preserves full geometry, physical spheres and pitched FLU axes; background controls exclude actors")

        mesh_output = folder / "mesh.html"
        mesh_command = [sys.executable, str(entry), str(mesh_source), "--output", str(mesh_output)]
        result = subprocess.run(mesh_command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and mesh_output.is_file(), result.stderr
        assert "scene" not in page_summary(mesh_output.read_text())
        result = subprocess.run([*mesh_command, "--vehicle-position", "0.2", "-0.3", "1",
                                 "--target-position", "2", "1", "1.5", "--vehicle-rpy-deg", "0", "90", "0"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        state = page_summary(mesh_output.read_text())["scene"]
        assert state["vehicle"]["position_m"] == [0.2, -0.3, 1]
        near(state["vehicle"]["radius_m"], expected_config["vehicle"]["radius_m"])
        for option, value in (("--voxel-size", "0.2"), ("--point-size", "3"),
                               ("--voxel-fill", "0.9"), ("--max-points", "100000"), ("--max-voxels", "12000")):
            rejected_output = folder / "rejected-mesh.html"
            rejected = subprocess.run([sys.executable, str(entry), str(mesh_source), "--output", str(rejected_output),
                                       option, value], cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert rejected.returncode != 0 and option in rejected.stderr
            assert not rejected_output.exists(), "OBJ accepted irrelevant PLY control"
        protected_output = folder / "input-alias.html"
        protected_output.hardlink_to(mesh_source)
        rejected = subprocess.run([sys.executable, str(entry), str(mesh_source), "--output", str(protected_output)],
                                  cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert rejected.returncode != 0 and "不能覆盖" in rejected.stderr
        assert hashlib.sha256(mesh_source.read_bytes()).hexdigest() == mesh_before
        checks.append("OBJ CLI selects mesh/scene automatically, refuses irrelevant PLY parameters and protects source aliases")

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
