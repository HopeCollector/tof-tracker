"""Verify PLY input, occupied display bins, cube geometry, offline HTML and HTTP scope."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "map_viewer"))
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
        metadata = re.search(r'<script id="viewer-summary" type="application/json">(.*?)</script>', page).group(1)
        summary = json.loads(metadata)
        assert summary["input_points"] == 6 and summary["display_voxels"] == 4
        match = re.search(r'Plotly\.newPlot\(\s*"map",\s*', page)
        traces, _ = json.JSONDecoder().raw_decode(page[match.end():])
        assert traces[0]["type"] == "scatter3d" and traces[0]["visible"] is False
        assert traces[0]["x"] == [p[0] for p in points]
        assert traces[1]["type"] == "mesh3d" and len(traces[1]["i"]) == 48
        assert '"aspectmode":"data"' in page and '"title":{"text":"X (m)"}' in page
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

    args.results.write_text(json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2) + "\n")
    print(f"passed: {len(checks)} groups; {args.results}")


if __name__ == "__main__":
    main()
