"""Verify map/scene geometry, unified viewing CLI, offline HTML and HTTP scope."""

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from devtools.config import SceneConfig
from devtools.mesh import MeshMap
from devtools.scene import Pose, Scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "map_viewer"))
import view_map


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
        source = folder / "exact & <map>.obj"
        mesh_text, expected_objects = obj_boxes()
        source.write_text(mesh_text, encoding="ascii")
        source_before = hashlib.sha256(source.read_bytes()).hexdigest()
        mesh_map = MeshMap.from_obj(source)
        page = view_map.build_mesh_html(mesh_map)
        traces, layout = page_figure(page)
        assert len(traces) == 3, "obstacles must share one draw call"
        assert traces[0]["type"] == traces[1]["type"] == "mesh3d"
        expected_vertices = [point for _, vertices, _ in expected_objects[:2] for point in vertices]
        assert list(zip(*(traces[0][axis] for axis in "xyz"))) == expected_vertices
        expected_faces = [tuple(index - 1 + offset for index in face)
                          for offset, (_, _, faces) in zip((0, 8), expected_objects[:2]) for face in faces]
        assert list(zip(*(traces[0][axis] for axis in "ijk"))) == expected_faces
        assert list(zip(*(traces[1][axis] for axis in "xyz"))) == list(expected_objects[2][1])
        assert list(zip(*(traces[1][axis] for axis in "ijk"))) == [(0, 1, 2), (0, 2, 3)]
        assert traces[0]["customdata"] == ["box_&lt;one&gt;"] * 8 + ["box_two"] * 8
        assert len(traces[2]["x"]) == 24 * 3, "box wireframe must omit face diagonals"
        for mesh in mesh_map.objects[:2]:
            expected_edges = {tuple(sorted((a, b))) for a, b in (
                (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7))}
            assert set(view_map.mesh_edges(mesh)) == expected_edges
        assert layout["scene"]["aspectmode"] == "data"
        summary = page_summary(page)
        assert summary["format"] == "obj" and summary["obstacle_count"] == 2
        assert summary["input_vertices"] == summary["display_vertices"] == 20
        assert summary["input_triangles"] == summary["display_triangles"] == 26
        assert summary["bounds_m"] == [[-3, 3], [-2, 2], [0, 2]]
        checks.append("exact independent OBJ vertices/faces, separate ground, physical crease edges and no display sampling")

        parsed = PageParser()
        parsed.feed(page)
        assert not parsed.external, "HTML requires external scripts or styles"
        assert {"mesh-mode", "ground-visible", "reset-camera", "map"} <= parsed.controls.keys()
        assert not {"point-size", "voxel-fill", "mode"} & parsed.controls.keys()
        assert "box_<one>" not in page and "window.invalid = 1</script>" not in page
        assert "{visible: [mode !== 'wire', groundControl.checked, mode !== 'solid']}, [0, 1, 2]" in page
        assert "'scene.camera':" in page
        assert "flex:1;min-height:0;overflow:hidden" in page
        checks.append("embedded Plotly, escaped names/metadata, background-only display controls and bounded viewport")

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
        checks.append("closed outward spheres, physical radius/bounds and independent volume")

        center = (0.2, -0.3, 1.0)
        config = SceneConfig(vehicle_radius_m=0.25, target_radius_m=0.5)
        scene = Scene.from_obj(source, config,
                               Pose(center, (0, math.sqrt(0.5), 0, math.sqrt(0.5))), (2, 1, 1.5))
        scene_page = view_map.build_mesh_html(mesh_map, scene=scene)
        rendered, _ = page_figure(scene_page)
        assert len(rendered) == 8
        for trace, sphere in zip(rendered[3:5], (scene.vehicle, scene.target)):
            for point in zip(*(trace[axis] for axis in "xyz")):
                near(math.dist(point, sphere.pose.position_m), sphere.radius_m)
        length = config.vehicle_radius_m * 2.5
        # Independent +90 degree pitch: front points down, left stays left, up points front.
        endpoints = ((center[0], center[1], center[2] - length),
                     (center[0], center[1] + length, center[2]),
                     (center[0] + length, center[1], center[2]))
        for trace, endpoint in zip(rendered[5:], endpoints):
            for axis, start, end in zip("xyz", center, endpoint):
                near(trace[axis][0], start)
                near(trace[axis][1], end)
        state = page_summary(scene_page)["scene"]
        assert state["background_vertices"] == 20 and state["background_triangles"] == 26
        assert scene.background.objects == mesh_map.objects
        assert "{visible: [mode !== 'wire', groundControl.checked, mode !== 'solid']}, [0, 1, 2]" in scene_page
        assert "场景查看" in scene_page and "机体前 / 左 / 上轴" in scene_page
        other_source = folder / "other.obj"
        other_source.write_text(mesh_text)
        other_scene = Scene.from_obj(other_source, config, Pose(center), (2, 1, 1.5))
        try:
            view_map.build_mesh_html(mesh_map, scene=other_scene)
        except ValueError:
            pass
        else:
            raise AssertionError("viewer accepted mismatched scene/map sources")
        checks.append("scene retains complete map, physical spheres, pitched FLU axes and background identity")

        output = folder / "public" / "map.html"
        command = [sys.executable, str(entry), str(source), "--output", str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and output.is_file(), result.stderr
        map_page = output.read_text()
        assert "scene" not in page_summary(map_page)
        assert len(page_figure(map_page)[0]) == 3
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
        checks.append("map-only CLI exports complete offline HTML without changing the input")

        config_path, scene_output = folder / "physical.json", folder / "scene.html"
        for vehicle_radius, target_radius in ((0.2, 0.3), (0.45, 0.7)):
            config_path.write_text(json.dumps({"vehicle": {"radius_m": vehicle_radius},
                                              "target": {"radius_m": target_radius}}))
            config_before = hashlib.sha256(config_path.read_bytes()).hexdigest()
            result = subprocess.run([sys.executable, str(entry), str(source),
                                     "--config", str(config_path), "--output", str(scene_output),
                                     "--vehicle-position", "-1", "0", "1",
                                     "--vehicle-rpy-deg", "90", "0", "0",
                                     "--target-position", "2", "1", "1.5"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0 and scene_output.is_file(), result.stderr
            state = page_summary(scene_output.read_text())["scene"]
            assert state["background_vertices"] == 20 and state["background_triangles"] == 26
            assert state["vehicle"]["radius_m"] == vehicle_radius
            assert state["target"]["radius_m"] == target_radius
            assert state["vehicle"]["position_m"] == [-1, 0, 1]
            assert state["target"]["position_m"] == [2, 1, 1.5]
            for value, expected in zip(state["vehicle"]["quaternion_xyzw"],
                                       (math.sqrt(0.5), 0, 0, math.sqrt(0.5))):
                near(value, expected)
            rendered, _ = page_figure(scene_output.read_text())
            assert len(rendered) == 8
            for trace, name in zip(rendered[3:5], ("vehicle", "target")):
                sphere = state[name]
                for point in zip(trace["x"], trace["y"], trace["z"]):
                    near(math.dist(point, sphere["position_m"]), sphere["radius_m"])
            length = vehicle_radius * 2.5
            rolled_endpoints = ((-1 + length, 0, 1), (-1, 0, 1 + length), (-1, -length, 1))
            for trace, endpoint in zip(rendered[5:], rolled_endpoints):
                for axis, end in zip("xyz", endpoint):
                    near(trace[axis][1], end)
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_before
        checks.append("scene CLI loads physical radii and renders full roll without changing configuration")

        default_output = folder / "default-scene.html"
        default_config = ROOT / "configs" / "devtools.json"
        default_before = hashlib.sha256(default_config.read_bytes()).hexdigest()
        result = subprocess.run([sys.executable, str(entry), str(source), "--output", str(default_output),
                                 "--vehicle-position", "0.2", "-0.3", "1",
                                 "--target-position", "2", "1", "1.5"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and default_output.is_file(), result.stderr
        default_page = default_output.read_text()
        state = page_summary(default_page)["scene"]
        expected_config = json.loads(default_config.read_text())
        assert state["vehicle"]["position_m"] == list(center)
        assert state["vehicle"]["quaternion_xyzw"] == [0, 0, 0, 1]
        for name in ("vehicle", "target"):
            near(state[name]["radius_m"], expected_config[name]["radius_m"])
        length = expected_config["vehicle"]["radius_m"] * 2.5
        endpoints = ((center[0] + length, center[1], center[2]),
                     (center[0], center[1] + length, center[2]),
                     (center[0], center[1], center[2] + length))
        for trace, endpoint in zip(page_figure(default_page)[0][5:], endpoints):
            for axis, start, end in zip("xyz", center, endpoint):
                near(trace[axis][0], start)
                near(trace[axis][1], end)
        assert hashlib.sha256(default_config.read_bytes()).hexdigest() == default_before
        checks.append("paired positions use default physical configuration and identity FLU attitude")

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
            rejected = subprocess.run([sys.executable, str(entry), str(source),
                                       "--output", str(rejected_output), *options],
                                      cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert rejected.returncode != 0
            assert "--vehicle-position" in rejected.stderr and "--target-position" in rejected.stderr
            assert not rejected_output.exists(), "incomplete scene wrote HTML"
        checks.append("incomplete scene arguments fail before output")

        for source_path, options in (
                (source, ["--port", "0"]), (source, ["--port", "65536"]),
                (source, ["--vehicle-position", "nan", "0", "1", "--target-position", "1", "0", "1"]),
                (source, ["--vehicle-position", "0", "0", "1", "--target-position", "1", "0", "inf"]),
                (source, ["--vehicle-position", "0", "0", "1", "--target-position", "1", "0", "1",
                          "--vehicle-rpy-deg", "0", "nan", "0"]),
                (source, ["--voxel-size", "0.2"]),
                (source.with_suffix(".ply"), []), (folder / "missing.obj", [])):
            rejected_output = folder / "invalid.html"
            rejected = subprocess.run([sys.executable, str(entry), str(source_path),
                                       "--output", str(rejected_output), *options],
                                      cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert rejected.returncode != 0 and not rejected_output.exists(), (source_path, options)
        invalid_obj = folder / "invalid.obj"
        invalid_obj.write_text("o ground\nv 0 0 0\n")
        rejected = subprocess.run([sys.executable, str(entry), str(invalid_obj),
                                   "--output", str(folder / "invalid.html")],
                                  cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert rejected.returncode != 0 and not (folder / "invalid.html").exists()
        checks.append("bad format/geometry/port/nonfinite poses and removed display parameters fail without output")

        source_alias = folder / "source-alias.html"
        source_alias.hardlink_to(source)
        source_symlink = folder / "source-link.html"
        source_symlink.symlink_to(source)
        config_alias = folder / "config-alias.html"
        config_alias.hardlink_to(config_path)
        config_symlink = folder / "config-link.html"
        config_symlink.symlink_to(config_path)
        config_before = hashlib.sha256(config_path.read_bytes()).hexdigest()
        for protected_output in (source, config_path, source_alias, source_symlink, config_alias, config_symlink):
            rejected = subprocess.run([sys.executable, str(entry), str(source),
                                       "--output", str(protected_output),
                                       "--vehicle-position", "0", "0", "1", "--target-position", "1", "0", "1",
                                       "--config", str(config_path)],
                                      cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert rejected.returncode != 0 and "不能覆盖" in rejected.stderr
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_before
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
        checks.append("input/config files and their hardlink/symlink aliases are protected")

        # The child chooses a free port while exercising the normal CLI server
        # lifecycle, including graceful Ctrl+C; no fixed port is assumed.
        server_script = """
import sys
sys.path.insert(0, sys.argv.pop(1))
import view_map
real = view_map.ThreadingHTTPServer
def report_server(address, handler):
    server = real((address[0], 0), handler)
    print(server.server_address[1], flush=True)
    return server
view_map.ThreadingHTTPServer = report_server
view_map.main()
"""
        server = subprocess.Popen([sys.executable, "-c", server_script, str(entry.parent),
                                   str(source), "--output", str(output), "--serve"],
                                  cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for _ in range(4):
                line = server.stdout.readline().strip()
                if line.isdecimal():
                    port = int(line)
                    break
            else:
                raise AssertionError("server did not report its bound port")
            url = f"http://127.0.0.1:{port}"
            for path in ("/map.html", "/"):
                with urlopen(url + path, timeout=5) as response:
                    assert response.read() == output.read_bytes()
                    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
                    assert response.headers["X-Content-Type-Options"] == "nosniff"
            for path in ("/../input.obj", "/%2e%2e/input.obj", "/AGENTS.md", "/map.html/extra", "/public/"):
                try:
                    urlopen(url + path, timeout=5)
                except HTTPError as error:
                    assert error.code == 404
                else:
                    raise AssertionError(f"server exposed unexpected path {path}")
            server.send_signal(signal.SIGINT)
            _, stderr = server.communicate(timeout=5)
            assert server.returncode == 0, stderr
            try:
                urlopen(url + "/", timeout=2)
            except URLError:
                pass
            else:
                raise AssertionError("server socket remained open after Ctrl+C")
        finally:
            if server.poll() is None:
                server.terminate()
                server.communicate(timeout=5)
        checks.append("foreground HTTP serves only its HTML and Ctrl+C closes the listener cleanly")

    args.results.write_text(json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2) + "\n")
    print(f"passed: {len(checks)} groups; {args.results}")


if __name__ == "__main__":
    main()
