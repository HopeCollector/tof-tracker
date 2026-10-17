"""HTTP test fixtures and assertions shared by the current review checks.

This module starts the real review service in a subprocess and checks its public
protocol. It imports only NumPy and the standard library: no renderer, sensor
implementation, historical GL test, or GPU context is imported by the client.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BODY = {"position_m": [0.0, 0.0, 1.5], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
BOX_BOUNDS = ((4.0, 4.5), (-2.5, 1.0), (0.2, 3.8))
BOX_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
EXPECTED_VIEW_IDS = ("tof/right", "tof/front", "tof/left", "tof/rear", "camera")
ATOL_M = 2e-5


def fixture():
    """Generate an asymmetric project OBJ using independent test geometry."""
    (x0, x1), (y0, y1), (z0, z1) = BOX_BOUNDS
    objects = (
        {"name": "ground", "kind": "ground",
         "vertices": [(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)],
         "triangles": [(0, 1, 2), (0, 2, 3)]},
        {"name": "wall", "kind": "obstacle",
         "vertices": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                      (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
         "triangles": BOX_FACES},
    )
    lines = ["# unit metre", "# size_xyz 40 40 10"]
    offset = 1
    for obj in objects:
        lines.append("o " + obj["name"])
        lines.extend("v " + " ".join(str(v) for v in vertex) for vertex in obj["vertices"])
        lines.extend("f " + " ".join(str(v + offset) for v in face) for face in obj["triangles"])
        offset += len(obj["vertices"])
    return "\n".join(lines) + "\n", objects


def close(actual, expected, tolerance=ATOL_M):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True), (
        "values differ", float(np.nanmax(np.abs(actual - expected))))


def array(value, dtype=None, shape=None):
    """Decode the little-endian wire format, preserving raw NaN/Inf values."""
    assert set(value) == {"dtype", "shape", "data"}, set(value)
    allowed = {"float32": np.dtype("<f4"), "int32": np.dtype("<i4"), "uint8": np.dtype("u1")}
    assert value["dtype"] in allowed, value["dtype"]
    if dtype is not None:
        assert value["dtype"] == dtype, (value["dtype"], dtype)
    dimensions = value["shape"]
    assert isinstance(dimensions, list) and dimensions and all(type(v) is int and v > 0 for v in dimensions)
    if shape is not None:
        assert tuple(dimensions) == tuple(shape), (dimensions, shape)
    data = base64.b64decode(value["data"], validate=True)
    assert len(data) == math.prod(dimensions) * allowed[value["dtype"]].itemsize
    return np.frombuffer(data, dtype=allowed[value["dtype"]]).reshape(dimensions)


def quaternion_matrix(xyzw):
    """Independent Hamilton xyzw rotation, without production transforms."""
    x, y, z, w = xyzw
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))), dtype=float)


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(pose["quaternion_xyzw"])
    matrix[:3, 3] = pose["position_m"]
    return matrix


class Service:
    """A separate service process, private map root, and public HTTP requests."""

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sensor-review-test-")
        self.folder = Path(self.temp.name)
        self.map_root = self.folder / "maps"
        self.map_root.mkdir()
        self.obj_text, self.objects = fixture()
        (self.map_root / "fixture.obj").write_text(self.obj_text, encoding="ascii")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.log = (self.folder / "server.log").open("w+", encoding="utf-8")
        command = [sys.executable, str(ROOT / "tools/sensor_review/server.py"),
                   "--host", "127.0.0.1", "--port", str(port), "--map-root", str(self.map_root)]
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    raise RuntimeError("review server exited: " + self.log.read())
                try:
                    self.bootstrap = self.request("GET", "/api/bootstrap")
                    return self
                except (URLError, TimeoutError):
                    time.sleep(.1)
            raise RuntimeError("review server startup timed out")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.log.close()
        self.temp.cleanup()

    def request(self, method, path, body=None, *, expected=200, raw_body=None):
        data = raw_body if raw_body is not None else (
            json.dumps(body, allow_nan=False).encode() if body is not None else None)
        req = Request(self.url + path, data=data, method=method,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=120) as response:
                status, result = response.status, response.read()
        except HTTPError as error:
            status, result = error.code, error.read()
        if expected == "4xx":
            assert 400 <= status < 500, (method, path, status, result[:300])
        else:
            assert status == expected, (method, path, status, result[:300])
        value = json.loads(result)
        if expected == "4xx":
            assert "error" in value, value
        return value

    def upload(self, filename="uploaded.obj", obj_text=None):
        return self.request("POST", "/api/map", {"filename": filename,
                            "obj_text": self.obj_text if obj_text is None else obj_text})

    def sample(self, map_id, target=None, request_id="sample", body=None):
        return self.request("POST", "/api/sample", {"request_id": request_id, "map_id": map_id,
                            "vehicle": BODY if body is None else body, "target": target})


def geometry(sample):
    return sample["diagnostics"]["geometry_info"]


def views(sample):
    assert tuple(view["id"] for view in sample["views"]) == EXPECTED_VIEW_IDS
    return {view["id"]: view for view in sample["views"]}


def check_map(value, objects):
    asset = value["map"]
    assert isinstance(asset["id"], str) and asset["id"]
    assert len(asset["objects"]) == len(objects)
    for actual, expected in zip(asset["objects"], objects):
        assert actual["id"] == "map/" + expected["name"]
        assert actual["kind"] == expected["kind"]
        close(actual["vertices"], expected["vertices"], 0)
        assert actual["triangles"] == [list(face) for face in expected["triangles"]]
    close(asset["bounds_m"], ((-20, 20), (-20, 20), (0, 3.8)), 0)
    close(asset["declared_bounds_m"], ((-20, 20), (-20, 20), (0, 10)), 0)
    return asset["id"]


def verify_bootstrap(service):
    bootstrap = service.bootstrap
    assert bootstrap["torch_version"].split("+")[0] == "2.9.1", bootstrap["torch_version"]
    assert bootstrap["config"]["sensors"]["tof"]["width"] == 8
    maps = bootstrap["maps"]
    assert len(maps) == 1 and maps[0]["label"].endswith("fixture.obj"), maps
    assert set(maps[0]) == {"id", "label"}
    assert not maps[0]["id"].startswith("/") and ".." not in maps[0]["id"]
    value = service.request("POST", "/api/map", {"map_id": maps[0]["id"]})
    check_map(value, service.objects)
    return {"torch_version": bootstrap["torch_version"], "discovered_maps": len(maps)}


def verify_upload_and_map_invalidation(service):
    first = service.upload()
    first_id = check_map(first, service.objects)
    second = service.upload("second.obj", service.obj_text + "# second review asset\n")
    second_id = check_map(second, service.objects)
    assert first_id != second_id
    assert second["scene_revision"] > first["scene_revision"]
    service.request("POST", "/api/sample", {"request_id": "stale", "map_id": first_id,
                    "vehicle": BODY, "target": None}, expected="4xx")
    reloaded = service.request("POST", "/api/map", {"map_id": first_id})
    assert check_map(reloaded, service.objects) == first_id
    return {"map_change_rejects_stale_sample": True, "canonical_objects": len(service.objects)}


def verify_request_validation(service):
    map_id = check_map(service.upload(), service.objects)
    base = {"request_id": "invalid", "map_id": map_id, "vehicle": BODY, "target": None}
    bad_samples = [
        {}, {**base, "map_id": "unknown-map"},
        {**base, "vehicle": {**BODY, "position_m": [0, 1]}},
        {**base, "vehicle": {**BODY, "quaternion_xyzw": [0, 0, 0, 0]}},
        {**base, "vehicle": {**BODY, "position_m": [True, 0, 1]}},
        {**base, "target": {"position_m": [0, 1]}},
    ]
    for bad in bad_samples:
        service.request("POST", "/api/sample", bad, expected="4xx")
    service.request("POST", "/api/sample", raw_body=b'{"vehicle":{"position_m":[NaN,0,1]}}', expected="4xx")
    for filename in ("../escape.obj", "/tmp/escape.obj", "nested/file.obj", "nested\\file.obj", "wrong.html"):
        service.request("POST", "/api/map", {"filename": filename, "obj_text": service.obj_text}, expected="4xx")
    for bad_map in ({"map_id": "../fixture.obj"}, {"path": "/etc/passwd"},
                    {"filename": "bad.obj", "obj_text": "o invalid\nv 0 0 0\n"}):
        service.request("POST", "/api/map", bad_map, expected="4xx")
    for path in ("/api/raw?path=/etc/passwd", "/api/map?path=/etc/passwd"):
        service.request("GET", path, expected="4xx")
    valid = service.sample(map_id, None, "still-alive")
    assert valid["request_id"] == "still-alive"
    return {"invalid_requests_rejected": len(bad_samples)+1+5+3+2, "worker_survived": True}


def verify_transition(service):
    maximum_endpoint_error = 0.0
    for start, end in (([0, 0, 1], [5, -2, 3]), ([1, 2, 3], [1, 2, 3])):
        value = service.request("POST", "/api/transition", {
            "start_position_m": start, "end_position_m": end, "duration_s": 1.2})
        times = np.asarray(value["times"])
        positions = np.asarray(value["positions_m"])
        velocity = np.asarray(value["velocities_m_s"])
        acceleration = np.asarray(value["accelerations_m_s2"])
        assert value["duration_s"] == 1.2
        assert len(times) > 2 and times[0] == 0 and times[-1] == 1.2
        assert np.all(np.diff(times) > 0)
        assert positions.shape == velocity.shape == acceleration.shape == (len(times), 3)
        assert np.isfinite(positions).all() and np.isfinite(velocity).all() and np.isfinite(acceleration).all()
        close(positions[[0, -1]], [start, end], 1e-12)
        close(velocity[[0, -1]], np.zeros((2, 3)), 1e-12)
        close(acceleration[[0, -1]], np.zeros((2, 3)), 1e-12)
        u = times/value["duration_s"]
        expected_progress = 10*u**3-15*u**4+6*u**5
        close(value["progress"], expected_progress, 1e-12)
        close(positions, np.asarray(start)+(np.asarray(end)-start)*expected_progress[:, None], 1e-12)
        if start == end:
            close(positions, np.broadcast_to(start, positions.shape), 1e-12)
        maximum_endpoint_error = max(maximum_endpoint_error, float(np.abs(velocity[[0, -1]]).max()),
                                     float(np.abs(acceleration[[0, -1]]).max()))
    for body in ({"start_position_m": [0, 0], "end_position_m": [1, 1, 1], "duration_s": 1},
                 {"start_position_m": [0, 0, 0], "end_position_m": [1, 1, 1], "duration_s": 0}):
        service.request("POST", "/api/transition", body, expected="4xx")
    return {"moving_and_stationary_transitions": 2, "endpoint_derivative_max": maximum_endpoint_error}
