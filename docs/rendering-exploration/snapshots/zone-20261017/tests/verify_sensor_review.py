"""Exercise the online sensor review service through its real HTTP boundary.

Run in the project's CUDA/graphics-capable Docker image. The service subprocess
owns its EGL context on its rendering worker; this client never creates a second
renderer. Expected images come from independent ray/box/ground geometry, not a
copy of the service's image conversion or postprocessing implementation.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BODY = {"position_m": [0.0, 0.0, 1.5], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
TARGET_A = {"position_m": [2.0, 0.0, 1.5]}
TARGET_B = {"position_m": [2.0, 5.0, 1.5]}
SIZE = (40.0, 40.0, 10.0)
BOX_BOUNDS = ((4.0, 4.5), (-2.5, 1.0), (0.2, 3.8))
BOX_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
EXPECTED_VIEW_IDS = ("tof/right", "tof/front", "tof/left", "tof/rear", "camera")
ATOL_M = 2e-5


def fixture():
    """Small asymmetric map makes row/column flips and mount order observable."""
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
    """Decode the public little-endian binary protocol without replacing NaN/inf."""
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
    """Independent Hamilton xyzw to active rotation, no shared transform helper."""
    x, y, z, w = xyzw
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))), dtype=float)


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(pose["quaternion_xyzw"])
    matrix[:3, 3] = pose["position_m"]
    return matrix


def pinhole_rays(config):
    """Top-down pixel centers; FLU +X forward, +Y left, +Z up."""
    height, width = config["height"], config["width"]
    y = (1 - 2*(np.arange(width)+0.5)/width) * math.tan(math.radians(config["hfov_deg"])/2)
    z = (1 - 2*(np.arange(height)+0.5)/height) * math.tan(math.radians(config["vfov_deg"])/2)
    zz, yy = np.meshgrid(z, y, indexing="ij")
    rays = np.stack((np.ones_like(yy), yy, zz), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def analytic_map(config, matrix, object_ids):
    """Independent closest closed-box/ground intersection and declared coverage."""
    rays = pinhole_rays(config) @ matrix[:3, :3].T
    origin = matrix[:3, 3]
    lower, upper = np.asarray(BOX_BOUNDS).T
    with np.errstate(divide="ignore", invalid="ignore"):
        first, second = (lower-origin)/rays, (upper-origin)/rays
        parallel = np.abs(rays) < 1e-14
        first = np.where(parallel, -np.inf, np.minimum(first, second))
        # Recompute the far slab independently: first has already been changed.
        second = np.where(parallel, np.inf, np.maximum((lower-origin)/rays, (upper-origin)/rays))
        entering, leaving = first.max(axis=-1), second.min(axis=-1)
        outside_parallel = (parallel & ((origin < lower) | (origin > upper))).any(axis=-1)
        box_hit = np.where((entering <= leaving) & (leaving > 0) & ~outside_parallel,
                           np.where(entering > 0, entering, leaving), np.inf)
        ground_hit = -origin[2] / rays[..., 2]
    ground_point = origin + np.where(np.isfinite(ground_hit), ground_hit, 0)[..., None]*rays
    ground_hit = np.where((ground_hit > 0) & (np.abs(ground_point[..., 0]) <= 20)
                          & (np.abs(ground_point[..., 1]) <= 20), ground_hit, np.inf)
    nearest = np.minimum(box_hit, ground_hit)
    identities = np.where(box_hit < ground_hit, object_ids.index("map/wall"), object_ids.index("map/ground"))
    guard = max(config["max_range_m"] * .001,
                4*32*np.finfo(np.float32).eps*max(1, config["max_range_m"], *np.abs(origin)))
    hit = nearest < config["max_range_m"] + guard
    raw = np.where(hit, nearest, 0)
    identities = np.where(hit, identities, -1)
    end = origin + config["max_range_m"]*rays
    bounds = np.asarray(((-20, 20), (-20, 20), (0, 10)))
    origin_known = np.all((origin >= bounds[:, 0]) & (origin <= bounds[:, 1]))
    covered = origin_known & ((end >= bounds[:, 0]) & (end <= bounds[:, 1])).all(axis=-1)
    point = origin + raw[..., None]*rays
    known_hit = origin_known & ((point >= bounds[:, 0]-ATOL_M) & (point <= bounds[:, 1]+ATOL_M)).all(axis=-1)
    valid = hit & known_hit & (nearest >= config["min_range_m"]) & (nearest <= config["max_range_m"])
    final = np.where(valid, nearest, np.nan)
    final = np.where(~hit & covered, np.inf, final)
    return raw, identities, final


def analytic_zone_map(config, matrix, object_ids):
    """Continuous box/ground minima from an independent active-constraint oracle.

    The oracle enumerates feasible face constraints rather than reproducing
    shader polygon clipping. The domain boundary gives the nearest unknown
    direction within a cell, so coverage covers the whole angular prefix.
    """
    from devtools.config import DepthConfig
    from verify_tof_zones import nearest_box_in_zone, nearest_ground_in_zone

    depth_config = DepthConfig(**config)
    shape = (config["height"], config["width"])
    raw = np.zeros(shape, dtype=np.float32)
    identities = np.full(shape, -1, dtype=np.int32)
    final = np.full(shape, np.nan, dtype=np.float32)
    origin = matrix[:3, 3]
    domain = ((-20., 20.), (-20., 20.), (0., 10.))
    origin_known = all(low <= p <= high for p, (low, high) in zip(origin, domain))
    guard = max(config["max_range_m"] * .001,
                4*32*np.finfo(np.float32).eps*max(1, config["max_range_m"], *np.abs(origin)))
    for row in range(shape[0]):
        for col in range(shape[1]):
            box_distance, _ = nearest_box_in_zone(depth_config, matrix, BOX_BOUNDS, row, col)
            ground_distance, _ = nearest_ground_in_zone(depth_config, matrix, ((-20, 20), (-20, 20)), row, col)
            nearest = min(box_distance, ground_distance)
            exit_distance, _ = nearest_box_in_zone(depth_config, matrix, domain, row, col)
            hit = nearest < config["max_range_m"] + guard
            if hit:
                raw[row, col] = nearest
                identities[row, col] = object_ids.index("map/wall" if box_distance < ground_distance else "map/ground")
                if (origin_known and nearest <= exit_distance + ATOL_M
                        and config["min_range_m"] <= nearest <= config["max_range_m"]):
                    final[row, col] = nearest
            elif origin_known and exit_distance >= config["max_range_m"]:
                final[row, col] = math.inf
    return raw, identities, final


class Service:
    """Isolated process and map root; every operation uses public HTTP routes."""

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
    # Choosing a registered asset again is explicit and returns its canonical mesh.
    reloaded = service.request("POST", "/api/map", {"map_id": first_id})
    assert check_map(reloaded, service.objects) == first_id
    return {"map_change_rejects_stale_sample": True, "canonical_objects": len(service.objects)}


def verify_no_target_analytic_and_binary(service):
    map_id = check_map(service.upload(), service.objects)
    sample = service.sample(map_id, request_id="analytic")
    assert sample["request_id"] == "analytic" and sample["map_id"] == map_id
    assert isinstance(sample["sample_id"], (str, int))
    assert geometry(sample)["instanced_draw_count"] == 2
    assert geometry(sample)["cpu_image_readbacks"] == 0
    assert "target" not in geometry(sample)["object_ids"]
    camera_finite = camera_nan = tof_empty = 0
    sensor_config = service.bootstrap["config"]["sensors"]
    mounts = sensor_config["tof_mounts"]
    if isinstance(mounts, dict):
        mounts = [mounts[name] for name in ("right", "front", "left", "rear")]
    mounts = [*mounts, sensor_config["camera_mount"]]
    for view, mount in zip(views(sample).values(), mounts):
        config = view["config"]
        shape = (config["height"], config["width"])
        assert view["object_ids"] == ["map/ground", "map/wall"]
        matrix = np.asarray(view["pose"]["matrix"])
        close(matrix, pose_matrix(BODY) @ pose_matrix(mount), 2e-7)
        close(matrix[:3, 3], view["pose"]["position_m"], 0)
        close(matrix[:3, :3], quaternion_matrix(view["pose"]["quaternion_xyzw"]), 2e-7)
        raw = array(view["raw_distance"], "float32", shape)
        identities = array(view["raw_object_index"], "int32", shape)
        measured = array(view["range_m"], "float32", shape)
        oracle = analytic_zone_map if view["id"].startswith("tof/") else analytic_map
        expected_raw, expected_ids, expected_final = oracle(config, matrix, view["object_ids"])
        if view["id"].startswith("tof/"):
            assert view["sampling"] == "zone_nearest"
            points = array(view["hit_point_sensor_m"], "float32", (*shape, 3))
            triangles = array(view["triangle_index"], "int32", shape)
            close(np.linalg.norm(points, axis=-1), raw)
            assert np.array_equal(triangles >= 0, identities >= 0)
        else:
            assert view["sampling"] == "pixel_center"
        close(raw, expected_raw)
        assert np.array_equal(identities, expected_ids), (view["id"], np.argwhere(identities != expected_ids)[:8])
        for classify in (np.isnan, np.isposinf, np.isfinite):
            assert np.array_equal(classify(measured), classify(expected_final)), view["id"]
        finite = np.isfinite(expected_final)
        close(measured[finite], expected_final[finite])
        assert not np.isneginf(measured).any()
        if view["id"] == "camera":
            camera_finite = int(np.isfinite(measured).sum())
            camera_nan = int(np.isnan(measured).sum())
            if "target_mask" in view:
                assert not array(view["target_mask"], "uint8", shape).any()
                assert not np.any(view["target_hint"])
                assert view["visible_fraction"] == view["unoccluded_pixels"] == 0
        else:
            tof_empty += int(np.isposinf(measured).sum())
    assert camera_finite > 0 and camera_nan > 0 and tof_empty > 0
    second = service.sample(map_id, request_id="analytic-repeat")
    assert geometry(second)["instanced_draw_count"] - geometry(sample)["instanced_draw_count"] == 2
    assert second["sample_id"] != sample["sample_id"]
    assert second["scene_version"] == sample["scene_version"]
    return {"views": 5, "camera_finite_pixels": camera_finite,
            "camera_nan_pixels": camera_nan, "tof_empty_pixels": tof_empty,
            "no_target_draws": 2, "analytic_tolerance_m": ATOL_M}


def verify_mounted_pose_rotation(service):
    map_id = check_map(service.upload(), service.objects)
    # A complete nonlevel attitude catches accidental yaw-only body conversion.
    quaternion = [.1, -.2, .3, math.sqrt(1-.1**2-.2**2-.3**2)]
    body = {"position_m": [.5, -.3, 1.8], "quaternion_xyzw": quaternion}
    sample = service.sample(map_id, request_id="roll-pitch-yaw", body=body)
    config = service.bootstrap["config"]["sensors"]
    mounts = config["tof_mounts"]
    if isinstance(mounts, dict):
        mounts = [mounts[name] for name in ("right", "front", "left", "rear")]
    for view, mount in zip(views(sample).values(), [*mounts, config["camera_mount"]]):
        expected = pose_matrix(body) @ pose_matrix(mount)
        close(view["pose"]["matrix"], expected, 2e-7)
        close(quaternion_matrix(view["pose"]["quaternion_xyzw"]), expected[:3, :3], 2e-7)
        raw = array(view["raw_distance"], "float32")
        oracle = analytic_zone_map if view["id"].startswith("tof/") else analytic_map
        expected_raw, expected_ids, _ = oracle(view["config"], np.asarray(view["pose"]["matrix"]), view["object_ids"])
        close(raw, expected_raw, 4e-5)
        assert np.array_equal(array(view["raw_object_index"], "int32"), expected_ids)
    return {"full_attitude_mounted_views": 5}


def verify_target_motion_same_pass(service):
    map_id = check_map(service.upload(), service.objects)
    first = service.sample(map_id, TARGET_A, "target-A")
    moved = service.sample(map_id, TARGET_B, "target-B")
    restored = service.sample(map_id, TARGET_A, "target-A-restored")
    initial = geometry(first)
    assert initial["instanced_draw_count"] == 3
    assert first["scene_version"] < moved["scene_version"] < restored["scene_version"]
    # An update never rewrites the JSON/owned binary image already returned.
    old_binary = views(first)["camera"]["raw_distance"]["data"]
    for index, value in enumerate((first, moved, restored)):
        info = geometry(value)
        assert info["scene_geometry_buffer_name"] == initial["scene_geometry_buffer_name"]
        assert info["scene_geometry_buffer_bytes"] == initial["scene_geometry_buffer_bytes"]
        assert info["geometry_uploads"] == initial["geometry_uploads"] == 1
        assert info["instanced_draw_count"] == 3*(index+1)
        assert info["object_transform_updates"] == initial["object_transform_updates"]+index
        assert info["object_transform_upload_bytes"] == initial["object_transform_upload_bytes"]+64*index
        for view in views(value).values():
            assert "target" in view["object_ids"]
    camera_a, camera_b, camera_again = (views(value)["camera"] for value in (first, moved, restored))
    mask_a = array(camera_a["target_mask"], "uint8")
    mask_b = array(camera_b["target_mask"], "uint8")
    assert mask_a.any() and not mask_b.any()
    assert camera_a["visible_fraction"] > 0 and camera_b["visible_fraction"] == 0
    assert camera_a["target_hint"][0] == 1 and not np.any(camera_b["target_hint"])
    assert np.array_equal(mask_a, array(camera_again["target_mask"], "uint8"))
    assert views(first)["camera"]["raw_distance"]["data"] == old_binary
    for name in ("raw_distance", "raw_object_index", "range_m", "target_mask"):
        assert camera_a[name] == camera_again[name], name
    # Full-pass identities and processed ranges must explain the visible mask.
    ids = array(camera_a["raw_object_index"], "int32")
    ranges = array(camera_a["range_m"], "float32")
    target_index = camera_a["object_ids"].index("target")
    assert np.array_equal(mask_a.astype(bool), (ids == target_index) & np.isfinite(ranges))
    target_only_distance = array(camera_a["target_only_distance"], "float32", mask_a.shape)
    target_only_index = array(camera_a["target_only_object_index"], "int32", mask_a.shape)
    assert np.isin(target_only_index, (-1, target_index)).all()
    config = camera_a["config"]
    unoccluded = ((target_only_index == target_index)
                  & (target_only_distance >= config["min_range_m"])
                  & (target_only_distance <= config["max_range_m"]))
    assert camera_a["unoccluded_pixels"] == int(unoccluded.sum()) > 0
    assert camera_a["visible_fraction"] == float(mask_a.sum())/camera_a["unoccluded_pixels"]
    rows, cols = np.nonzero(mask_a)
    expected_hint = [1, cols.min()/mask_a.shape[1], rows.min()/mask_a.shape[0],
                     (cols.max()+1)/mask_a.shape[1], (rows.max()+1)/mask_a.shape[0]]
    close(camera_a["target_hint"], expected_hint, 1e-7)
    return {"target_move_transform_bytes": 64, "geometry_uploads": 1,
            "sample_draws": 3, "target_visible_pixels": int(mask_a.sum())}


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
    # An invalid request must not poison the owner thread or the active scene.
    valid = service.sample(map_id, None, "still-alive")
    assert valid["request_id"] == "still-alive"
    return {"invalid_requests_rejected": len(bad_samples)+1+5+3+2, "worker_survived": True}


def verify_concurrent_snapshot_requests(service):
    map_id = check_map(service.upload(), service.objects)
    service.sample(map_id, TARGET_A, "warm")
    targets = [TARGET_A, TARGET_B, TARGET_A, TARGET_B]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(service.sample, map_id, target, f"parallel-{index}")
                   for index, target in enumerate(targets)]
        results = [future.result() for future in futures]
    assert len({item["sample_id"] for item in results}) == len(results)
    draw_counts = sorted(geometry(item)["instanced_draw_count"] for item in results)
    assert draw_counts == [6, 9, 12, 15], draw_counts
    for index, (target, item) in enumerate(zip(targets, results)):
        assert item["request_id"] == f"parallel-{index}" and item["map_id"] == map_id
        camera = views(item)["camera"]
        visible = bool(array(camera["target_mask"], "uint8").any())
        assert visible == (target == TARGET_A), (index, visible)
        assert item["scene_version"] == geometry(item)["scene_version"]
        assert geometry(item)["geometry_uploads"] == 1
    return {"concurrent_requests": len(results), "serialized_draw_counts": draw_counts}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/sensor_review/verification.json")
    args = parser.parse_args()
    groups = (verify_bootstrap, verify_upload_and_map_invalidation,
              verify_no_target_analytic_and_binary, verify_mounted_pose_rotation,
              verify_target_motion_same_pass, verify_request_validation,
              verify_concurrent_snapshot_requests, verify_transition)
    report = {"passed": False, "checks": [], "failures": []}
    try:
        with Service() as service:
            for group in groups:
                try:
                    details = group(service)
                    report["checks"].append({"name": group.__name__, "passed": True, **details})
                    print(f"PASS {group.__name__}", flush=True)
                except Exception as error:
                    failure = {"name": group.__name__, "passed": False,
                               "error": str(error), "traceback": traceback.format_exc()}
                    report["checks"].append(failure)
                    report["failures"].append(failure)
                    print(f"FAIL {group.__name__}: {error}", flush=True)
            service.log.flush()
            service.log.seek(0)
            report["server_log"] = service.log.read()
    except Exception as error:
        report["failures"].append({"name": "service_lifecycle", "error": str(error),
                                   "traceback": traceback.format_exc()})
    report["passed"] = not report["failures"]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "failures": len(report["failures"]), "results": str(args.results)}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

