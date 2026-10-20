"""Online inspection of the formal GPU sensors; run inside the project Docker.

The browser submits world state and displays diagnostic downloads. A dedicated
worker owns the Scene and CUDA ray renderer, so HTTP request concurrency never
moves renderer ownership between threads. No rendering or measurement policy
is implemented by this web adapter.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future, TimeoutError
from dataclasses import asdict, dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import queue
import signal
import sys
import tempfile
import threading
from urllib.parse import urlsplit

import numpy as np
import torch

# The script remains directly runnable without adding tools to the online core.
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from devtools.config import load_config
from devtools.mesh import MeshMap
from devtools.scene import Pose, Scene, Sphere
from devtools.sensors import RayRenderer, SensorSampler
from tools.sensor_review.fixtures.layout import TOF_ORDER, mounted_pose, poses_to_tensor
from tools.sensor_review.fixtures.perception import sample_measurement_diagnostics
from devtools.trajectory import TrajectoryState, fit_trajectory


LOG = logging.getLogger("sensor-review")
MAX_MAP_BYTES = 32 * 1024 * 1024
MAX_BODY_BYTES = 40 * 1024 * 1024
MAX_UPLOADS = 8


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value):
        raise RequestError(400, f"expected object with fields: {', '.join(required)}")
    if set(value) - set(required) - set(optional):
        raise RequestError(400, "request contains unsupported fields")


def _finite_vector(value, name):
    if (not isinstance(value, list) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise RequestError(400, f"{name} must contain three finite numbers")
    return tuple(float(v) for v in value)


def _array(tensor, dtype):
    """Explicit UI boundary: one owned little-endian download per image."""
    array = np.asarray(tensor.detach().cpu().numpy(), dtype=np.dtype(dtype).newbyteorder("<"))
    array = np.ascontiguousarray(array)
    return {"dtype": np.dtype(dtype).name, "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii")}


def _subray_grid(values, samples):
    """Present the actual subcell rays as a 2-D image, without resampling."""
    height, width = values.shape[:2]
    return values.reshape(height, width, samples, samples).permute(0, 2, 1, 3).reshape(
        height * samples, width * samples)


def _sample_diagnostics(view, raw, index, config):
    samples = config.samples_per_axis
    view.update(
        hit_point_sensor_m=_array(raw.hit_point_sensor_m[index], "float32"),
        triangle_index=_array(raw.triangle_index[index], "int32"),
        selected_subray=_array(raw.selected_subray[index], "int32"),
        subray_distance=_array(_subray_grid(raw.subray_distance_m[index], samples), "float32"),
        subray_object_index=_array(_subray_grid(raw.subray_object_index[index], samples), "int32"),
        subray_config={**asdict(config), "width": config.width * samples,
                       "height": config.height * samples, "samples_per_axis": 1},
        rays_per_cell=samples * samples,
    )


@dataclass(frozen=True)
class MapEntry:
    map_id: str
    label: str
    path: Path
    root: Path


class ReviewState:
    """One review session. Every method runs on the same graphics worker."""

    def __init__(self, roots, config_path, upload_root):
        self.config = load_config(config_path)
        self.entries = {}
        for root in roots:
            root = Path(root).resolve()
            if not root.is_dir():
                continue
            for candidate in sorted(root.rglob("*.obj")):
                path = candidate.resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    continue
                if len(self.entries) >= 512:
                    raise ValueError("map roots contain more than 512 OBJ assets; use a narrower root")
                map_id = "map-" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
                self.entries[map_id] = MapEntry(map_id, str(candidate.relative_to(root)), path, root)
        self.upload_root = Path(upload_root).resolve()
        self.uploads = 0
        self.active_id = None
        self.scene = None
        self.target_present = False
        self.renderer = None
        self.sampler = None
        self.scene_revision = 0
        self.sample_id = 0

    def bootstrap(self, _payload=None):
        return {"maps": [{"id": e.map_id, "label": e.label} for e in self.entries.values()],
                "config": asdict(self.config), "torch_version": torch.__version__}

    def _read_map(self, entry):
        # Recheck resolution when opening: a symlink changed after discovery
        # must not turn a registered asset into arbitrary filesystem access.
        path = entry.path.resolve()
        if not path.is_relative_to(entry.root) or not path.is_file():
            raise RequestError(404, "registered map is no longer available inside its map root")
        if path.stat().st_size > MAX_MAP_BYTES:
            raise RequestError(413, "OBJ map exceeds the 32 MiB limit")
        try:
            return MeshMap.from_obj(path)
        except (ValueError, UnicodeError) as exc:
            raise RequestError(400, f"invalid project OBJ map: {exc}") from exc

    def load_map(self, payload):
        if isinstance(payload, dict) and "map_id" in payload:
            _fields(payload, ("map_id",))
            if not isinstance(payload["map_id"], str) or payload["map_id"] not in self.entries:
                raise RequestError(404, "unknown map_id; select an asset from /api/bootstrap")
            entry = self.entries[payload["map_id"]]
            mesh = self._read_map(entry)
        else:
            _fields(payload, ("filename", "obj_text"))
            filename, text = payload["filename"], payload["obj_text"]
            if (not isinstance(filename, str) or not filename or len(filename) > 160
                    or "/" in filename or "\\" in filename
                    or not filename.lower().endswith(".obj")):
                raise RequestError(400, "filename must be a simple .obj file name without a path")
            if not isinstance(text, str):
                raise RequestError(400, "obj_text must contain OBJ text")
            try:
                data = text.encode("ascii")
            except UnicodeEncodeError as exc:
                raise RequestError(400, "project OBJ maps use ASCII records") from exc
            if len(data) > MAX_MAP_BYTES:
                raise RequestError(413, "OBJ map exceeds the 32 MiB limit")
            if self.uploads >= MAX_UPLOADS:
                raise RequestError(409, "this session has reached its eight-upload limit; restart to clear uploads")
            # User filenames are labels only, never paths or overwrite targets.
            with tempfile.NamedTemporaryFile(dir=self.upload_root, suffix=".obj", delete=False) as stream:
                stream.write(data)
                path = Path(stream.name)
            map_id = "upload-" + hashlib.sha256(data).hexdigest()[:20]
            entry = MapEntry(map_id, filename, path, self.upload_root)
            try:
                mesh = self._read_map(entry)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            self.entries[map_id] = entry
            self.uploads += 1

        # Parsing completes before replacing the current scene or GPU context.
        self._close_renderer()
        self.active_id = entry.map_id
        self.scene = Scene(mesh, Sphere(Pose((0., 0., 1.5)), self.config.vehicle_radius_m),
                           Sphere(Pose((4., 0., 1.5)), self.config.target_radius_m))
        self.target_present = False
        self.scene_revision += 1
        return {"map": {"id": entry.map_id, "name": entry.label,
                        "objects": [{"id": "map/" + obj.name, "kind": obj.kind,
                                     "vertices": obj.vertices_m, "triangles": obj.triangles}
                                    for obj in mesh.objects],
                        "bounds_m": mesh.bounds_m,
                        "declared_bounds_m": self.scene.get_geometry().map_bounds_m},
                "scene_revision": self.scene_revision}

    def sample(self, payload):
        _fields(payload, ("request_id", "map_id", "vehicle", "target"))
        request_id = payload["request_id"]
        if (isinstance(request_id, bool) or not isinstance(request_id, (str, int))
                or len(str(request_id)) > 128):
            raise RequestError(400, "request_id must be a short string or integer")
        map_id = payload["map_id"]
        if not isinstance(map_id, str) or map_id not in self.entries:
            raise RequestError(404, "unknown map_id")
        if map_id != self.active_id or self.scene is None:
            raise RequestError(409, "map_id is not the loaded map; reload it before sampling")
        _fields(payload["vehicle"], ("position_m", "quaternion_xyzw"))
        try:
            body = Pose(payload["vehicle"]["position_m"], payload["vehicle"]["quaternion_xyzw"])
        except (ValueError, TypeError, OverflowError) as exc:
            raise RequestError(400, f"invalid vehicle pose: {exc}") from exc
        target = payload["target"]
        if target is not None:
            _fields(target, ("position_m",))
            target_position = _finite_vector(target["position_m"], "target.position_m")
        else:
            target_position = None

        try:
            return self._render_sample(payload, body, target_position)
        except Exception:
            # The renderer closes itself after some allocation/interop errors.
            # Do not retain a poisoned context or retry a failed sample silently.
            # Preserve the original failure even if resource cleanup also fails.
            try:
                self._close_renderer()
            except Exception:
                LOG.exception("renderer cleanup after failed sample also failed")
            raise

    def _render_sample(self, payload, body, target_position):
        request_id, map_id = payload["request_id"], payload["map_id"]
        scene = self.scene.with_vehicle_pose(body)
        if target_position is not None:
            scene = scene.with_target_position(target_position)
        present = target_position is not None
        excluded = {"vehicle"} if present else {"vehicle", "target"}
        snapshot = scene.get_geometry(exclude_ids=excluded)
        if self.renderer is None or present != self.target_present:
            self._close_renderer()
            self.renderer = RayRenderer(snapshot)
            self.sampler = SensorSampler(self.renderer, runtime=self.config.sampling_runtime)
        else:
            self.renderer.sync(snapshot)
        if (body != self.scene.vehicle.pose or present != self.target_present
                or (present and target_position != self.scene.target.pose.position_m)):
            self.scene_revision += 1
        self.scene, self.target_present = scene, present
        sensors = self.config.sensors
        with torch.no_grad():
            diagnostic = sample_measurement_diagnostics(
                self.sampler, sensors, poses_to_tensor((body,)), snapshot,
                self.config.perception,
                target_id="target" if present else None)
            # Only these UI exports cross the device boundary. No GPU frame is
            # retained in session state after this response has been serialized.
            tof_matrices = diagnostic.tof_poses.detach().cpu().numpy().reshape(4, 4, 4)
            camera_matrix = diagnostic.camera_poses[0].detach().cpu().numpy()
            views = []
            labels = {"right": "右 ToF", "front": "前 ToF", "left": "左 ToF", "rear": "后 ToF"}
            for index, name in enumerate(TOF_ORDER):
                view = self._view(
                    "tof/" + name, labels[name], sensors.tof,
                    mounted_pose(body, sensors.tof_mounts[index]), tof_matrices[index],
                    diagnostic.tof_raw.distance_m[index], diagnostic.tof_raw.object_index[index],
                    diagnostic.layout.tof_m[0, index], diagnostic.tof_raw.object_ids)
                view.update(sampling="weighted_quantile_subrays",
                            reduction={"kind": "weighted_quantile", "quantile": sensors.tof_depth_quantile,
                                       "signal_floor": sensors.tof_signal_floor})
                _sample_diagnostics(view, diagnostic.tof_raw, index, sensors.tof)
                views.append(view)
            camera = diagnostic.layout.camera
            view = self._view("camera", "前向相机", sensors.camera,
                              mounted_pose(body, sensors.camera_mount), camera_matrix,
                              diagnostic.camera_raw.distance_m[0],
                              diagnostic.camera_raw.object_index[0], camera.depth.range_m[0],
                              diagnostic.camera_raw.object_ids)
            view["sampling"] = "nearest_subrays"
            view["reduction"] = {"kind": "nearest"}
            view["recognition_source"] = "development_environment"
            view["target_only_source"] = "analytic_sphere_geometry"
            _sample_diagnostics(view, diagnostic.camera_raw, 0, sensors.camera)
            view["target_coverage"] = _array(camera.target_coverage[0], "float32")
            view.update(target_mask=_array(camera.target_mask[0], "uint8"),
                        target_hint=camera.target_hint[0].detach().cpu().tolist(),
                        visible_fraction=float(camera.visible_fraction[0].detach().cpu()),
                        unoccluded_pixels=int(camera.unoccluded_pixels[0].detach().cpu()),
                        unoccluded_samples=int(camera.unoccluded_samples[0].detach().cpu()))
            if present:
                view["target_only_distance"] = _array(diagnostic.target_raw.distance_m[0], "float32")
                view["target_only_object_index"] = _array(diagnostic.target_raw.object_index[0], "int32")
                view["target_only_subray_distance"] = _array(_subray_grid(
                    diagnostic.target_raw.subray_distance_m[0], sensors.camera.samples_per_axis), "float32")
                view["target_only_subray_object_index"] = _array(_subray_grid(
                    diagnostic.target_raw.subray_object_index[0], sensors.camera.samples_per_axis), "int32")
            views.append(view)
        self.sample_id += 1
        return {"request_id": request_id, "map_id": map_id, "sample_id": self.sample_id,
                "scene_revision": self.scene_revision,
                "scene_version": diagnostic.layout.scene_version, "t_ref": diagnostic.layout.t_ref,
                "diagnostics": {"geometry_info": self.renderer.geometry_info,
                                "memory_info": self.renderer.memory_info,
                                "execution_info": diagnostic.layout.execution_info,
                                "perception_source": "development_environment"},
                "views": views}

    @staticmethod
    def _view(view_id, label, config, pose, matrix, raw_distance, raw_index, ranges, object_ids):
        return {"id": view_id, "label": label, "config": asdict(config),
                "pose": {"position_m": matrix[:3, 3].tolist(),
                         "quaternion_xyzw": pose.quaternion_xyzw, "matrix": matrix.tolist()},
                "raw_distance": _array(raw_distance, "float32"),
                "raw_object_index": _array(raw_index, "int32"),
                "range_m": _array(ranges, "float32"), "object_ids": object_ids}

    def transition(self, payload):
        _fields(payload, ("start_position_m", "end_position_m", "duration_s"))
        start = _finite_vector(payload["start_position_m"], "start_position_m")
        end = _finite_vector(payload["end_position_m"], "end_position_m")
        duration = payload["duration_s"]
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or not .1 <= duration <= 10):
            raise RequestError(400, "duration_s must be finite and in [0.1,10]")
        duration_tensor = torch.tensor(duration, dtype=torch.float64)
        zero = torch.zeros(3, dtype=torch.float64)
        scalar_zero = torch.zeros((), dtype=torch.float64)

        def state(position):
            return TrajectoryState(torch.tensor(position, dtype=torch.float64), zero, zero,
                                   scalar_zero, scalar_zero)

        trajectory = fit_trajectory(state(start), state(end), duration_tensor)
        times = torch.linspace(0., duration, max(2, math.ceil(duration * 60) + 1), dtype=torch.float64)
        sampled = trajectory.sample(times)
        # Rotation follows the same smooth translation progress, independently
        # of whether start and end positions happen to be identical.
        progress_trajectory = fit_trajectory(state((0., 0., 0.)), state((1., 0., 0.)), duration_tensor)
        progress = progress_trajectory.sample(times).position[:, 0]
        return {"duration_s": float(duration), "times": times.tolist(),
                "positions_m": sampled.position.tolist(), "progress": progress.tolist(),
                "velocities_m_s": sampled.velocity.tolist(),
                "accelerations_m_s2": sampled.acceleration.tolist()}

    def _close_renderer(self):
        self.sampler = None
        renderer, self.renderer = self.renderer, None
        if renderer is not None:
            renderer.close()

    def close(self):
        self._close_renderer()


class ReviewWorker:
    """Bounded command queue; renderer creation, use and close share a thread."""

    def __init__(self, roots, config_path):
        self.commands = queue.Queue(maxsize=16)
        self.ready = Future()
        self.closed = False
        self.thread = threading.Thread(target=self._run, args=(roots, config_path),
                                       name="sensor-review-gpu", daemon=False)
        self.thread.start()
        self.ready.result(timeout=30)

    def _run(self, roots, config_path):
        state = None
        try:
            with tempfile.TemporaryDirectory(prefix="sensor-review-uploads-") as directory:
                state = ReviewState(roots, config_path, directory)
                self.ready.set_result(True)
                while True:
                    item = self.commands.get()
                    if item is None:
                        break
                    action, payload, future = item
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        methods = {"bootstrap": state.bootstrap, "map": state.load_map,
                                   "sample": state.sample, "transition": state.transition}
                        future.set_result(methods[action](payload))
                    except Exception as exc:
                        future.set_exception(exc)
        except Exception as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            else:
                LOG.exception("review worker failed")
        finally:
            if state is not None:
                state.close()

    def call(self, action, payload=None):
        if self.closed:
            raise RequestError(503, "review service is shutting down")
        future = Future()
        try:
            self.commands.put_nowait((action, payload, future))
        except queue.Full as exc:
            raise RequestError(503, "review queue is full; wait for the current sample") from exc
        try:
            return future.result(timeout=120)
        except TimeoutError as exc:
            future.cancel()
            raise RequestError(504, "review operation timed out; it may still be completing") from exc

    def close(self):
        self.closed = True
        self.commands.put(None)
        self.thread.join(timeout=120)


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, worker, web_root, vendor_root):
        self.worker = worker
        web_root, vendor_root = Path(web_root), Path(vendor_root)
        self.assets = {"/": web_root / "index.html", "/index.html": web_root / "index.html",
                       "/app.js": web_root / "app.js", "/style.css": web_root / "style.css"}
        for name in ("three.module.js", "three.core.js", "OrbitControls.js", "three-LICENSE.txt",
                     "package.json", "package-lock.json"):
            self.assets["/vendor/" + name] = vendor_root / name
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "SensorReview/1"

    def log_message(self, format, *args):
        LOG.info("%s %s", self.client_address[0], format % args)

    def _send(self, status, data, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _execute(self, action, payload=None):
        try:
            self._send(200, self.server.worker.call(action, payload))
        except RequestError as exc:
            self._send(exc.status, {"error": str(exc)})
        except (ValueError, TypeError, OverflowError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            LOG.exception("review operation %s failed", action)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/bootstrap":
            self._execute("bootstrap")
            return
        asset = self.server.assets.get(path)
        if asset is None or not asset.is_file():
            self._send(404, {"error": "unknown route or unavailable static asset"})
            return
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
                 ".txt": "text/plain; charset=utf-8"}
        self._send(200, asset.read_bytes(), types.get(asset.suffix, "application/octet-stream"))

    def do_POST(self):
        actions = {"/api/map": "map", "/api/sample": "sample", "/api/transition": "transition"}
        action = actions.get(urlsplit(self.path).path)
        if action is None:
            self._send(404, {"error": "unknown API route"})
            return
        # Same-origin browser writes only; CLI tools need not send Origin.
        origin = self.headers.get("Origin")
        if origin is not None and urlsplit(origin).netloc != self.headers.get("Host"):
            self._send(403, {"error": "cross-origin requests are not accepted"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0:
                raise RequestError(411, "Content-Length is required")
            if length > MAX_BODY_BYTES:
                raise RequestError(413, "request body exceeds the 40 MiB limit")
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                raise RequestError(400, "incomplete request body")

            def reject_constant(value):
                raise ValueError(f"non-finite JSON number {value} is not accepted")

            payload = json.loads(body, parse_constant=reject_constant)
        except RequestError as exc:
            self._send(exc.status, {"error": str(exc)})
            return
        except (ValueError, UnicodeError, TimeoutError, OSError) as exc:
            self._send(400, {"error": f"invalid JSON request: {exc}"})
            return
        self._execute(action, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--map-root", type=Path, action="append", default=None)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/devtools.json")
    parser.add_argument("--vendor-root", type=Path, default=Path("/opt/sensor-review-web"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = ReviewWorker(args.map_root or [REPOSITORY / "outputs"], args.config)
    try:
        server = ReviewHTTPServer((args.host, args.port), worker, Path(__file__).parent / "web", args.vendor_root)
    except Exception:
        worker.close()
        raise

    def shutdown(_signal, _frame):
        threading.Thread(target=server.shutdown, name="sensor-review-stop", daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    LOG.info("sensor review listening on %s:%d inside container", args.host, server.server_port)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.close()


if __name__ == "__main__":
    main()
