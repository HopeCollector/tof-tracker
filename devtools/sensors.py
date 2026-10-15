"""GPU mesh sensing of one immutable scene through its public geometry API.

The pinhole optical frame is FLU: +X forward, +Y left, +Z up. Pixels are
sampled once at their centers, rows down / columns right. Outputs are Euclidean
line-of-sight metres, never optical-axis depth. This is a noiseless development
sensor, not a calibrated ToF zone-response or RGB appearance model.
"""

from dataclasses import dataclass
import math

import numpy as np

from .config import DepthConfig, MountConfig, SensorConfig, _finite_number
from .mesh import MeshObject
from .scene import Pose, Scene, Sphere

TOF_ORDER = ("right", "front", "left", "rear")


def _readonly(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _roundoff_m(config, pose):
    return 32 * np.finfo(np.float32).eps * max(1, max(abs(x) for x in pose.position_m), config.max_range_m)


def _coverage(bounds, config, pose, distance):
    """Keep negative-space proof strict; tolerate float32 hit reconstruction."""
    if bounds is None:
        return np.ones(distance.shape, dtype=bool), np.zeros(distance.shape, dtype=bool)
    limits, origin = np.asarray(bounds), np.asarray(pose.position_m)
    direction = pixel_directions(config) @ pose.as_transform().rotation.T
    origin_inside = np.all((origin >= limits[:, 0]) & (origin <= limits[:, 1]))
    end = origin + config.max_range_m * direction
    full = origin_inside & np.all((end >= limits[:, 0]) & (end <= limits[:, 1]), axis=-1)
    point = origin + distance[..., None] * direction
    # Plane evaluation and range readback are float32. A surface
    # exactly on z=0 or a map wall can reconstruct just outside the boundary.
    # This tolerance applies only to measured hits, never to empty-space proof.
    tolerance = _roundoff_m(config, pose)
    known = origin_inside & np.all((point >= limits[:, 0] - tolerance) &
                                   (point <= limits[:, 1] + tolerance), axis=-1)
    return known, full


@dataclass(frozen=True)
class DepthFrame:
    """One image plus separate development diagnostics.

    range_m: finite within inclusive [min,max]; +inf only for a fully covered
    empty segment through max; NaN for unknown coverage, too-close surfaces or
    otherwise unavailable returns. object_index is -1 for no rasterized hit;
    its remaining values index object_ids, including too-close hits. Never feed
    these identities, the sampling pose or map metadata to the online model.
    """

    range_m: np.ndarray
    object_index: np.ndarray
    object_ids: tuple[str, ...]
    t_ref: float | None


@dataclass(frozen=True)
class CameraFrame:
    depth: DepthFrame
    target_mask: np.ndarray
    target_hint: np.ndarray
    visible_fraction: float
    unoccluded_pixels: int


@dataclass(frozen=True)
class RigFrame:
    """Raw observation fields; invalid ToF values still require model adaptation."""

    tof_m: np.ndarray
    target_hint: np.ndarray
    camera: CameraFrame
    t_ref: float | None


def pixel_directions(config: DepthConfig):
    """Unit FLU rays at pixel centers, shape (H,W,3), useful for diagnostics."""
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    u = (2 * (np.arange(config.width) + 0.5) / config.width - 1)
    v = (1 - 2 * (np.arange(config.height) + 0.5) / config.height)
    y, z = np.meshgrid(-u * math.tan(math.radians(config.hfov_deg) / 2),
                       v * math.tan(math.radians(config.vfov_deg) / 2))
    rays = np.stack((np.ones_like(y), y, z), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def mounted_pose(body_pose: Pose, mount: MountConfig) -> Pose:
    """Compose a complete sensor-to-body mount with the body-to-world pose."""
    if not isinstance(body_pose, Pose) or not isinstance(mount, MountConfig):
        raise TypeError("expected Pose and MountConfig")
    position = body_pose.body_to_world(mount.position_m)
    # Hamilton product: body-to-world * sensor-to-body, retaining all 3 axes.
    a, b = body_pose.quaternion_xyzw, mount.quaternion_xyzw
    xyz = (a[3] * np.array(b[:3]) + b[3] * np.array(a[:3]) + np.cross(a[:3], b[:3]))
    w = a[3] * b[3] - np.dot(a[:3], b[:3])
    return Pose(position, tuple(float(x) for x in (*xyz, w)))


def _sphere_mesh(sphere: Sphere):
    # Fixed inscribed mesh: 64 latitude / 128 longitude divisions. This is a
    # sensor approximation of the public analytic sphere, not safety inflation.
    lat, lon = 64, 128
    points = [(0.0, 0.0, 1.0)]
    for row in range(1, lat):
        theta = math.pi * row / lat
        for col in range(lon):
            phi = 2 * math.pi * col / lon
            points.append((math.sin(theta) * math.cos(phi),
                           math.sin(theta) * math.sin(phi), math.cos(theta)))
    south = len(points)
    points.append((0.0, 0.0, -1.0))
    faces = []
    for col in range(lon):
        nxt = (col + 1) % lon
        faces.append((0, col + 1, nxt + 1))
        for row in range(lat - 2):
            a, b = 1 + row * lon + col, 1 + row * lon + nxt
            faces.extend(((a, a + lon, b), (b, a + lon, b + lon)))
        last = 1 + (lat - 2) * lon
        faces.append((last + col, south, last + nxt))
    vertices = np.asarray(points) * sphere.radius_m + sphere.pose.position_m
    return vertices, np.asarray(faces)


_VERTEX_SHADER = """
#version 330
in vec3 position;
in vec3 normal;
in float object_id;
uniform mat3 world_to_sensor;
uniform vec3 sensor_origin;
uniform vec2 tangent_fov;
flat out vec4 plane_sensor;
flat out float identity;
void main() {
    vec3 point_sensor = world_to_sensor * (position - sensor_origin);
    vec3 normal_sensor = world_to_sensor * normal;
    plane_sensor = vec4(normal_sensor, dot(normal_sensor, point_sensor));
    identity = object_id;
    // No positive near plane or finite far plane: all front-facing distances
    // survive clipping. The fragment shader owns radial range/depth semantics.
    gl_Position = vec4(-point_sensor.y / tangent_fov.x,
                        point_sensor.z / tangent_fov.y, 0.0, point_sensor.x);
}
"""
_FRAGMENT_SHADER = """
#version 330
flat in vec4 plane_sensor;
flat in float identity;
uniform float render_radius;
uniform vec2 tangent_fov;
uniform vec2 image_size;
layout(location=0) out vec2 measurement;
void main() {
    vec2 pixel = 2.0 * gl_FragCoord.xy / image_size - 1.0;
    vec3 ray = vec3(1.0, -pixel.x * tangent_fov.x, pixel.y * tangent_fov.y);
    // Rasterization selected the triangle. Evaluate its flat plane here rather
    // than interpolating huge clipped varyings near the eye plane (w=0).
    float x = plane_sensor.w / dot(plane_sensor.xyz, ray);
    float r = x * length(ray);
    if (r < 0.0 || r > render_radius) discard;
    // Monotonic radial depth orders surfaces on the same pixel ray correctly.
    gl_FragDepth = r / render_radius;
    measurement = vec2(r, identity);
}
"""


class MeshSensor:
    """Prepare one scene once, then sample full world poses on an EGL GPU.

    Only scene.get_geometry(exclude_ids={'vehicle'}) is read. Buffers are reused
    for all poses and configurations; sample_many currently loops render/readback
    on one context, not a vectorized GPU batch. Use one instance per thread,
    close it (or use with), and create a new instance for changed scene facts.
    No CPU fallback is accepted. Context creation needs graphics-capable Docker.
    """

    def __init__(self, scene: Scene):
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        import moderngl

        self._closed = False
        self._resources = []
        self._targets = {}
        self._snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        self.object_ids = tuple(obj.object_id for obj in self._snapshot.objects)
        if len(self.object_ids) >= 2**24:
            raise ValueError("too many objects for exact float32 diagnostic IDs")
        self._target_index = self.object_ids.index("target")
        self._ctx = moderngl.create_context(standalone=True, backend="egl", require=330)
        try:
            self.renderer_info = {name: self._ctx.info[name] for name in
                                  ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
            renderer = self.renderer_info["GL_RENDERER"].lower()
            if any(tag in renderer for tag in ("llvmpipe", "softpipe", "swrast", "software", "lavapipe", "swiftshader")):
                raise RuntimeError(f"GPU renderer required; got {renderer}")
            self._program = self._ctx.program(vertex_shader=_VERTEX_SHADER,
                                               fragment_shader=_FRAGMENT_SHADER)
            self._resources.append(self._program)
            blocks, offset = [], 0
            for index, obj in enumerate(self._snapshot.objects):
                shape = obj.geometry
                if isinstance(shape, MeshObject):
                    vertices, faces = np.asarray(shape.vertices_m), np.asarray(shape.triangles)
                elif isinstance(shape, Sphere):
                    vertices, faces = _sphere_mesh(shape)
                else:
                    raise TypeError(f"unsupported public geometry: {type(shape).__name__}")
                triangles = vertices[faces]
                normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                triangles = triangles.reshape(-1, 3)
                block = np.column_stack((triangles, np.repeat(normals, 3, axis=0),
                                         np.full(len(triangles), index + 1))).astype("f4")
                if not np.all(np.isfinite(block)):
                    raise ValueError("scene geometry must be representable in float32")
                blocks.append(block)
                if index == self._target_index:
                    self._target_vertices = (offset, len(block))
                offset += len(block)
            buffer = self._ctx.buffer(np.concatenate(blocks).tobytes())
            self._resources.append(buffer)
            self._vao = self._ctx.vertex_array(self._program, [(buffer, "3f 3f 1f", "position", "normal", "object_id")])
            self._resources.append(self._vao)
            self.triangle_count = offset // 3
            self.geometry_uploads = 1
            self._ctx.enable_only(moderngl.DEPTH_TEST)
            self._ctx.depth_func = "<"
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Release all scene/framebuffer/context resources, once."""
        if self._closed:
            return
        self._closed = True
        for resource in reversed(self._resources):
            resource.release()
        self._ctx.release()

    def _rasterize(self, config, pose, target_only=False):
        if self._closed:
            raise RuntimeError("MeshSensor is closed")
        if not isinstance(config, DepthConfig) or not isinstance(pose, Pose):
            raise TypeError("expected DepthConfig and complete sensor-world Pose")
        if (max(abs(x) for x in pose.position_m) > np.finfo(np.float32).max / 2 or
                config.max_range_m > np.finfo(np.float32).max / 2 or
                min(math.tan(math.radians(fov) / 2) for fov in
                    (config.hfov_deg, config.vfov_deg)) < np.finfo(np.float32).tiny):
            raise ValueError("pose, range and FOV must be representable by the float32 GPU backend")
        size = (config.width, config.height)
        if size not in self._targets:
            color = self._ctx.texture(size, 2, dtype="f4")
            depth = self._ctx.depth_renderbuffer(size)
            fbo = self._ctx.framebuffer([color], depth)
            self._resources.extend((color, depth, fbo))
            self._targets[size] = fbo
        fbo = self._targets[size]
        fbo.use()
        fbo.clear(0, 0, 0, 0, depth=1.0)
        self._ctx.viewport = (0, 0, *size)
        world_to_sensor = pose.as_transform().rotation.T.astype("f4")
        self._program["world_to_sensor"].write(world_to_sensor.T.tobytes())
        self._program["sensor_origin"].value = pose.position_m
        self._program["image_size"].value = size
        self._program["tangent_fov"].value = (math.tan(math.radians(config.hfov_deg) / 2),
                                               math.tan(math.radians(config.vfov_deg) / 2))
        # A guard shell prevents range-rounding uncertainty from becoming an
        # empty pixel at the render cutoff. Near-max observed hits stay N/A if
        # rounding alone could move them back into valid range.
        self._program["render_radius"].value = config.max_range_m + max(
            config.max_range_m * 0.001, 4 * _roundoff_m(config, pose))
        if target_only:
            first, count = self._target_vertices
            self._vao.render(first=first, vertices=count)
        else:
            self._vao.render()
        pixels = np.frombuffer(fbo.read(components=2, dtype="f4", alignment=1), dtype="f4")
        pixels = pixels.reshape(config.height, config.width, 2)[::-1]
        return pixels[..., 0].copy(), pixels[..., 1].astype("i4") - 1

    def sample(self, config: DepthConfig, sensor_pose: Pose) -> DepthFrame:
        """Configuration + full sensor world pose + prepared scene -> one image.

        Known map_bounds_m is the complete domain supplied by Scene. A valid
        hit must lie before leaving it; a miss is +inf only when the entire
        [origin, origin + max_range*ray] segment is covered. Without declared
        bounds, explicit surface returns are kept but misses stay NaN.
        Surfaces closer than min_range block farther ones and produce NaN.
        """
        distance, identity = self._rasterize(config, sensor_pose)
        hit = identity >= 0
        known_hit, full_coverage = _coverage(self._snapshot.map_bounds_m, config, sensor_pose, distance)
        result = np.full(distance.shape, np.nan, dtype="f4")
        valid = hit & known_hit & (distance >= config.min_range_m) & (distance <= config.max_range_m)
        result[valid] = distance[valid]
        # Too-close hits must remain NaN even when the whole far segment is known.
        empty = (~hit | (distance > config.max_range_m + _roundoff_m(config, sensor_pose))) & full_coverage
        result[empty] = np.inf
        return DepthFrame(_readonly(result), _readonly(identity), self.object_ids, self._snapshot.t_ref)

    def sample_many(self, config: DepthConfig, sensor_poses) -> tuple[DepthFrame, ...]:
        """Sample each full pose against the same uploaded scene, in input order."""
        return tuple(self.sample(config, pose) for pose in sensor_poses)

    def camera(self, config: DepthConfig, sensor_pose: Pose, min_visible_fraction: float) -> CameraFrame:
        """Nearest-surface target mask and visibility-gated normalized half-open box.

        Denominator is target-only rasterization at exactly the same pose/config,
        within the camera's valid radial range, image and declared map coverage.
        A zero denominator or zero visible pixels always means no detection.
        This noiseless geometry baseline emits confidence 1 for detection.
        """
        min_visible_fraction = _finite_number(min_visible_fraction, "min_visible_fraction")
        if not 0 <= min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must be finite in [0,1]")
        frame = self.sample(config, sensor_pose)
        distance, identity = self._rasterize(config, sensor_pose, target_only=True)
        unoccluded = ((identity == self._target_index) &
                      (distance >= config.min_range_m) & (distance <= config.max_range_m))
        # Apply the same coverage policy as the complete image, without replacing
        # occluders by an assumed far distance or using their identities as input.
        known_target, _ = _coverage(self._snapshot.map_bounds_m, config, sensor_pose, distance)
        unoccluded &= known_target
        visible = (frame.object_index == self._target_index) & np.isfinite(frame.range_m)
        count = int(np.count_nonzero(unoccluded))
        fraction = float(np.count_nonzero(visible) / count) if count else 0.0
        hint = np.zeros(5, dtype="f4")
        if count and np.any(visible) and fraction >= min_visible_fraction:
            rows, cols = np.nonzero(visible)
            hint[:] = (1, cols.min() / config.width, rows.min() / config.height,
                       (cols.max() + 1) / config.width, (rows.max() + 1) / config.height)
        return CameraFrame(frame, _readonly(visible), _readonly(hint), fraction, count)


def sample_rig(sensor: MeshSensor, config: SensorConfig, body_pose: Pose) -> RigFrame:
    """External assembly: four calls in right/front/left/rear order + camera.

    All channels use the supplied complete body pose and one scene reference
    time. This creates synthetic simultaneous readings, not real-device timing.
    """
    if not isinstance(config, SensorConfig):
        raise TypeError("config must be a SensorConfig")
    frames = [sensor.sample(config.tof, mounted_pose(body_pose, mount)) for mount in config.tof_mounts]
    camera = sensor.camera(config.camera, mounted_pose(body_pose, config.camera_mount),
                           config.min_visible_fraction)
    return RigFrame(_readonly(np.stack([frame.range_m for frame in frames])),
                    camera.target_hint, camera, camera.depth.t_ref)
