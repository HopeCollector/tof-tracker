"""One GPU scene, complete FLU view poses, and owned CUDA raster batches.

The renderer provides geometry measurements only. Sensor range validity,
coverage, target recognition, and installation transforms belong to callers.
Public geometry remains in world coordinates: local sphere tessellation and
object transforms are private GPU storage choices.
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from functools import wraps
import hashlib
import math
import threading

import numpy as np
import torch

from .config import DepthConfig
from .mesh import MeshObject
from .scene import GeometrySnapshot, Sphere

from ._gl_cuda import CUDA, GL


def _on_cuda_device(method):
    """Interop uses CUDA's current device, independently of tensor devices."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._require_owner()
        with torch.cuda.device(self.device):
            return method(self, *args, **kwargs)
    return guarded


def _validate_render_config(config):
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    tangent = tuple(math.tan(math.radians(angle) / 2)
                    for angle in (config.hfov_deg, config.vfov_deg))
    if (config.max_range_m > float(np.finfo(np.float32).max) / 2
            or min(tangent) < float(np.finfo(np.float32).tiny)):
        raise ValueError("sensor range and field of view must be representable in float32")


@dataclass(frozen=True)
class RasterBatch:
    """Independent GPU measurements and the immutable scene they describe.

    ``distance_m`` is float32 ``[B,H,W]`` radial distance, with zero for a
    raster miss. ``object_index`` is int64 with -1 for a miss and otherwise
    indexes ``object_ids``. These are raw measurements, before sensor validity
    and coverage policy. Storage survives subsequent render/sync/close calls.
    """

    distance_m: torch.Tensor
    object_index: torch.Tensor
    object_ids: tuple[str, ...]
    scene_version: int
    t_ref: float | None
    map_bounds_m: tuple[tuple[float, float], ...] | None


def _sphere_mesh(radius_m):
    """Fixed 64x128 inscribed sphere tessellation in local metres.

    Sphere quaternion does not rotate this physical approximation; this keeps
    the same world-axis tessellation used by the original development sensor.
    """
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
    return np.asarray(points) * radius_m, np.asarray(faces)


_FRAGMENT_SHADER = """
#version 450 core
flat in vec4 plane_sensor;
flat in float identity;
flat in float render_radius;
uniform vec2 tangent_fov;
uniform vec2 image_size;
layout(location=0) out vec2 measurement;
void main() {
    vec2 pixel = 2.0 * gl_FragCoord.xy / image_size - 1.0;
    vec3 ray = vec3(1.0, -pixel.x * tangent_fov.x, pixel.y * tangent_fov.y);
    // Evaluate a flat triangle plane rather than clipped varying interpolation
    // near the eye plane. Depth ordering is radial on each pixel's fixed ray.
    float x = plane_sensor.w / dot(plane_sensor.xyz, ray);
    float r = x * length(ray);
    if (r < 0.0 || r > render_radius) discard;
    gl_FragDepth = r / render_radius;
    measurement = vec2(r, identity);
}
"""


_VERTEX_SHADER = """
#version 450 core
#extension GL_ARB_shader_viewport_layer_array : require
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in float object_id;
layout(std430, binding=0) readonly buffer PoseData { float pose_data[]; };
layout(std430, binding=1) readonly buffer ObjectData { mat4 object_to_world[]; };
uniform vec2 tangent_fov;
flat out vec4 plane_sensor;
flat out float identity;
flat out float render_radius;
void main() {
    int i = gl_InstanceID * 17;
    mat3 sensor_to_world = mat3(
        vec3(pose_data[i], pose_data[i+4], pose_data[i+8]),
        vec3(pose_data[i+1], pose_data[i+5], pose_data[i+9]),
        vec3(pose_data[i+2], pose_data[i+6], pose_data[i+10]));
    mat3 world_to_sensor = transpose(sensor_to_world);
    vec3 sensor_origin = vec3(pose_data[i+3], pose_data[i+7], pose_data[i+11]);
    mat4 object_pose = object_to_world[int(object_id) - 1];
    vec3 point_world = (object_pose * vec4(position, 1.0)).xyz;
    vec3 normal_world = mat3(object_pose) * normal;
    vec3 point_sensor = world_to_sensor * (point_world - sensor_origin);
    vec3 normal_sensor = world_to_sensor * normal_world;
    plane_sensor = vec4(normal_sensor, dot(normal_sensor, point_sensor));
    identity = object_id;
    render_radius = pose_data[i+16];
    gl_Layer = gl_InstanceID;
    gl_Position = vec4(-point_sensor.y / tangent_fov.x,
                        point_sensor.z / tangent_fov.y, 0.0, point_sensor.x);
}
"""
# OpenGL enumerants kept local to the small interop surface.
_ARRAY_BUFFER, _SSBO, _PBO = 0x8892, 0x90D2, 0x88EB
_STATIC_DRAW, _DYNAMIC_DRAW, _STREAM_READ = 0x88E4, 0x88E8, 0x88E1
_TEXTURE_ARRAY, _FRAMEBUFFER = 0x8C1A, 0x8D40


class BatchRenderer:
    """Thread-confined renderer with a shared scene and GPU input/output.

    ``render`` accepts finite rigid ``float32 CUDA [B,4,4]`` camera-FLU-to-world
    matrices.  It deliberately does not read GPU values onto the CPU to validate
    them.  Each return owns its CUDA storage and remains valid after later
    renders and after closing the renderer.  Calls use PyTorch's current stream;
    cross-stream input dependencies follow normal PyTorch caller obligations.

    Rendering is one instanced draw per full-scene or selected-object pass.
    No loop issues per-view GL work, and
    no per-view geometry copy is made.  Registration/map/unmap may synchronize
    graphics and CUDA queues; GPU residency does not imply zero synchronization.
    """

    def __init__(self, snapshot: GeometrySnapshot, device="cuda"):
        import moderngl

        if not isinstance(snapshot, GeometrySnapshot):
            raise TypeError("snapshot must be a GeometrySnapshot")
        if not snapshot.objects:
            raise ValueError("renderer requires at least one object")
        self._owner_thread = threading.get_ident()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("BatchRenderer requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._closed = False
        self._ctx = self._program = self._geometry = None
        self._gl = self._cuda = None
        self._vao = self._pose_buffer = self._object_buffer = 0
        self._pose_resource = None
        self._targets = {}
        self._batch_size = None
        self._pose_capacity = 0
        self._prepare_info = {"batch_size": None}
        self._draw_count = self._render_calls = self._geometry_uploads = 0
        self._transform_uploads = 0
        self._transform_updates = 0
        self._transform_upload_bytes = 0
        self._scene_version = 0
        self._object_ranges = {}
        self.object_ids = tuple(obj.object_id for obj in snapshot.objects)
        if len(self.object_ids) >= 2**24:
            raise ValueError("too many objects for exact float32 diagnostic IDs")
        self._snapshot = snapshot
        object_transforms = self._transforms(snapshot)
        previous_device = torch.cuda.current_device()
        torch.cuda.set_device(self.device)
        try:
            self._ctx = moderngl.create_context(standalone=True, backend="egl", require=450)
            self.renderer_info = {name: self._ctx.info[name] for name in
                                  ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
            renderer = self.renderer_info["GL_RENDERER"].lower()
            if any(tag in renderer for tag in ("llvmpipe", "softpipe", "swrast", "software",
                                               "lavapipe", "swiftshader")):
                raise RuntimeError(f"GPU renderer required; got {renderer}")
            if "GL_ARB_shader_viewport_layer_array" not in self._ctx.extensions:
                raise RuntimeError("GL_ARB_shader_viewport_layer_array is required")
            self._gl, self._cuda = GL(), CUDA()
            self._program = self._ctx.program(vertex_shader=_VERTEX_SHADER,
                                               fragment_shader=_FRAGMENT_SHADER)
            blocks, offset = [], 0
            for index, obj in enumerate(snapshot.objects):
                shape = obj.geometry
                if isinstance(shape, MeshObject):
                    vertices, faces = np.asarray(shape.vertices_m), np.asarray(shape.triangles)
                elif isinstance(shape, Sphere):
                    vertices, faces = _sphere_mesh(shape.radius_m)
                else:
                    raise TypeError(f"unsupported public geometry: {type(shape).__name__}")
                triangles = vertices[faces]
                normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                triangles = triangles.reshape(-1, 3)
                block = np.column_stack((triangles, np.repeat(normals, 3, axis=0),
                                         np.full(len(triangles), index + 1))).astype("f4")
                if not np.all(np.isfinite(block)):
                    raise ValueError("scene geometry must be representable in float32")
                blocks.append(block)
                self._object_ranges[obj.object_id] = (offset, len(block))
                offset += len(block)
            packed_geometry = np.concatenate(blocks)
            self._geometry = self._ctx.buffer(packed_geometry.tobytes())
            self._geometry_uploads = 1
            self.triangle_count = offset // 3
            self.vertex_count = offset
            gl = self._gl
            self._object_buffer = gl.create("Buffers")
            gl.glBindBuffer(_SSBO, self._object_buffer)
            gl.glBufferData(_SSBO, object_transforms.nbytes,
                            object_transforms.ctypes.data_as(C.c_void_p), _DYNAMIC_DRAW)
            self._transform_uploads = 1
            self._transform_upload_bytes = object_transforms.nbytes
            self._object_transforms = object_transforms
            self._vao = gl.create("VertexArrays")
            gl.glBindVertexArray(self._vao)
            gl.glBindBuffer(_ARRAY_BUFFER, self._geometry.glo)
            for index, width, byte_offset in ((0, 3, 0), (1, 3, 12), (2, 1, 24)):
                gl.glEnableVertexAttribArray(index)
                gl.glVertexAttribPointer(index, width, 0x1406, 0, 28, C.c_void_p(byte_offset))
            self._uniforms = {name: gl.glGetUniformLocation(self._program.glo, name.encode())
                              for name in ("tangent_fov", "image_size")}
            self._max_layers = int(self._ctx.info["GL_MAX_ARRAY_TEXTURE_LAYERS"])
            gl.glEnable(0x0B71)  # depth test
            gl.glDepthFunc(0x0201)  # less
            gl.glDepthMask(1)
            for flag in (0x0BE2, 0x0B44, 0x0C11, 0x8037, 0x809D):
                gl.glDisable(flag)  # blend, cull, scissor, polygon offset, multisample
            gl.glPixelStorei(0x0D05, 1)  # PACK_ALIGNMENT
            gl.glPixelStorei(0x0D02, 0)  # PACK_ROW_LENGTH
            gl.glPixelStorei(0x806C, 0)  # PACK_IMAGE_HEIGHT
            gl.check("single-scene setup")
        except Exception:
            self.close()
            raise
        finally:
            torch.cuda.set_device(previous_device)

    @property
    def geometry_info(self):
        return {
            "backend": "opengl_layered_instanced_cuda_pbo",
            "device": str(self.device), "renderer": dict(self.renderer_info),
            "object_ids": list(self.object_ids),
            "triangle_count": self.triangle_count, "vertex_count": self.vertex_count,
            "scene_geometry_buffer_name": self._geometry.glo if self._geometry else None,
            "scene_geometry_buffer_bytes": self._geometry.size if self._geometry else 0,
            "geometry_uploads": self._geometry_uploads,
            "object_transform_uploads": self._transform_uploads,
            "object_transform_updates": self._transform_updates,
            "object_transform_upload_bytes": self._transform_upload_bytes,
            "scene_version": self._scene_version,
            "batch_geometry_policy": "one VBO; world meshes and local spheres; independent of B",
            "full_pass_instanced_draws": 1, "selected_object_pass_instanced_draws": 1,
            "instanced_draw_count": self._draw_count, "render_calls": self._render_calls,
            "max_array_texture_layers": self._max_layers,
            "input_pose": "float32 CUDA B44; row-major camera FLU to world",
            "output": "owned CUDA distance float32 and identity int64 BHW",
            "output_transfer": "whole array to PBO in GL; one CUDA D2D PBO-to-Torch copy",
            "cpu_image_readbacks": 0, "per_view_draw_calls": 0,
            "interop_synchronization": "CUDA GL-resource map/unmap on current PyTorch stream",
            "vertex_shader_sha256": hashlib.sha256(_VERTEX_SHADER.encode()).hexdigest(),
            "fragment_shader_sha256": hashlib.sha256(_FRAGMENT_SHADER.encode()).hexdigest(),
            "near_clip_policy": "unchanged GL w=forward, clip z=0, no positive near plane",
            "depth_component_bits": 24,
        }

    @property
    def memory_info(self):
        geometry = self._geometry.size if self._geometry else 0
        transforms = len(self.object_ids) * 64 if self._object_buffer else 0
        pose = self._pose_capacity * 17 * 4
        color = sum(item["pixels"] * 8 for item in self._targets.values())
        depth = sum(item["pixels"] * 4 for item in self._targets.values())
        pbo = sum(item["pixels"] * 8 for item in self._targets.values())
        return {
            "scene_geometry_buffer_bytes": geometry, "pose_ssbo_bytes": pose,
            "object_transform_ssbo_bytes": transforms,
            "layered_color_texture_bytes": color,
            "layered_depth_texture_bytes_estimate": depth,
            "output_pbo_bytes": pbo,
            "tracked_total_bytes": geometry + transforms + pose + color + depth + pbo,
            "tracked_total_scope": "explicit GL storage; depth24 estimated as 4 bytes/texel; "
                                   "excludes driver padding/objects and all Torch allocations",
            "input_pose_transfer_bytes_per_render": (self._batch_size or 0) * 17 * 4,
            "output_cuda_transfer_bytes_per_render": (
                self._last_output_bytes if hasattr(self, "_last_output_bytes") else None),
            "image_cpu_transfer_bytes_per_render": 0,
            "texture_array_count": len(self._targets),
        }

    @property
    def prepare_info(self):
        return dict(self._prepare_info)

    @property
    def scene_version(self):
        return self._scene_version

    @staticmethod
    def _transforms(snapshot):
        # std430 mat4[] is column-major; transpose each host row-major matrix.
        transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(snapshot.objects), axis=0)
        for index, obj in enumerate(snapshot.objects):
            if isinstance(obj.geometry, Sphere):
                transforms[index, :3, 3] = obj.geometry.pose.position_m
        with np.errstate(over="ignore", invalid="ignore"):
            packed = np.ascontiguousarray(transforms.transpose(0, 2, 1), dtype=np.float32)
        if not np.all(np.isfinite(packed)):
            raise ValueError("object transforms must be representable in float32")
        return packed

    @_on_cuda_device
    def sync(self, snapshot: GeometrySnapshot) -> int:
        """Synchronize world facts between batches, without uploading geometry.

        Object order/identity, mesh references, and sphere radii must remain
        unchanged. Unsupported geometry changes are rejected before GPU writes;
        construct a new renderer for them. Sphere center changes upload only
        their object matrix. Quaternion-only or metadata changes update the
        scene version without uploading unchanged transforms.
        """
        self._require_open()
        if not isinstance(snapshot, GeometrySnapshot):
            raise TypeError("snapshot must be a GeometrySnapshot")
        if len(snapshot.objects) != len(self._snapshot.objects):
            raise ValueError("geometry structure changed; rebuild BatchRenderer")
        changed = (snapshot.t_ref != self._snapshot.t_ref
                   or snapshot.map_bounds_m != self._snapshot.map_bounds_m)
        for old, new in zip(self._snapshot.objects, snapshot.objects):
            if (old.object_id != new.object_id or old.role != new.role
                    or type(old.geometry) is not type(new.geometry)):
                raise ValueError("object identity/type/order changed; rebuild BatchRenderer")
            if isinstance(old.geometry, MeshObject):
                if old.geometry is not new.geometry:
                    raise ValueError("mesh geometry changed; rebuild BatchRenderer")
            elif isinstance(old.geometry, Sphere):
                if old.geometry.radius_m != new.geometry.radius_m:
                    raise ValueError("sphere radius changed; rebuild BatchRenderer")
                changed |= old.geometry.pose != new.geometry.pose
        transforms = self._transforms(snapshot)
        indices = np.flatnonzero(np.any(transforms != self._object_transforms, axis=(1, 2)))
        if indices.size:
            self._gl.activate()
            self._gl.glBindBuffer(_SSBO, self._object_buffer)
            try:
                for index in indices:
                    matrix = transforms[index]
                    self._gl.glBufferSubData(_SSBO, int(index) * 64, 64,
                                            matrix.ctypes.data_as(C.c_void_p))
                self._gl.check("object transform update")
            except Exception:
                # A driver failure must not leave a half-updated usable scene.
                self.close()
                raise
            self._transform_uploads += len(indices)
            self._transform_updates += len(indices)
            self._transform_upload_bytes += len(indices) * 64
            self._object_transforms = transforms
        if changed:
            self._scene_version += 1
        self._snapshot = snapshot
        return self._scene_version

    @_on_cuda_device
    def prepare(self, batch_size, *, config=None):
        """Warm the pose buffer and optional layered output for one batch.

        Pose capacity only grows. Outputs cache by B/H/W, so alternating a
        four-camera ToF group and a single-camera group reuses storage. Call
        ``release_batch`` to release all such caches without touching geometry.
        """
        self._require_open()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > self._max_layers:
            raise ValueError(f"batch size {batch_size} exceeds GL array-layer limit {self._max_layers}")
        if config is not None:
            _validate_render_config(config)
        if batch_size > self._pose_capacity:
            gl = self._gl
            gl.activate()
            if self._pose_resource:
                torch.cuda.synchronize(self.device)
                self._cuda.unregister(self._pose_resource)
                self._pose_resource = None
                gl.delete("Buffers", self._pose_buffer)
                self._pose_buffer = 0
            try:
                self._pose_buffer = gl.create("Buffers")
                gl.glBindBuffer(_SSBO, self._pose_buffer)
                gl.glBufferData(_SSBO, batch_size * 17 * 4, None, _DYNAMIC_DRAW)
                self._pose_resource = self._cuda.register(self._pose_buffer, 2)  # write-discard
                gl.check("pose SSBO preparation")
            except Exception:
                self.close()
                raise
            self._pose_capacity = batch_size
        self._batch_size = batch_size
        if config is not None:
            self._target(config)
        self._prepare_info = {
            "batch_size": batch_size,
            "batch_raw_mesh_arrays_bytes": 0,
            "scene_geometry_buffer_name": self._geometry.glo,
            "scene_geometry_buffer_bytes": self._geometry.size,
            "geometry_uploads": self._geometry_uploads,
            "pose_ssbo_bytes": self._pose_capacity * 17 * 4,
            "selected_object_geometry_policy": "same VBO range, no geometry copy",
        }
        return self.prepare_info

    def _target(self, config):
        _validate_render_config(config)
        key = (self._batch_size, config.width, config.height)
        if key in self._targets:
            return self._targets[key]
        gl = self._gl
        gl.activate()
        width, height, batch = config.width, config.height, self._batch_size
        byte_count = batch * width * height * 8
        if byte_count > 2**31 - 1:
            raise ValueError("single glGetTextureImage transfer exceeds GLsizei range")
        item = {"color": 0, "depth": 0, "fbo": 0, "pbo": 0, "resource": None,
                "pixels": batch * width * height, "bytes": byte_count}
        # Save before allocation, so an allocation failure can release every
        # resource and close this renderer instead of caching a partial target.
        self._targets[key] = item
        try:
            for name, internal_format in (("color", 0x8230), ("depth", 0x81A6)):  # RG32F, DEPTH24
                item[name] = gl.create("Textures")
                gl.glBindTexture(_TEXTURE_ARRAY, item[name])
                gl.glTexStorage3D(_TEXTURE_ARRAY, 1, internal_format, width, height, batch)
                gl.glTexParameteri(_TEXTURE_ARRAY, 0x2801, 0x2600)
                gl.glTexParameteri(_TEXTURE_ARRAY, 0x2800, 0x2600)
            item["fbo"] = gl.create("Framebuffers")
            gl.glBindFramebuffer(_FRAMEBUFFER, item["fbo"])
            gl.glFramebufferTexture(_FRAMEBUFFER, 0x8CE0, item["color"], 0)
            gl.glFramebufferTexture(_FRAMEBUFFER, 0x8D00, item["depth"], 0)
            attachment = (C.c_uint * 1)(0x8CE0)
            gl.glDrawBuffers(1, attachment)
            status = gl.glCheckFramebufferStatus(_FRAMEBUFFER)
            if status != 0x8CD5:
                raise RuntimeError(f"Layered framebuffer incomplete: 0x{status:04x}")
            item["pbo"] = gl.create("Buffers")
            gl.glBindBuffer(_PBO, item["pbo"])
            gl.glBufferData(_PBO, byte_count, None, _STREAM_READ)
            gl.glBindBuffer(_PBO, 0)
            item["resource"] = self._cuda.register(item["pbo"], 1)  # read-only from CUDA
            gl.check("layered framebuffer/PBO preparation")
        except Exception:
            self.close()
            raise
        return item

    @torch.no_grad()
    @_on_cuda_device
    def render(self, poses_cuda, config, only_object: str | None = None) -> RasterBatch:
        self._require_open()
        _validate_render_config(config)
        if (not isinstance(poses_cuda, torch.Tensor) or poses_cuda.device != self.device
                or poses_cuda.dtype != torch.float32 or poses_cuda.ndim != 3
                or tuple(poses_cuda.shape[1:]) != (4, 4)):
            raise ValueError("poses must be float32 CUDA [B,4,4] on renderer.device")
        batch = len(poses_cuda)
        if only_object is not None and (not isinstance(only_object, str)
                                        or only_object not in self._object_ranges):
            raise ValueError(f"unknown object ID {only_object!r}")
        if batch != self._batch_size:
            self.prepare(batch)
        target = self._target(config)
        gl, cuda = self._gl, self._cuda
        gl.activate()
        stream = torch.cuda.current_stream(self.device).cuda_stream

        # Preserve the original host float64 guard computation, after the
        # renderer's explicitly float32 pose-input quantization.
        maximum = poses_cuda[:, :3, 3].to(torch.float64).abs().amax(dim=1)
        maximum = maximum.clamp_min(max(1., config.max_range_m))
        roundoff = 32 * np.finfo(np.float32).eps * maximum
        guard = torch.maximum(roundoff * 4, torch.full_like(roundoff, config.max_range_m * .001))
        radius = (config.max_range_m + guard).to(torch.float32)
        packed = torch.cat((poses_cuda.reshape(batch, 16), radius[:, None]), dim=1).contiguous()
        pointer, size = cuda.map(self._pose_resource, stream)
        try:
            if size < packed.numel() * packed.element_size():
                raise RuntimeError("mapped pose buffer is smaller than the input batch")
            cuda.copy(pointer, packed.data_ptr(), packed.numel() * packed.element_size(), stream)
        finally:
            cuda.unmap(self._pose_resource, stream)

        gl.glBindFramebuffer(_FRAMEBUFFER, target["fbo"])
        gl.glClearBufferfv(0x1800, 0, (C.c_float * 4)(0, 0, 0, 0))
        gl.glClearBufferfv(0x1801, 0, (C.c_float * 1)(1))
        gl.glViewport(0, 0, config.width, config.height)
        gl.glUseProgram(self._program.glo)
        gl.glBindVertexArray(self._vao)
        gl.glBindBufferBase(_SSBO, 0, self._pose_buffer)
        gl.glBindBufferBase(_SSBO, 1, self._object_buffer)
        gl.glUniform2f(self._uniforms["tangent_fov"],
                      math.tan(math.radians(config.hfov_deg) / 2),
                      math.tan(math.radians(config.vfov_deg) / 2))
        gl.glUniform2f(self._uniforms["image_size"], config.width, config.height)
        first, count = self._object_ranges[only_object] if only_object else (0, self.vertex_count)
        gl.glDrawArraysInstanced(0x0004, first, count, batch)
        self._draw_count += 1
        self._render_calls += 1

        # One whole-array graphics transfer, followed by one whole-array CUDA
        # copy.  The pointer argument is offset zero in the bound PBO, never a
        # CPU address.  Map waits for preceding GL writes on this resource.
        gl.glBindBuffer(_PBO, target["pbo"])
        gl.glGetTextureImage(target["color"], 0, 0x8227, 0x1406,
                             target["bytes"], C.c_void_p(0))
        gl.glBindBuffer(_PBO, 0)
        gl.glFlush()
        raw = torch.empty((batch, config.height, config.width, 2),
                          dtype=torch.float32, device=self.device)
        pointer, size = cuda.map(target["resource"], stream)
        try:
            if size < target["bytes"]:
                raise RuntimeError("mapped output PBO is smaller than the layered image")
            cuda.copy(raw.data_ptr(), pointer, target["bytes"], stream)
        finally:
            cuda.unmap(target["resource"], stream)
        gl.check("instanced draw and whole-array GPU transfer")
        self._last_output_bytes = target["bytes"]
        distance = raw[..., 0].flip((1,)).contiguous()
        identity = raw[..., 1].flip((1,)).to(torch.int64) - 1
        return RasterBatch(distance, identity, self.object_ids, self._scene_version,
                           self._snapshot.t_ref, self._snapshot.map_bounds_m)

    @_on_cuda_device
    def release_batch(self):
        self._require_owner()
        if not self._gl or self._ctx is None:
            return
        self._gl.activate()
        if self._pose_resource or self._targets:
            torch.cuda.synchronize(self.device)
        for item in self._targets.values():
            self._cuda.unregister(item["resource"])
            self._gl.delete("Buffers", item["pbo"])
            self._gl.delete("Framebuffers", item["fbo"])
            self._gl.delete("Textures", item["depth"])
            self._gl.delete("Textures", item["color"])
        self._targets.clear()
        self._cuda.unregister(self._pose_resource)
        self._pose_resource = None
        self._gl.delete("Buffers", self._pose_buffer)
        self._pose_buffer = 0
        self._batch_size = None
        self._pose_capacity = 0
        self._prepare_info = {"batch_size": None}

    def _require_open(self):
        self._require_owner()
        if self._closed:
            raise RuntimeError("BatchRenderer is closed")

    def _require_owner(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("BatchRenderer must be used and closed on its creating thread")

    @_on_cuda_device
    def close(self):
        self._require_owner()
        if self._closed:
            return
        if self._gl:
            self.release_batch()
            self._gl.delete("Buffers", self._object_buffer)
            self._object_buffer = 0
            self._gl.delete("VertexArrays", self._vao)
            self._vao = 0
        if self._geometry:
            self._geometry.release()
            self._geometry = None
        if self._program:
            self._program.release()
            self._program = None
        if self._ctx:
            self._ctx.release()
            self._ctx = None
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()
