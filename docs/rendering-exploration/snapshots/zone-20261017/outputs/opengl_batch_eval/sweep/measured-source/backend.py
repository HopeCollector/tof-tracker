"""Experimental layered OpenGL rasterization with GPU-only CUDA outputs.

One immutable public scene is uploaded once.  A CUDA batch of complete FLU
camera-to-world matrices is copied into one SSBO.  An instanced vertex shader
routes every view into its own texture-array layer in one draw.  A single
texture-array transfer into a CUDA-registered PBO and one device copy produce
owned PyTorch tensors; image pixels never pass through host memory.

The triangle layout, projection, plane-distance fragment calculation, radial
depth ordering, and guard shell are adapted from ``devtools.sensors``.  This
experiment does not change or replace that module.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import math

import numpy as np
import torch

from devtools.config import DepthConfig
from devtools.mesh import MeshObject
from devtools.scene import Sphere
from devtools.sensors import _FRAGMENT_SHADER, _sphere_mesh

from experiments.opengl_batch_eval.gl_api import CUDA, GL


_VERTEX_SHADER = """
#version 450 core
#extension GL_ARB_shader_viewport_layer_array : require
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in float object_id;
layout(std430, binding=0) readonly buffer PoseData { float pose_data[]; };
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
    vec3 point_sensor = world_to_sensor * (position - sensor_origin);
    vec3 normal_sensor = world_to_sensor * normal;
    plane_sensor = vec4(normal_sensor, dot(normal_sensor, point_sensor));
    identity = object_id;
    render_radius = pose_data[i+16];
    gl_Layer = gl_InstanceID;
    gl_Position = vec4(-point_sensor.y / tangent_fov.x,
                        point_sensor.z / tangent_fov.y, 0.0, point_sensor.x);
}
"""
_BATCH_FRAGMENT_SHADER = _FRAGMENT_SHADER.replace(
    "#version 330", "#version 450 core").replace(
    "uniform float render_radius;", "flat in float render_radius;")

# OpenGL enumerants kept local to the deliberately small experiment surface.
_ARRAY_BUFFER, _SSBO, _PBO = 0x8892, 0x90D2, 0x88EB
_STATIC_DRAW, _DYNAMIC_DRAW, _STREAM_READ = 0x88E4, 0x88E8, 0x88E1
_TEXTURE_ARRAY, _FRAMEBUFFER = 0x8C1A, 0x8D40


class GLBatchRenderer:
    """Single-threaded, immutable-scene, GPU-input/GPU-output batch renderer.

    ``render`` accepts finite rigid ``float32 CUDA [B,4,4]`` camera-FLU-to-world
    matrices.  It deliberately does not read GPU values onto the CPU to validate
    them.  Each return owns its CUDA storage and remains valid after later
    renders and after closing the renderer.  Calls use PyTorch's current stream;
    cross-stream input dependencies follow normal PyTorch caller obligations.

    Rendering is one instanced draw per full-scene or target-only pass.  Camera
    visibility requires those two passes.  No loop issues per-view GL work, and
    no per-view geometry copy is made.  Registration/map/unmap may synchronize
    graphics and CUDA queues; GPU residency does not imply zero synchronization.
    """

    def __init__(self, scene, device="cuda"):
        import moderngl

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the experiment requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        self._closed = False
        self._ctx = self._program = self._geometry = None
        self._gl = self._cuda = None
        self._vao = self._pose_buffer = 0
        self._pose_resource = None
        self._targets = {}
        self._batch_size = None
        self._prepare_info = {"batch_size": None}
        self._draw_count = self._render_calls = self._geometry_uploads = 0
        snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        self.object_ids = tuple(obj.object_id for obj in snapshot.objects)
        if len(self.object_ids) >= 2**24:
            raise ValueError("too many objects for exact float32 diagnostic IDs")
        self._target_index = self.object_ids.index("target")
        self._snapshot = snapshot
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
                                               fragment_shader=_BATCH_FRAGMENT_SHADER)
            blocks, offset = [], 0
            for index, obj in enumerate(snapshot.objects):
                shape = obj.geometry
                if isinstance(shape, MeshObject):
                    vertices, faces = np.asarray(shape.vertices_m), np.asarray(shape.triangles)
                elif isinstance(shape, Sphere):
                    vertices, faces = _sphere_mesh(shape)
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
                if index == self._target_index:
                    self._target_vertices = (offset, len(block))
                offset += len(block)
            packed_geometry = np.concatenate(blocks)
            self._geometry = self._ctx.buffer(packed_geometry.tobytes())
            self._geometry_uploads = 1
            self.triangle_count = offset // 3
            self.vertex_count = offset
            gl = self._gl
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
            "batch_geometry_policy": "one world-space VBO, independent of B",
            "full_pass_instanced_draws": 1, "target_only_pass_instanced_draws": 1,
            "instanced_draw_count": self._draw_count, "render_calls": self._render_calls,
            "max_array_texture_layers": self._max_layers,
            "input_pose": "float32 CUDA B44; row-major camera FLU to world",
            "output": "owned CUDA distance float32 and identity int64 BHW",
            "output_transfer": "whole array to PBO in GL; one CUDA D2D PBO-to-Torch copy",
            "cpu_image_readbacks": 0, "per_view_draw_calls": 0,
            "interop_synchronization": "CUDA GL-resource map/unmap on current PyTorch stream",
            "vertex_shader_sha256": hashlib.sha256(_VERTEX_SHADER.encode()).hexdigest(),
            "fragment_shader_sha256": hashlib.sha256(_BATCH_FRAGMENT_SHADER.encode()).hexdigest(),
            "source_fragment_shader_sha256": hashlib.sha256(_FRAGMENT_SHADER.encode()).hexdigest(),
            "near_clip_policy": "unchanged GL w=forward, clip z=0, no positive near plane",
            "depth_component_bits": 24,
        }

    @property
    def memory_info(self):
        geometry = self._geometry.size if self._geometry else 0
        pose = self._batch_size * 17 * 4 if self._batch_size else 0
        color = sum(item["pixels"] * 8 for item in self._targets.values())
        depth = sum(item["pixels"] * 4 for item in self._targets.values())
        pbo = sum(item["pixels"] * 8 for item in self._targets.values())
        return {
            "scene_geometry_buffer_bytes": geometry, "pose_ssbo_bytes": pose,
            "layered_color_texture_bytes": color,
            "layered_depth_texture_bytes_estimate": depth,
            "output_pbo_bytes": pbo,
            "tracked_total_bytes": geometry + pose + color + depth + pbo,
            "tracked_total_scope": "explicit GL storage; depth24 estimated as 4 bytes/texel; "
                                   "excludes driver padding/objects and all Torch allocations",
            "input_pose_transfer_bytes_per_render": pose,
            "output_cuda_transfer_bytes_per_render": (
                self._last_output_bytes if hasattr(self, "_last_output_bytes") else None),
            "image_cpu_transfer_bytes_per_render": 0,
            "texture_array_count": len(self._targets),
        }

    @property
    def prepare_info(self):
        return dict(self._prepare_info)

    def prepare(self, batch_size, *, include_target=True, config=None):
        self._require_open()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > self._max_layers:
            raise ValueError(f"batch size {batch_size} exceeds GL array-layer limit {self._max_layers}")
        if self._batch_size != batch_size:
            self.release_batch()
            gl = self._gl
            gl.activate()
            self._pose_buffer = gl.create("Buffers")
            gl.glBindBuffer(_SSBO, self._pose_buffer)
            gl.glBufferData(_SSBO, batch_size * 17 * 4, None, _DYNAMIC_DRAW)
            self._pose_resource = self._cuda.register(self._pose_buffer, 2)  # write-discard
            self._batch_size = batch_size
            gl.check("pose SSBO preparation")
        if config is not None:
            self._target(config)
        self._prepare_info = {
            "batch_size": batch_size, "include_target": include_target,
            "batch_raw_mesh_arrays_bytes": 0,
            "scene_geometry_buffer_name": self._geometry.glo,
            "scene_geometry_buffer_bytes": self._geometry.size,
            "geometry_uploads": self._geometry_uploads,
            "pose_ssbo_bytes": batch_size * 17 * 4,
            "target_geometry_policy": "same VBO range, no target copy",
        }
        return self.prepare_info

    def _target(self, config):
        if not isinstance(config, DepthConfig):
            raise TypeError("config must be a DepthConfig")
        key = (config.width, config.height)
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
        # Save before allocation, so close() can release partially built targets.
        self._targets[key] = item
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
        return item

    @torch.no_grad()
    def render(self, poses_cuda, config, target_only=False):
        self._require_open()
        if (not isinstance(poses_cuda, torch.Tensor) or poses_cuda.device != self.device
                or poses_cuda.dtype != torch.float32 or poses_cuda.ndim != 3
                or tuple(poses_cuda.shape[1:]) != (4, 4)):
            raise ValueError("poses must be float32 CUDA [B,4,4] on renderer.device")
        batch = len(poses_cuda)
        if batch != self._batch_size:
            raise ValueError("call prepare(B) before render")
        target = self._target(config)
        gl, cuda = self._gl, self._cuda
        gl.activate()
        stream = torch.cuda.current_stream(self.device).cuda_stream

        # Match the current host double-precision guard computation, after the
        # experiment's explicitly float32 pose-input quantization.
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
        gl.glUniform2f(self._uniforms["tangent_fov"],
                      math.tan(math.radians(config.hfov_deg) / 2),
                      math.tan(math.radians(config.vfov_deg) / 2))
        gl.glUniform2f(self._uniforms["image_size"], config.width, config.height)
        first, count = self._target_vertices if target_only else (0, self.vertex_count)
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
        return distance, identity

    def release_batch(self):
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
        self._prepare_info = {"batch_size": None}

    def _require_open(self):
        if self._closed:
            raise RuntimeError("GLBatchRenderer is closed")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._gl:
            self.release_batch()
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

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()
