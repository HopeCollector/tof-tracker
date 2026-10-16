"""Small ctypes surface for this experiment's OpenGL/CUDA interoperability.

All entry points are supplied by the container's NVIDIA OpenGL/CUDA runtime.
This module compiles no extension and contains no CPU image readback path.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
from pathlib import Path
import sysconfig


def _library(names, extra=()):
    errors = []
    for name in [*(ctypes.util.find_library(n) for n in names), *names, *extra]:
        if not name:
            continue
        try:
            return C.CDLL(str(name))
        except OSError as exc:
            errors.append(str(exc))
    raise RuntimeError("Could not load runtime library: " + "; ".join(errors))


def _function(library, name, restype, *argtypes):
    function = getattr(library, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


class GL:
    """Explicit typed entry points, with an EGL-context restoration handle."""

    def __init__(self):
        library = _library(["OpenGL", "libOpenGL.so.0", "GL", "libGL.so.1"])
        u, i, p, z, f = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_float
        specifications = {
            "glGetError": (u,),
            "glGenBuffers": (None, i, C.POINTER(u)),
            "glDeleteBuffers": (None, i, C.POINTER(u)),
            "glBindBuffer": (None, u, u),
            "glBufferData": (None, u, z, p, u),
            "glBindBufferBase": (None, u, u, u),
            "glGenVertexArrays": (None, i, C.POINTER(u)),
            "glDeleteVertexArrays": (None, i, C.POINTER(u)),
            "glBindVertexArray": (None, u),
            "glEnableVertexAttribArray": (None, u),
            "glVertexAttribPointer": (None, u, i, u, C.c_ubyte, i, p),
            "glGenTextures": (None, i, C.POINTER(u)),
            "glDeleteTextures": (None, i, C.POINTER(u)),
            "glBindTexture": (None, u, u),
            "glTexStorage3D": (None, u, i, u, i, i, i),
            "glTexParameteri": (None, u, u, i),
            "glGetTextureImage": (None, u, i, u, u, i, p),
            "glGenFramebuffers": (None, i, C.POINTER(u)),
            "glDeleteFramebuffers": (None, i, C.POINTER(u)),
            "glBindFramebuffer": (None, u, u),
            "glFramebufferTexture": (None, u, u, u, i),
            "glDrawBuffers": (None, i, C.POINTER(u)),
            "glCheckFramebufferStatus": (u, u),
            "glClearBufferfv": (None, u, i, C.POINTER(f)),
            "glUseProgram": (None, u),
            "glGetUniformLocation": (i, u, C.c_char_p),
            "glUniform1f": (None, i, f),
            "glUniform2f": (None, i, f, f),
            "glViewport": (None, i, i, i, i),
            "glEnable": (None, u),
            "glDisable": (None, u),
            "glDepthFunc": (None, u),
            "glDepthMask": (None, C.c_ubyte),
            "glPixelStorei": (None, u, i),
            "glDrawArraysInstanced": (None, u, i, i, i),
            "glFlush": (None,),
        }
        for name, (restype, *argtypes) in specifications.items():
            setattr(self, name, _function(library, name, restype, *argtypes))
        egl = _library(["EGL", "libEGL.so.1"])
        self._get_display = _function(egl, "eglGetCurrentDisplay", p)
        self._get_context = _function(egl, "eglGetCurrentContext", p)
        self._get_surface = _function(egl, "eglGetCurrentSurface", p, u)
        self._make_current = _function(egl, "eglMakeCurrent", u, p, p, p, p)
        self._get_egl_error = _function(egl, "eglGetError", u)
        self._display = self._get_display()
        self._context = self._get_context()
        self._draw_surface = self._get_surface(0x3059)
        self._read_surface = self._get_surface(0x305A)
        if not self._display or not self._context:
            raise RuntimeError("The experiment requires a current EGL context")

    def activate(self):
        if self._get_context() != self._context:
            if not self._make_current(self._display, self._draw_surface,
                                      self._read_surface, self._context):
                raise RuntimeError(f"eglMakeCurrent failed: 0x{self._get_egl_error():04x}")

    def check(self, operation):
        error = self.glGetError()
        if error:
            raise RuntimeError(f"OpenGL error 0x{error:04x} after {operation}")

    def create(self, kind):
        result = C.c_uint()
        getattr(self, "glGen" + kind)(1, C.byref(result))
        return result.value

    def delete(self, kind, value):
        if value:
            name = C.c_uint(value)
            getattr(self, "glDelete" + kind)(1, C.byref(name))


class CUDA:
    """CUDA runtime GL-buffer registration; all copies stay device-to-device."""

    def __init__(self):
        paths = list(Path("/usr/local/cuda/lib64").glob("libcudart.so*"))
        runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/lib"
        paths.extend(runtime.glob("libcudart.so*"))
        library = _library(["cudart", "libcudart.so.12", "libcudart.so"], paths)
        p, u, i, z = C.c_void_p, C.c_uint, C.c_int, C.c_size_t
        self._error_string = _function(library, "cudaGetErrorString", C.c_char_p, i)
        self._register = _function(library, "cudaGraphicsGLRegisterBuffer", i,
                                   C.POINTER(p), u, u)
        self._unregister = _function(library, "cudaGraphicsUnregisterResource", i, p)
        self._map = _function(library, "cudaGraphicsMapResources", i, i, C.POINTER(p), p)
        self._unmap = _function(library, "cudaGraphicsUnmapResources", i, i, C.POINTER(p), p)
        self._pointer = _function(library, "cudaGraphicsResourceGetMappedPointer", i,
                                 C.POINTER(p), C.POINTER(z), p)
        self._copy = _function(library, "cudaMemcpyAsync", i, p, p, z, i, p)

    def check(self, status, operation):
        if status:
            detail = self._error_string(status)
            raise RuntimeError(f"{operation} failed ({status}): "
                               + (detail.decode() if detail else "unknown CUDA error"))

    def register(self, buffer, flags):
        resource = C.c_void_p()
        self.check(self._register(C.byref(resource), buffer, flags),
                   "cudaGraphicsGLRegisterBuffer")
        return resource

    def unregister(self, resource):
        if resource:
            self.check(self._unregister(resource), "cudaGraphicsUnregisterResource")

    def map(self, resource, stream):
        array = (C.c_void_p * 1)(resource.value)
        self.check(self._map(1, array, stream), "cudaGraphicsMapResources")
        pointer, size = C.c_void_p(), C.c_size_t()
        try:
            self.check(self._pointer(C.byref(pointer), C.byref(size), resource),
                       "cudaGraphicsResourceGetMappedPointer")
        except Exception:
            self.unmap(resource, stream)
            raise
        return pointer.value, size.value

    def unmap(self, resource, stream):
        array = (C.c_void_p * 1)(resource.value)
        self.check(self._unmap(1, array, stream), "cudaGraphicsUnmapResources")

    def copy(self, destination, source, size, stream):
        self.check(self._copy(destination, source, size, 3, stream),
                   "cudaMemcpyAsync(device_to_device)")
