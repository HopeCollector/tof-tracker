ARG BASE_IMAGE=tof-tracker-pytorch3d-eval:torch2.7-cu128
FROM ${BASE_IMAGE}
# The base already provides Torch, ModernGL, libEGL, and NVML Python bindings.
# CUDA/GL interop uses ctypes and the NVIDIA driver supplied by --gpus all.
ENV PYTHONPYCACHEPREFIX=/tmp/opengl-batch-python-cache \
    XDG_CACHE_HOME=/tmp/opengl-batch-cache \
    MPLCONFIGDIR=/tmp/opengl-batch-matplotlib-cache
