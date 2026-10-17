# Isolated experiment image. The existing CUDA-devel project image supplies
# Python 3.10, Torch 2.7.1+cu128 and nvcc 12.8; no host packages are installed.
ARG BASE_IMAGE=tof-tracker-mesh-eval:warp1.15-cu128
FROM ${BASE_IMAGE}
ARG PYTORCH3D_REV=88e182f989c80836f4bd744e0d9cb1852762ce01
ENV TORCH_CUDA_ARCH_LIST=12.0 FORCE_CUDA=1 MAX_JOBS=2 \
    PYTHONPATH=/workspace:/workspace/src \
    PYTHONPYCACHEPREFIX=/tmp/pytorch3d-python-cache \
    XDG_CACHE_HOME=/tmp/pytorch3d-cache
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libegl1 libgl1 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir ninja==1.11.1.4 iopath==0.1.10 \
        fvcore==0.1.5.post20221221 moderngl==5.12.0 glcontext==3.0.0
RUN git clone --filter=blob:none https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d-source \
    && cd /tmp/pytorch3d-source && git checkout "$PYTORCH3D_REV" \
    && python -m pip install -v --no-build-isolation --no-deps . \
    && printf '%s\n' "$PYTORCH3D_REV" > /opt/pytorch3d-source-revision \
    && rm -rf /tmp/pytorch3d-source
RUN python -m pip install --no-cache-dir nvidia-ml-py==12.570.86
ENV PYTHONPYCACHEPREFIX=/tmp/pytorch3d-runtime-python-cache \
    XDG_CACHE_HOME=/tmp/pytorch3d-runtime-cache
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
