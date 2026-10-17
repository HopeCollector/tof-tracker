# Default builds from a public pinned base. A pre-existing project image may be
# supplied with --build-arg BASE_IMAGE=... to reuse the same Torch toolchain.
ARG BASE_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
FROM ${BASE_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates git cmake g++ make python3 python3-venv \
    libglm-dev libspdlog-dev libcereal-dev \
    && rm -rf /var/lib/apt/lists/* \
    && (test -x /opt/venv/bin/python || python3 -m venv /opt/venv)
ENV PATH="/opt/venv/bin:${PATH}" PYTHONPATH=/workspace:/workspace/src \
    HOME=/tmp PYTHONPYCACHEPREFIX=/tmp/python-cache XDG_CACHE_HOME=/tmp/adaptive-sdf-cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python -c 'import torch,numpy; assert torch.__version__ == "2.9.1+cu128"; assert numpy.__version__ == "2.5.4"' \
    || (python -m pip install --no-cache-dir numpy==2.5.4 \
        && python -m pip install --no-cache-dir torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128)
RUN git clone https://github.com/UPC-ViRVIG/SdfLib.git /opt/SdfLib \
    && git -C /opt/SdfLib checkout --detach 8db373ef71d65be24badf6ae10750a932bbc223b \
    && test "$(git -C /opt/SdfLib rev-parse HEAD)" = 8db373ef71d65be24badf6ae10750a932bbc223b
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
