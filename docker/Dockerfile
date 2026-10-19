# Browser dependencies are installed by npm, with no Node runtime in the image.
FROM node:24.15.0-bookworm-slim@sha256:4e6b70dd6cbfc88c8157ba19aa3d9f9cce6ba4703576d55459e45efcbc9c5f5d AS web-assets
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund \
    && mkdir /assets \
    && cp node_modules/three/build/three.module.js node_modules/three/build/three.core.js /assets/ \
    && cp node_modules/three/examples/jsm/controls/OrbitControls.js /assets/ \
    && cp node_modules/three/LICENSE /assets/three-LICENSE.txt \
    && cp package.json package-lock.json /assets/

# Project development environment.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        python3 \
        python3-setuptools \
        python3-venv \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# The venv uses Ubuntu's build tooling. Project dependencies are installed here,
# never in the host Python environment or a bind-mounted host virtualenv.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    LANG=C.UTF-8 \
    PYTHONPYCACHEPREFIX=/tmp/python-cache \
    PIP_CACHE_DIR=/tmp/pip-cache

COPY --from=web-assets /assets /opt/sensor-review-web

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
COPY devtools ./devtools
COPY training ./training
# Python requirements come from pyproject.toml; use the CUDA 12.8 wheel index.
RUN python -m pip install --no-build-isolation --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 -e .
ENV PYTHONPYCACHEPREFIX=/tmp/runtime-python-cache

CMD ["bash"]
