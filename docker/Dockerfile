# Project development base, including the optional CUDA/GL sensor runtime.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libegl1 \
        libgl1 \
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

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
COPY devtools ./devtools
COPY training ./training
# CUDA 12.8 wheels support the Blackwell GPU used by the sensor checks. Install
# the explicit CUDA build first; it also satisfies the sensors extra pin.
RUN python -m pip install --no-cache-dir torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-build-isolation --no-cache-dir -e '.[viewer,sensors]'
ENV PYTHONPYCACHEPREFIX=/tmp/runtime-python-cache

CMD ["bash"]
