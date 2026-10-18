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

# Browser modules for the review tool.
COPY docker/fetch_review_assets.py /tmp/fetch_review_assets.py
RUN python /tmp/fetch_review_assets.py

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
COPY devtools ./devtools
COPY training ./training
# Python requirements come from pyproject.toml; use the CUDA 12.8 wheel index.
RUN python -m pip install --no-build-isolation --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 -e '.[viewer,gpu]'
ENV PYTHONPYCACHEPREFIX=/tmp/runtime-python-cache

CMD ["bash"]
