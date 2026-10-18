# Isolated experiment image; does not modify the shared development image.
ARG BASE_IMAGE=tof-tracker-esdf-eval:nvblox-24eee49-cu128
FROM ${BASE_IMAGE}
RUN python -m pip install --no-cache-dir warp-lang==1.15.0 trimesh==4.8.3
ENV PYTHONPATH=/workspace:/workspace/src \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/mesh-distance-cache
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
