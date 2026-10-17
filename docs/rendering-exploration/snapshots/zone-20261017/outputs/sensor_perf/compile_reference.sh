#!/usr/bin/env bash
set -euo pipefail
# Run inside the EXISTING yopo-local/tof:dev container, with only the reference
# Simulator/src mounted read-only at /reference and this outputs directory /out.
# No GPU is required for compilation. Override BENCHMARK_CUDA_ARCH for other GPUs.
benchmark_arch="${BENCHMARK_CUDA_ARCH:-sm_120}"
nvcc --version > /out/reference_compiler.txt
sha256sum /reference/src/sensor_simulator.cu \
    /reference/include/sensor_simulator.cuh /reference/include/tof_renderer.hpp \
    /reference/include/target_geometry.hpp /reference/include/scene_snapshot.hpp \
    /reference/include/cuda_toolkit/se3.cuh /reference/include/cuda_toolkit/matrix.cuh \
    /reference/include/cuda_toolkit/helper_math.h > /out/reference_sources.sha256
nvcc -O3 -std=c++17 -arch="$benchmark_arch" \
    -I/reference/include -I/usr/include/eigen3 -I/usr/include/pcl-1.10 \
    -I/usr/include/opencv4 \
    /reference/src/sensor_simulator.cu /out/reference_benchmark.cu \
    -lpcl_common -lopencv_core -lyaml-cpp \
    -o /out/reference_benchmark
