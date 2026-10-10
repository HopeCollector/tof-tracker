#!/usr/bin/env bash
# Run through ./docker/dev.sh run bash docker/check-env.sh.
set -euo pipefail

check_dir="$(mktemp -d /tmp/tof-tracker-env.XXXXXX)"
trap 'rm -rf "$check_dir"' EXIT

python - <<'PY'
import importlib
import importlib.metadata
import sys
from pathlib import Path

assert sys.prefix == "/opt/venv", sys.prefix
print(f"Installed project: tof-tracker {importlib.metadata.version('tof-tracker')}")
for name in ("tof_tracker", "devtools", "training"):
    module = importlib.import_module(name)
    source = Path(module.__file__).resolve()
    assert source.is_relative_to("/workspace"), source
    print(f"Python import: {name} -> {source}")
print(f"Python: {sys.version.split()[0]}; venv: {sys.prefix}")
PY

cat > "$check_dir/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(environment_check LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
add_executable(environment_check check.cpp)
target_include_directories(environment_check PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(environment_check PRIVATE Eigen3::Eigen ${PCL_LIBRARIES})
target_compile_options(environment_check PRIVATE ${PCL_DEFINITIONS})
CMAKE

cat > "$check_dir/check.cpp" <<'CPP'
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <cmath>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    static_assert(__cplusplus >= 201703L, "C++17 is required");
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.push_back(pcl::PointXYZ(1.f, 2.f, 3.f));
    cloud.push_back(pcl::PointXYZ(-1.f, 0.f, 0.5f));
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() = Eigen::Vector3f(10.f, -2.f, 1.f);
    pcl::PointCloud<pcl::PointXYZ> moved;
    pcl::transformPointCloud(cloud, moved, transform);
    const auto path = std::filesystem::path(argv[1]) / "check.pcd";
    if (pcl::io::savePCDFileBinary(path.string(), moved) != 0) return 2;
    pcl::PointCloud<pcl::PointXYZ> loaded;
    if (pcl::io::loadPCDFile(path.string(), loaded) != 0) return 3;
    if (loaded.size() != 2) return 4;
    const Eigen::Vector3f expected[] = {{11.f, 0.f, 4.f}, {9.f, -2.f, 1.5f}};
    for (std::size_t i = 0; i < loaded.size(); ++i) {
        if ((loaded[i].getVector3fMap() - expected[i]).norm() > 1e-6f) return 5;
    }
    std::cout << "C++17 + Eigen + PCL: two translated points match after PCD round-trip\n";
}
CPP

cmake -S "$check_dir" -B "$check_dir/build" >/dev/null
cmake --build "$check_dir/build" --parallel 2 >/dev/null
"$check_dir/build/environment_check" "$check_dir"

# This artifact also verifies that writes to the repository use the host UID/GID.
mkdir -p outputs/docker
printf 'uid=%s gid=%s\n' "$(id -u)" "$(id -g)" > outputs/docker/environment-check.txt
echo "Environment check passed; ownership record: outputs/docker/environment-check.txt"
