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
project(environment_check LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(environment_check check.cpp)
CMAKE

cat > "$check_dir/check.cpp" <<'CPP'
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    static_assert(__cplusplus >= 201703L, "C++17 is required");
    const auto path = std::filesystem::path(argv[1]) / "check.txt";
    const std::string expected = "tof-tracker C++17 environment";
    {
        std::ofstream output(path);
        output << expected << '\n';
        output.close();
        if (!output) return 2;
    }
    std::ifstream input(path);
    std::string loaded;
    if (!std::getline(input, loaded) || loaded != expected) return 3;
    std::cout << "C++17 filesystem + file round-trip passed\n";
}
CPP

cmake -S "$check_dir" -B "$check_dir/build" >/dev/null
cmake --build "$check_dir/build" --parallel 2 >/dev/null
"$check_dir/build/environment_check" "$check_dir"

# This artifact also verifies that writes to the repository use the host UID/GID.
mkdir -p outputs/docker
printf 'uid=%s gid=%s\n' "$(id -u)" "$(id -g)" > outputs/docker/environment-check.txt
echo "Environment check passed; ownership record: outputs/docker/environment-check.txt"
