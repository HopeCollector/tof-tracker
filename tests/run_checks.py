"""Run the project's checks in the development container, stopping on failure.

The map generator must already be built; ``just check`` prepares it. This runner
builds the mesh-distance library before the formal ESDF checks. Verification
scripts document their scopes and report their results.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = [
        ["bash", "docker/check-env.sh"],
        [sys.executable, "tests/verify_mapgen.py", "--binary", "outputs/mapgen/build/offline_mapgen",
         "--results", "outputs/mapgen/verification.json"],
        [sys.executable, "tests/verify_geometry.py", "--results", "outputs/geometry/verification.json"],
        [sys.executable, "tests/verify_scene.py", "--results", "outputs/scene/verification.json"],
        [sys.executable, "tests/verify_trajectory.py", "--results", "outputs/trajectory/verification.json"],
        ["cmake", "-S", "devtools/esdf", "-B", "outputs/esdf/build", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", "outputs/esdf/build", "--parallel", "2"],
        [sys.executable, "tests/verify_mesh_distance.py", "--library", "outputs/esdf/build/libmesh_distance.so"],
        [sys.executable, "tests/verify_esdf.py", "--results", "outputs/esdf/verification.json"],
        [sys.executable, "tests/verify_map_viewer.py", "--results", "outputs/map_viewer/verification.json"],
    ]
    for command in checks:
        print("check:", " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
