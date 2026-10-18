"""Run the project's checks in the development container, stopping on failure.

The map generator must already be built; ``just check`` prepares it. Each
verification script documents its own scope and writes its results to outputs/.
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
        [sys.executable, "tests/verify_map_viewer.py", "--results", "outputs/map_viewer/verification.json"],
        [sys.executable, "tests/verify_ray_sensors.py", "--results", "outputs/sensors/ray-verification.json"],
        [sys.executable, "tests/verify_sensor_batch.py", "--results", "outputs/sensors/batch-verification.json"],
        [sys.executable, "tests/verify_ray_review.py", "--results", "outputs/sensor_review/ray-verification.json"],
    ]
    for command in checks:
        print(f"check: {command[1]}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
