"""Run the compiled offline generator and check its PLY geometry independently."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile


def read_ply(path):
    with path.open() as stream:
        header = []
        for line in stream:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        assert header[:2] == ["ply", "format ascii 1.0"]
        assert header[-1] == "end_header"
        counts = [int(line.split()[-1]) for line in header if line.startswith("element vertex ")]
        assert len(counts) == 1
        assert [line for line in header if line.startswith("property ")] == [
            "property float x", "property float y", "property float z"]
        points = [tuple(map(float, line.split())) for line in stream if line.strip()]
    assert len(points) == counts[0] and points
    assert all(len(point) == 3 and all(math.isfinite(v) for v in point) for point in points)
    return points


def lattice(points, sizes, resolution, height=0):
    """Identify samples of the independently specified plane grid, within float error."""
    nx, ny = [round(size / resolution) for size in sizes[:2]]
    cells = set()
    for x, y, z in points:
        if abs(z - height) > 2e-6:
            continue
        i, j = round((x + sizes[0] / 2) / resolution), round((y + sizes[1] / 2) / resolution)
        if (0 <= i < nx and 0 <= j < ny
                and abs(x - (i * resolution - sizes[0] / 2)) < 2e-6
                and abs(y - (j * resolution - sizes[1] / 2)) < 2e-6):
            cells.add((i, j))
    return cells


def check_geometry(points, sizes, resolution):
    sx, sy, sz = sizes
    assert all(-sx / 2 - 2e-6 <= x <= sx / 2 + 2e-6
               and -sy / 2 - 2e-6 <= y <= sy / 2 + 2e-6
               and -2e-6 <= z <= sz + 2e-6 for x, y, z in points)
    expected = {(i, j) for i in range(round(sx / resolution))
                for j in range(round(sy / resolution))}
    assert lattice(points, sizes, resolution) == expected, "incomplete ground grid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}; compile it first")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    sizes, resolution = (8, 6, 3), 0.2
    common = ["--size-x", "8", "--size-y", "6", "--size-z", "3", "--resolution", "0.2"]
    with tempfile.TemporaryDirectory(prefix="mapgen-check-", dir=args.results.parent) as temp:
        directory = Path(temp)

        def run(path, options=(), succeeds=True):
            result = subprocess.run([str(binary), "--output", str(path), *common, *options],
                                    capture_output=True, text=True, timeout=30)
            if succeeds:
                assert result.returncode == 0, result.stderr
                return read_ply(path)
            assert result.returncode != 0, f"accepted invalid request: {options}"
            return result

        baseline = {}
        for mode in ["pillars", "walls", "perlin"]:
            paths = [directory / f"{mode} {n}.ply" for n in range(3)]
            groups = [run(path, ["--type", mode, "--seed", str(seed)])
                      for path, seed in zip(paths, [17, 17, 18])]
            assert paths[0].read_bytes() == paths[1].read_bytes(), f"same seed differs: {mode}"
            assert set(groups[0]) != set(groups[2]), f"seed changed only metadata/order: {mode}"
            for points in groups:
                check_geometry(points, sizes, resolution)
                assert any(p[2] > 2e-6 for p in points), f"no above-ground geometry: {mode}"
            baseline[mode] = groups
            reports.append({"mode": mode, "seed": 17, "comparison_seed": 18,
                            "points": len(groups[0]),
                            "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                            "bounds": [[min(p[k] for p in groups[0]), max(p[k] for p in groups[0])]
                                       for k in range(3)],
                            "different_seed_added_vertices": len(set(groups[2]) - set(groups[0])),
                            "different_seed_removed_vertices": len(set(groups[0]) - set(groups[2])),
                            "checks": ["same seed identical bytes", "different seed different vertex sets",
                                       "PLY parsed", "finite points", "requested bounds", "complete ground grid"]})

        # Odd grid dimensions and non-default extents must retain the whole ground.
        for mode in ["pillars", "walls", "perlin"]:
            points = run(directory / f"odd-{mode}.ply", ["--type", mode, "--count", "2",
                         "--size-x", "1.4", "--size-y", "1", "--size-z", "1.2"])
            check_geometry(points, (1.4, 1, 1.2), resolution)

        floor_path = directory / "ground-only.ply"
        floor = run(floor_path, ["--count", "0", "--max-points", "1200"])
        check_geometry(floor, sizes, resolution)
        assert len(floor) == len(set(floor)) == 40 * 30 and all(p[2] == 0 for p in floor)
        ceiling = run(directory / "empty-walls-ceiling.ply",
                      ["--type", "walls", "--count", "0", "--ceiling", "1"])
        assert len(ceiling) == 2 * len(floor)
        assert set(ceiling) == set(floor) | {(x, y, 3.0) for x, y, _ in floor}
        assert any(abs(z / resolution - round(z / resolution)) > 1e-4
                   for _, _, z in baseline["walls"][0]), "wall tilt disappeared"

        # Three samples across an odd-width pillar, with no interior samples.
        pillar = run(directory / "odd-width-pillar.ply", ["--count", "1", "--seed", "17",
                     "--width-min", "0.6", "--width-max", "0.6"])
        floor_set = set(floor)
        obstacle = [p for p in pillar if p not in floor_set]
        xs, ys, zs = [sorted({p[k] for p in obstacle}) for k in range(3)]
        assert len(xs) == len(ys) == 3 and len(zs) > 2
        assert math.isclose(xs[-1] - xs[0], 0.4, abs_tol=2e-6)
        assert math.isclose(ys[-1] - ys[0], 0.4, abs_tol=2e-6)
        assert len(obstacle) == 9 * len(zs) - max(len(zs) - 2, 0)
        assert all(x in (xs[0], xs[-1]) or y in (ys[0], ys[-1]) or z in (zs[0], zs[-1])
                   for x, y, z in obstacle)

        # Perlin points occupy the declared XYZ grid; integer-frequency noise is
        # legitimately constant, so fill/seed do not guarantee obstacles.
        for x, y, z in baseline["perlin"][0]:
            assert all(abs(v - round(v)) < 2e-5
                       for v in ((x + 4) / resolution, (y + 3) / resolution, z / resolution))
        constant = run(directory / "constant-perlin.ply", ["--type", "perlin", "--complexity", "0.5"])
        assert set(constant) == set(floor) and len(constant) == len(floor)

        invalid = [
            ["--resolution", "0"], ["--size-x", "-1"], ["--size-x", "8.1"],
            ["--type", "unknown"], ["--seed", "4294967296"], ["--seed", "-1"],
            ["--size-z", "nan"], ["--size-z", "inf"], ["--max-points", "0"],
            ["--max-points", "10"], ["--max-points", "1200", "--count", "1"],
            ["--count", "-1"], ["--count", "100001"], ["--width-min", "0.1"],
            ["--width-min", "2", "--width-max", "1"], ["--type", "walls", "--ceiling", "2"],
            ["--type", "walls", "--wall-thickness", "0"],
            ["--type", "perlin", "--fill", "0"], ["--type", "perlin", "--fill", "1"],
            ["--type", "perlin", "--fractal", "0"], ["--type", "perlin", "--fractal", "17"],
            ["--type", "perlin", "--attenuation", "0"], ["--type", "perlin", "--complexity", "0.6"],
            ["--size-x", "1e40"], ["--size-x", "1e-310"], ["--unknown", "1"], ["--seed"],
            ["--type", "walls", "--size-x", "2147483645", "--size-y", "1", "--size-z", "1",
             "--resolution", "1", "--width-min", "2147483645", "--width-max", "2147483645",
             "--wall-thickness", "1", "--max-points", "4294967295", "--count", "1"],
            ["--size-x", "1e6", "--size-y", "0.001", "--size-z", "0.001", "--resolution", "0.001",
             "--width-min", "0.001", "--width-max", "0.001", "--count", "0", "--max-points", "4294967295"],
        ]
        for index, options in enumerate(invalid):
            path = directory / f"invalid-{index}.ply"
            run(path, options, succeeds=False)
            assert not path.exists(), f"invalid request left output: {options}"
        missing = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
        assert missing.returncode != 0
        run(directory / "absent-directory" / "map.ply", succeeds=False)

        previous = floor_path.read_bytes()
        run(floor_path, succeeds=False)
        assert floor_path.read_bytes() == previous, "overwrote existing file"
        dangling, target = directory / "dangling.ply", directory / "not-created.ply"
        dangling.symlink_to(target)
        run(dangling, succeeds=False)
        assert dangling.is_symlink() and not target.exists()
        existing_link = directory / "existing-link.ply"
        existing_link.symlink_to(floor_path)
        run(existing_link, succeeds=False)
        assert existing_link.is_symlink() and floor_path.read_bytes() == previous
        run(directory, succeeds=False)

        # A single complete map wins simultaneous publication; the loser cannot
        # overwrite the winner. No temporary file survives either path.
        concurrent_path = directory / "concurrent.ply"
        processes = [subprocess.Popen([str(binary), "--output", str(concurrent_path), *common,
                     "--seed", str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for seed in (17, 18)]
        for process in processes:
            process.communicate(timeout=30)
        assert sorted(p.returncode for p in processes) == [0, 1]
        assert set(read_ply(concurrent_path)) in [set(p) for p in baseline["pillars"][::2]]
        assert not list(directory.glob(".*.tmp-*")), "output temporary file was not cleaned"

    args.results.write_text(json.dumps({
        "status": "passed", "binary": str(binary), "maps": reports,
        "invalid_requests_checked": len(invalid) + 2,
        "additional_checks": ["odd-size grids", "exact zero-obstacle ground",
                              "empty walls with exact ceiling grid", "tilted walls retained",
                              "odd-width pillar shell", "Perlin voxel grid and degenerate case",
                              "invalid requests leave no output", "float precision boundary refused",
                              "existing files and symlinks preserved", "concurrent publication safe",
                              "temporary outputs cleaned"]}, indent=2) + "\n")
    print(f"passed: 3 modes, seed 17/18, {len(invalid) + 2} rejected requests; {args.results}")


if __name__ == "__main__":
    main()
