"""Check pillar OBJ dimensions, independent closed solids, ground and safe output publication."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile


def read_obj(path):
    """Parse generated OBJ independently of the viewer or scene loader."""
    objects, vertices, comments = [], [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "#":
            comments.append(line)
        elif parts[0] == "o":
            assert len(parts) == 2
            objects.append({"name": parts[1], "vertices": [], "faces": []})
        elif parts[0] == "v":
            assert objects and len(parts) == 4
            point = tuple(map(float, parts[1:]))
            assert all(math.isfinite(v) for v in point)
            vertices.append(point)
            objects[-1]["vertices"].append(len(vertices) - 1)
        elif parts[0] == "f":
            assert objects and len(parts) == 4
            assert all(part.isdecimal() for part in parts[1:])
            face = tuple(int(part) - 1 for part in parts[1:])
            assert len(set(face)) == 3 and all(0 <= i < len(vertices) for i in face)
            assert set(face) <= set(objects[-1]["vertices"]), "objects share vertices"
            objects[-1]["faces"].append(face)
        else:
            raise AssertionError(f"unexpected OBJ line: {line}")
    assert comments[0] == "# unit metre"
    assert len({obj["name"] for obj in objects}) == len(objects)
    assert not any("resolution" in line for line in comments)
    return {"objects": objects, "vertices": vertices, "comments": comments}


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def check_mesh(mesh, sizes, count):
    """Use independent corner, oriented-edge, normal and volume oracles."""
    sx, sy, sz = sizes
    vertices, objects = mesh["vertices"], mesh["objects"]
    assert [obj["name"] for obj in objects] == [f"pillar_{i}" for i in range(count)] + ["ground"]
    assert len(vertices) == count * 8 + 4
    assert all(-sx / 2 <= x <= sx / 2 and -sy / 2 <= y <= sy / 2 and 0 <= z <= sz
               for x, y, z in vertices)
    dimensions = []
    for obj in objects[:-1]:
        points = [vertices[i] for i in obj["vertices"]]
        assert len(points) == len(set(points)) == 8 and len(obj["faces"]) == 12
        axes = [sorted({p[k] for p in points}) for k in range(3)]
        assert all(len(axis) == 2 for axis in axes) and axes[2][0] == 0
        assert set(points) == {(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]}
        center = tuple((axis[0] + axis[1]) / 2 for axis in axes)
        edges, volume = {}, 0
        for face in obj["faces"]:
            a, b, c = (vertices[i] for i in face)
            normal = cross(subtract(b, a), subtract(c, a))
            assert dot(normal, normal) > 0
            face_center = tuple((a[k] + b[k] + c[k]) / 3 for k in range(3))
            assert dot(normal, subtract(face_center, center)) > 0, "inward triangle"
            volume += dot(subtract(a, center), cross(subtract(b, center), subtract(c, center))) / 6
            for i, j in zip(face, face[1:] + face[:1]):
                edges[(i, j)] = edges.get((i, j), 0) + 1
        assert len(edges) == 36 and all(n == 1 and edges.get((j, i)) == 1
                                      for (i, j), n in edges.items()), "open/nonmanifold box"
        spans = tuple(axis[1] - axis[0] for axis in axes)
        assert math.isclose(volume, math.prod(spans), rel_tol=1e-12) and volume > 0
        dimensions.append(spans)
    ground = objects[-1]
    assert len(ground["vertices"]) == 4 and len(ground["faces"]) == 2
    assert {vertices[i] for i in ground["vertices"]} == {
        (-sx / 2, -sy / 2, 0), (sx / 2, -sy / 2, 0),
        (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, 0)}
    area = 0
    for face in ground["faces"]:
        a, b, c = (vertices[i] for i in face)
        normal = cross(subtract(b, a), subtract(c, a))
        assert normal[:2] == (0, 0) and normal[2] > 0
        area += normal[2] / 2
    assert math.isclose(area, sx * sy, rel_tol=1e-12)
    metadata_sizes = [tuple(map(float, line.split()[2:])) for line in mesh["comments"]
                      if line.startswith("# size_xyz ")]
    assert metadata_sizes == [sizes]
    return dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}; compile it first")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    sizes = (8, 6, 3)
    with tempfile.TemporaryDirectory(prefix="mapgen-check-", dir=args.results.parent) as temp:
        directory = Path(temp)

        common = ["--size-x", "8", "--size-y", "6", "--size-z", "3"]

        def run(path, options=(), succeeds=True):
            result = subprocess.run([str(binary), "--output", str(path), *common, *options],
                                    capture_output=True, text=True, timeout=30)
            if succeeds:
                assert result.returncode == 0, result.stderr
                return read_obj(path)
            assert result.returncode != 0, f"accepted invalid mesh request: {options}"
            return result

        mesh_paths = [directory / f"mesh-{n}.obj" for n in range(3)]
        meshes = [run(path, ["--seed", str(seed)])
                  for path, seed in zip(mesh_paths, [17, 17, 18])]
        mesh_digest = hashlib.sha256(mesh_paths[0].read_bytes()).hexdigest()
        assert mesh_paths[0].read_bytes() == mesh_paths[1].read_bytes()
        assert set(meshes[0]["vertices"]) != set(meshes[2]["vertices"])
        for mesh in meshes:
            spans = check_mesh(mesh, sizes, 12)
            assert all(0 < x <= 1.2 and 0 < y <= 1.2 and 0.3 <= z <= 3 for x, y, z in spans)
        exact = run(directory / "physical-dimensions.obj", ["--count", "1", "--seed", "17",
                         "--width-min", "0.6", "--width-max", "0.6",
                         "--height-min", "1.3", "--height-max", "1.3"])
        spans = check_mesh(exact, sizes, 1)[0]
        assert all(math.isclose(actual, expected, abs_tol=1e-14)
                   for actual, expected in zip(spans, (0.6, 0.6, 1.3)))

        # No resolution multiple is needed, and narrow pillars retain actual width.
        narrow = run(directory / "narrow.obj", ["--count", "1", "--seed", "17",
                          "--size-x", "8.13", "--size-y", "6.17", "--size-z", "3.19",
                          "--width-min", "0.013", "--width-max", "0.013",
                          "--height-min", "0.071", "--height-max", "0.071"])
        spans = check_mesh(narrow, (8.13, 6.17, 3.19), 1)[0]
        assert all(math.isclose(actual, expected, abs_tol=1e-14)
                   for actual, expected in zip(spans, (0.013, 0.013, 0.071)))

        # A wider/taller physical obstacle clips to a complete closed box.
        clipped = run(directory / "clipped.obj", ["--count", "2", "--seed", "17",
                           "--width-min", "16", "--width-max", "16",
                           "--height-min", "8", "--height-max", "8"])
        assert check_mesh(clipped, sizes, 2) == [(8, 6, 3)] * 2
        edge = run(directory / "edge-clipped.obj", ["--count", "50", "--seed", "17",
                        "--width-min", "1.2", "--width-max", "1.2"])
        spans = check_mesh(edge, sizes, 50)
        assert any(x < 1.2 - 1e-12 or y < 1.2 - 1e-12 for x, y, _ in spans)
        assert all(x > 0 and y > 0 for x, y, _ in spans)

        ground_path = directory / "ground-only.obj"
        ground = run(ground_path, ["--type", "pillars", "--count", "0"])
        assert check_mesh(ground, sizes, 0) == []
        mesh_invalid = [
            ["--type", "walls"], ["--type", "perlin"], ["--type", "unknown"],
            ["--seed", "4294967296"], ["--seed", "-1"], ["--seed"],
            ["--unknown", "1"], ["--width-min", "0.6junk"],
            ["--resolution", "0.2"], ["--max-points", "5000000"],
            ["--wall-thickness", "0.2"], ["--ceiling", "0"],
            ["--complexity", "0.02"], ["--fill", "0.1"], ["--fractal", "1"],
            ["--attenuation", "0.1"], ["--count", "100001"], ["--count", "-1"],
            ["--width-min", "0"], ["--width-max", "0.1"],
            ["--height-min", "0"], ["--height-max", "0"],
            ["--height-min", "2", "--height-max", "1"],
            ["--size-x", "0"], ["--size-z", "nan"], ["--size-y", "inf"],
            ["--size-x", "1e40"], ["--size-x", "1e-310"],
            ["--width-min", "1e-310"], ["--height-max", "1e40"],
            ["--size-x", "1e30", "--count", "1"],
        ]
        for index, options in enumerate(mesh_invalid):
            path = directory / f"invalid-mesh-{index}.obj"
            run(path, options, succeeds=False)
            assert not path.exists()
        for suffix in (".ply", ".mesh"):
            path = directory / f"unsupported{suffix}"
            run(path, succeeds=False)
            assert not path.exists()
        missing = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
        assert missing.returncode != 0
        run(directory, succeeds=False)
        run(directory / "absent-mesh-directory" / "map.obj", succeeds=False)

        previous_mesh = ground_path.read_bytes()
        run(ground_path, succeeds=False)
        assert ground_path.read_bytes() == previous_mesh
        for label, target in [("dangling", directory / "no-mesh.obj"), ("existing", ground_path)]:
            link = directory / f"{label}-mesh-link.obj"
            link.symlink_to(target)
            run(link, succeeds=False)
            assert link.is_symlink()
        assert not (directory / "no-mesh.obj").exists()
        assert ground_path.read_bytes() == previous_mesh

        concurrent_mesh = directory / "concurrent.obj"
        processes = [subprocess.Popen([str(binary), "--output", str(concurrent_mesh), *common,
                     "--seed", str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for seed in (17, 18)]
        for process in processes:
            process.communicate(timeout=30)
        assert sorted(p.returncode for p in processes) == [0, 1]
        published = read_obj(concurrent_mesh)
        check_mesh(published, sizes, 12)
        assert set(published["vertices"]) in [set(m["vertices"]) for m in meshes[::2]]
        assert not list(directory.glob(".*.tmp-*"))

    args.results.write_text(json.dumps({
        "status": "passed", "binary": str(binary), "seed": 17, "comparison_seed": 18,
        "sha256": mesh_digest,
        "invalid_requests_checked": len(mesh_invalid) + 2,
        "checks": ["independent OBJ parsing", "physical dimensions and narrow pillars",
                   "closed outward manifold boxes", "positive volume",
                   "separate +Z ground plane", "closed boundary clipping",
                   "independent overlapping objects", "seed repeatability",
                   "invalid requests leave no output", "precision boundary refused",
                   "existing files and symlinks preserved", "concurrent publication safe",
                   "temporary outputs cleaned"]}, indent=2) + "\n")
    print(f"passed: pillar OBJ meshes, seed 17/18, "
          f"{len(mesh_invalid) + 2} rejected requests; {args.results}")


if __name__ == "__main__":
    main()
