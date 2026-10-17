"""Create a separate SdfLib copy with a bounded raw-Hermite consistency patch.

Run inside the experiment Docker image. The original checkout is read-only;
the destination must not exist. This is an experiment adaptation, not an
upstream fix or a controlled-smoothing construction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


REVISION = "8db373ef71d65be24badf6ae10750a932bbc223b"
HEADER = Path("include/SdfLib/InterpolationMethods.h")
EXPECTED_HEADER_SHA256 = "c258cab51c35f96116dd98d43a946fb474fbcbee318111c3cfa43fb0d83df265"
PATCH_NAME = "raw-hermite-first-sample-1um-v1"

PREAMBLE = """        // Experiment-only raw Hermite sharing; see ADAPTATION.json.
        // This cache is deliberately outside interpolateVertexValues: hanging
        // nodes constrained by a coarse polynomial retain their original data.
        // One mesh per process; deterministic first-sample choice needs one thread.
        static std::mutex sampleMutex;
        static const Mesh* sampleMesh = nullptr;
        static std::map<std::array<long long, 3>,
                        std::array<float, VALUES_PER_VERTEX>> rawSamples;
        const std::lock_guard<std::mutex> sampleGuard(sampleMutex);
        if (sampleMesh != nullptr && sampleMesh != &mesh)
            throw std::runtime_error("Raw nodal patch supports one mesh per process");
        sampleMesh = &mesh;
        constexpr double quantization = 1e-6;
        const std::array<long long, 3> sampleKey = {
            std::llround(static_cast<double>(point.x) / quantization),
            std::llround(static_cast<double>(point.y) / quantization),
            std::llround(static_cast<double>(point.z) / quantization)};
        const auto existingSample = rawSamples.find(sampleKey);
        if (existingSample != rawSamples.end())
        {
            outValues = existingSample->second;
            return;
        }

"""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def make_patched_copy(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination must be separate, non-nested directories")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing destination: {destination}")
    original = (source / HEADER).read_bytes()
    if _sha256(original) != EXPECTED_HEADER_SHA256:
        raise ValueError("source header differs from the inspected pinned SdfLib revision")
    text = original.decode("utf-8")
    marker = "\nstruct TriCubicInterpolation\n{"
    if text.count(marker) != 1:
        raise ValueError("expected one active TriCubicInterpolation definition")
    prefix, active = text.split(marker)
    function_marker = "        const std::vector<uint32_t>& indices = mesh.getIndices();"
    if active.count(function_marker) != 1:
        raise ValueError("raw point-value implementation changed")
    active = active.replace(function_marker, PREAMBLE + function_marker, 1)
    end_marker = "        outValues[4] = 0.0f; outValues[5] = 0.0f; outValues[6] = 0.0f; outValues[7] = 0.0f;"
    if active.count(end_marker) != 1:
        raise ValueError("raw Hermite output layout changed")
    active = active.replace(end_marker, end_marker + "\n        rawSamples.emplace(sampleKey, outValues);", 1)
    include_marker = "#include <array>"
    if prefix.count(include_marker) != 1:
        raise ValueError("header include layout changed")
    prefix = prefix.replace(include_marker, include_marker +
                            "\n#include <cmath>\n#include <map>\n#include <mutex>\n#include <stdexcept>", 1)
    patched = (prefix + marker + active).encode("utf-8")
    manifest = {
        "adaptation": PATCH_NAME,
        "upstream_revision": REVISION,
        "source_directory": str(source),
        "patched_file": str(HEADER),
        "original_header_sha256": _sha256(original),
        "patched_header_sha256": _sha256(patched),
        "change": "Cache all eight raw calculatePointValues Hermite entries by a quantized spatial key; return the first sample for repeated keys.",
        "key": "llround(double(point_axis) / 1e-6), independently for x,y,z",
        "quantization_m": 1e-6,
        "intended_bounds_m": [[-.8, -.8, -.8], [.8, .8, .8]],
        "intended_start_depth": 3,
        "intended_max_depth": [3, 4, 5, 6],
        "smallest_intended_cell_m": .025,
        "process_contract": "Exactly one immutable mesh and one octree construction per exporter process; run with --threads 1.",
        "mutex": "Cache access and raw sample creation are protected; more than one thread can still change which sample wins, and this does not repair upstream statistics races.",
        "preserved": ["upstream mesh-distance oracle and triangle selection", "octree refinement and topology algorithm", "coarse-to-fine interpolateVertexValues constraints", "tricubic coefficient construction and query evaluation"],
        "limits": ["The representative gradient at a medial tie remains arbitrary.", "This is not a prescribed smoothing width, a conservative SDF, or a global error guarantee.", "The 1um key is justified only for the inspected dyadic fixture grids; distinct geometry sampling nodes closer than that can be merged.", "Sharing raw Hermite inputs does not prove all adaptive coarse/fine interfaces continuous; rerun full seam checks."],
    }
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns(".git", "build", "_build", "__pycache__"))
    (destination / HEADER).write_bytes(patched)
    (destination / "ADAPTATION.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if (source / HEADER).read_bytes() != original:
        raise RuntimeError("original source unexpectedly changed during patch creation")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(make_patched_copy(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
