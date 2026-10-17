"""Create a separate SdfLib copy with optional raw-Hermite/value adaptations.

Run inside the experiment Docker image. The original checkout is read-only;
the destination must not exist. This is an experiment adaptation, not an
upstream fix or a controlled-smoothing construction.
"""

import argparse
import hashlib
import json
import math
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

ROBUST_POINT_VALUES = """        const std::vector<uint32_t>& indices = mesh.getIndices();
        const std::vector<glm::vec3>& vertices = mesh.getVertices();
        // The supplied nearestTriangleIndex may come from the upstream face
        // cache. Re-query the complete mesh, then recompute the selected face's
        // closest point: TriangleUtils misclassifies obtuse regions, and tmd's
        // expanded quadratic distance/closest point can cancel near a surface.
        (void)nearestTriangleIndex;
        (void)trianglesData;
        static std::unique_ptr<tmd::TriangleMeshDistance> robustMeshDistance;
        if (!robustMeshDistance)
        {
            std::vector<std::array<double, 3>> doubleVertices;
            doubleVertices.reserve(vertices.size());
            for (const glm::vec3 vertex : vertices)
                doubleVertices.push_back({vertex.x, vertex.y, vertex.z});
            std::vector<std::array<int, 3>> faces;
            faces.reserve(indices.size() / 3);
            for (std::size_t i = 0; i < indices.size(); i += 3)
            {
                for (int k = 0; k < 3; ++k)
                    if (indices[i + k] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
                        throw std::runtime_error("Robust mesh primitive requires int32 vertex indices");
                faces.push_back({static_cast<int>(indices[i]), static_cast<int>(indices[i + 1]), static_cast<int>(indices[i + 2])});
            }
            robustMeshDistance = std::make_unique<tmd::TriangleMeshDistance>(doubleVertices, faces);
        }
        const tmd::Result result = robustMeshDistance->signed_distance(
            std::array<double, 3>{point.x, point.y, point.z});
        if (!std::isfinite(result.distance) || result.triangle_id < 0 ||
            static_cast<std::size_t>(result.triangle_id) >= indices.size() / 3)
            throw std::runtime_error("Robust mesh primitive returned an invalid distance or face");
        const std::size_t offset = 3 * static_cast<std::size_t>(result.triangle_id);
        const glm::dvec3 p(point), a(vertices[indices[offset]]);
        const glm::dvec3 b(vertices[indices[offset + 1]]), c(vertices[indices[offset + 2]]);
        const glm::dvec3 normal = glm::cross(b - a, c - a);
        const double normalLength = glm::length(normal);
        if (!(normalLength > 0.0) || !std::isfinite(normalLength))
            throw std::runtime_error("Robust selected triangle is degenerate");
        const glm::dvec3 unitNormal = normal / normalLength;
        const double planeHeight = glm::dot(p - a, unitNormal);
        const glm::dvec3 projection = p - planeHeight * unitNormal;
        const std::array<double, 3> sides = {
            glm::dot(glm::cross(b - a, projection - a), normal),
            glm::dot(glm::cross(c - b, projection - b), normal),
            glm::dot(glm::cross(a - c, projection - c), normal)};
        const bool inFace = sides[0] >= 0.0 && sides[1] >= 0.0 && sides[2] >= 0.0;
        const bool faceInterior = sides[0] > 0.0 && sides[1] > 0.0 && sides[2] > 0.0;
        glm::dvec3 closest = a;
        double bestSquared = glm::dot(p - a, p - a);
        const auto consider = [&](glm::dvec3 candidate) {
            const glm::dvec3 delta = p - candidate;
            const double squared = glm::dot(delta, delta);
            if (squared < bestSquared) { bestSquared = squared; closest = candidate; }
        };
        if (inFace) consider(projection);
        const auto considerEdge = [&](glm::dvec3 start, glm::dvec3 end) {
            const glm::dvec3 edge = end - start;
            const double t = glm::clamp(glm::dot(p - start, edge) / glm::dot(edge, edge), 0.0, 1.0);
            consider(start + t * edge);
        };
        considerEdge(a, b); considerEdge(b, c); considerEdge(c, a);
        const glm::dvec3 delta = p - closest;
        const double magnitude = glm::length(delta);
        const double scale = std::max({glm::length(p), glm::length(a), glm::length(b), glm::length(c),
                                       glm::length(b - a), glm::length(c - b), glm::length(a - c)});
        const double zeroTolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
        // Strict face interiors have an unambiguous oriented-plane sign. At
        // edges/vertices keep the mesh pseudonormal sign, including signed zero.
        const bool negative = std::signbit(faceInterior ? planeHeight : result.distance);
        const double signedMagnitude = std::copysign(magnitude, negative ? -1.0 : 1.0);
        const bool numericalZero = magnitude <= zeroTolerance;
        const glm::dvec3 worldGradient = numericalZero ? unitNormal : delta / signedMagnitude;
        outValues[0] = static_cast<float>(numericalZero ? std::copysign(0.0, signedMagnitude) : signedMagnitude);
        const glm::vec3 gradient(worldGradient);
        if (!std::isfinite(outValues[0]) || !std::isfinite(gradient.x) ||
            !std::isfinite(gradient.y) || !std::isfinite(gradient.z))
            throw std::runtime_error("Robust raw Hermite value or gradient is nonfinite");

"""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def make_patched_copy(source, destination, node_key_step_m=None, robust_mesh_values=False):
    # The default branch must keep the already-validated patched bytes/hash.
    step = 1e-6 if node_key_step_m is None else float(node_key_step_m)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("node_key_step_m must be finite and positive")
    scaled_key = step != 1e-6
    preamble = PREAMBLE
    insertion = "\n        rawSamples.emplace(sampleKey, outValues);"
    if scaled_key:
        preamble = preamble.replace("constexpr double quantization = 1e-6;",
                                    f"constexpr double quantization = {step:.17g};")
        preamble = preamble.replace("rawSamples;\n", "rawSamples;\n"
                                    "        static std::map<std::array<long long, 3>, glm::vec3> rawSamplePositions;\n")
        preamble = preamble.replace("        const std::array<long long, 3> sampleKey", """        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(static_cast<double>(point[axis]) / quantization) ||
                std::abs(static_cast<double>(point[axis]) / quantization) >= 9.22e18)
                throw std::runtime_error("Raw nodal key is outside int64 range");
        const std::array<long long, 3> sampleKey""")
        preamble = preamble.replace("            outValues = existingSample->second;", """            // A coarse key may share repeated evaluations of one logical node,
            // but must never silently merge distinct adjacent sampling nodes.
            const glm::vec3 firstPoint = rawSamplePositions.at(sampleKey);
            for (int axis = 0; axis < 3; ++axis)
                if (std::abs(static_cast<double>(point[axis]) - firstPoint[axis]) >= 0.25 * quantization)
                    throw std::runtime_error("Raw nodal key collision: distinct sampling nodes; check step and grid alignment");
            outValues = existingSample->second;""")
        insertion += "\n        rawSamplePositions.emplace(sampleKey, point);"
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
    active = active.replace(function_marker, preamble + function_marker, 1)
    if robust_mesh_values:
        gradient_marker = "        outValues[1] = gradient.x; outValues[2] = gradient.y; outValues[3] = gradient.z;"
        if active.count(gradient_marker) != 1:
            raise ValueError("raw gradient output layout changed")
        begin = active.index(function_marker)
        end = active.index(gradient_marker)
        active = active[:begin] + ROBUST_POINT_VALUES + active[end:]
    end_marker = "        outValues[4] = 0.0f; outValues[5] = 0.0f; outValues[6] = 0.0f; outValues[7] = 0.0f;"
    if active.count(end_marker) != 1:
        raise ValueError("raw Hermite output layout changed")
    active = active.replace(end_marker, end_marker + insertion, 1)
    include_marker = "#include <array>"
    if prefix.count(include_marker) != 1:
        raise ValueError("header include layout changed")
    prefix = prefix.replace(include_marker, include_marker +
                            "\n#include <cmath>\n#include <map>\n#include <mutex>\n#include <stdexcept>", 1)
    if robust_mesh_values:
        prefix = prefix.replace(include_marker, include_marker +
                                "\n#include <memory>\n#include <limits>\n#include <InteractiveComputerGraphics/TriangleMeshDistance.h>", 1)
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
    if scaled_key:
        manifest.update({
            "adaptation": "raw-hermite-first-sample-dyadic-step-v2",
            "key": f"llround(double(point_axis) / {step:.17g}), independently for x,y,z",
            "quantization_m": step,
            "intended_bounds_m": None,
            "intended_start_depth": None,
            "intended_max_depth": None,
            "smallest_intended_cell_m": 2 * step,
            "coordinate_scope": "World-coordinate full-map or ROI dyadic grids; actual bounds and depths are recorded by each exporter JSON, not fixed by this patch.",
            "grid_contract": "Set step to the finest raw sampling lattice spacing (half the finest cell). Distinct logical sampling nodes must be at least one step apart on some axis. Same-key points differing by step/4 or more on any axis throw instead of merging.",
            "example_configurations": [
                {"cube_side_m": 8192 * step, "start_depth": 3, "max_depth": 12},
                {"cube_side_m": 512 * step, "start_depth": 5, "max_depth": 8},
            ],
            "limits": [
                "The representative gradient at a medial tie remains arbitrary.",
                "This is not a prescribed smoothing width, a conservative SDF, or a global error guarantee.",
                "Repeated logical nodes near a key rounding boundary can still receive different keys; use an aligned dyadic grid and check seams at the actual world-coordinate scale.",
                "The collision guard prevents merging adjacent logical nodes only under the declared sampling lattice spacing; it cannot infer a wrong maxDepth or step.",
                "Sharing raw Hermite inputs does not prove all adaptive coarse/fine interfaces continuous; rerun full seam checks.",
            ],
        })
    if robust_mesh_values:
        primitive_header = source / "libs/InteractiveComputerGraphics/InteractiveComputerGraphics/TriangleMeshDistance.h"
        manifest.update({
            "adaptation": "raw-hermite-first-sample-robust-triangle-v4",
            "robust_mesh_values": True,
            "raw_value_method": "Full tmd BVH selects a face on the immutable float32 mesh promoted to double. Recompute that triangle's closest point from its inside-plane projection and three clamped edge segments, using direct squared residuals. Ignore tmd distance magnitude/nearest_point and the supplied nearestTriangleIndex.",
            "raw_sign_method": "Strict selected-face interior: signbit(dot(point-a, oriented unit normal)). Edge/vertex projection: retain signbit(tmd signed distance), including negative zero.",
            "raw_gradient_method": "For nonzero recomputed distance, (point-recomputed_closest_point)/signed_norm. Numerical zero uses the selected triangle's oriented unit normal as a shared representative. Mixed Hermite derivatives remain zero.",
            "raw_zero_tolerance": "64 * double epsilon * max(norm(point), norm(each triangle vertex), each edge length); no float-scale or fixed-metre snapping.",
            "raw_primitive_header": str(primitive_header.relative_to(source)),
            "raw_primitive_sha256": _sha256(primitive_header.read_bytes()),
            "raw_primitive_license": "MIT, Copyright (c) 2021 Jose Antonio Fernandez Fernandez; original licensed header is retained in this source copy.",
            "change": "Cache raw Hermite entries as before; on a cache miss query a separate persistent tmd BVH and stably recompute distance/gradient on its selected triangle instead of TriangleUtils region classification or tmd expanded quadratic residuals.",
            "preserved": ["same stored float32 mesh geometry", "octree refinement and topology algorithm", "coarse-to-fine interpolateVertexValues constraints", "tricubic coefficient construction and query evaluation", "raw-node sharing and zero mixed Hermite derivatives"],
            "construction_cost": "An additional CPU BVH is built once per exporter process. Upstream nearest-face queries still run; this adaptation accepts duplicate search work to isolate raw-value correctness.",
        })
        manifest["limits"].extend([
            "Signed-distance semantics require the same closed consistently oriented mesh assumptions as tmd; this does not repair invalid geometry.",
            "Zero-distance and medial-tie gradient representatives are not unique and do not prove global C1 continuity.",
            "Double evaluation still queries float32-stored geometry and is not an exact arithmetic or global interpolation error certificate.",
            "The global nearest face is still selected by the bundled tmd BVH and its distance routine. The stable selected-triangle recomputation cannot certify that selection globally; test near-zero and competing-face cases independently.",
        ])
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
    parser.add_argument("--node-key-step-m", type=float, default=None,
                        help="Raw sampling lattice spacing in metres; omit for byte-identical legacy 1um patch. Full-map/ROI min cell .025m uses .0125m.")
    parser.add_argument("--robust-mesh-values", action="store_true",
                        help="Use bundled tmd BVH face selection plus stable double triangle recomputation; defaults keep existing patched source bytes unchanged.")
    args = parser.parse_args()
    print(json.dumps(make_patched_copy(args.source, args.destination, args.node_key_step_m,
                                      args.robust_mesh_values), indent=2))


if __name__ == "__main__":
    main()
