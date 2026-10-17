// Evaluate the actual compiled raw Hermite primitive, before octree fitting.
// Input mesh is the triangulated OBJ produced by real_map_geometry.py.
#include <SdfLib/InterpolationMethods.h>
#include <SdfLib/utils/Mesh.h>
#include <InteractiveComputerGraphics/TriangleMeshDistance.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
sdflib::Mesh read_union_obj(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read union OBJ");
    std::vector<glm::vec3> vertices;
    std::vector<std::uint32_t> indices;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream words(line);
        std::string type;
        words >> type;
        if (type == "v") {
            glm::vec3 v;
            if (!(words >> v.x >> v.y >> v.z) || !std::isfinite(v.x) ||
                !std::isfinite(v.y) || !std::isfinite(v.z))
                throw std::runtime_error("Invalid OBJ vertex");
            vertices.push_back(v);
        } else if (type == "f") {
            std::array<std::uint64_t, 3> face;
            std::string extra;
            if (!(words >> face[0] >> face[1] >> face[2]) || (words >> extra))
                throw std::runtime_error("Expected a plain three-index union OBJ face");
            for (const auto index : face) {
                if (index == 0 || index > vertices.size() || index - 1 > INT32_MAX)
                    throw std::runtime_error("OBJ face index is out of range");
                indices.push_back(static_cast<std::uint32_t>(index - 1));
            }
            const glm::dvec3 a(vertices[face[0] - 1]), b(vertices[face[1] - 1]), c(vertices[face[2] - 1]);
            if (!(glm::length(glm::cross(b - a, c - a)) > 0.0))
                throw std::runtime_error("Degenerate float32 mesh triangle");
        }
    }
    if (vertices.empty() || indices.empty() || vertices.size() > UINT32_MAX || indices.size() > UINT32_MAX)
        throw std::runtime_error("Empty or oversized mesh");
    sdflib::Mesh mesh(vertices.data(), static_cast<std::uint32_t>(vertices.size()),
                     indices.data(), static_cast<std::uint32_t>(indices.size()));
    mesh.computeBoundingBox();
    return mesh;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: mesh_value_probe union.obj points.txt output.txt");
        const auto mesh = read_union_obj(argv[1]);
        const auto data = sdflib::TriangleUtils::calculateMeshTriangleData(mesh);
        std::vector<std::array<double, 3>> vertices;
        std::vector<std::array<int, 3>> faces;
        for (const auto& v : mesh.getVertices()) vertices.push_back({v.x, v.y, v.z});
        const auto& indices = mesh.getIndices();
        for (std::size_t i = 0; i < indices.size(); i += 3)
            faces.push_back({static_cast<int>(indices[i]), static_cast<int>(indices[i + 1]), static_cast<int>(indices[i + 2])});
        const tmd::TriangleMeshDistance selector(vertices, faces);
        std::ifstream input(argv[2]);
        std::ofstream output(argv[3]);
        if (!input || !output) throw std::runtime_error("Cannot open point input or result output");
        output << "# interpolation_source_sha256 " << SDFLIB_INTERPOLATION_SOURCE_SHA256 << '\n'
               << "# x y z face_id tmd_distance tmd_nearest_x tmd_nearest_y tmd_nearest_z raw_distance raw_gx raw_gy raw_gz\n"
               << std::setprecision(std::numeric_limits<double>::max_digits10);
        double x, y, z;
        while (input >> x) {
            if (!(input >> y >> z)) throw std::runtime_error("Expected point triples");
            const glm::vec3 p(x, y, z);
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                throw std::runtime_error("Probe points must be finite float32-representable coordinates");
            const auto selected = selector.signed_distance(std::array<double, 3>{p.x, p.y, p.z});
            if (selected.triangle_id < 0 || static_cast<std::size_t>(selected.triangle_id) >= faces.size())
                throw std::runtime_error("Invalid selected face");
            std::array<float, sdflib::TriCubicInterpolation::VALUES_PER_VERTEX> raw;
            sdflib::TriCubicInterpolation::calculatePointValues(
                p, static_cast<std::uint32_t>(selected.triangle_id), mesh, data, raw);
            output << p.x << ' ' << p.y << ' ' << p.z << ' ' << selected.triangle_id << ' ' << selected.distance;
            for (int axis = 0; axis < 3; ++axis) output << ' ' << selected.nearest_point[axis];
            for (int column = 0; column < 4; ++column) output << ' ' << raw[column];
            output << '\n';
        }
        if (!input.eof()) throw std::runtime_error("Invalid point input");
        output.close();
        if (!output) throw std::runtime_error("Failed writing probe output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mesh_value_probe: " << error.what() << '\n';
        return 1;
    }
}
