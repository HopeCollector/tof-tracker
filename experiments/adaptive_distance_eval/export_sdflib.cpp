// Experiment adapter: build SdfLib's actual octree and export its public data.
// No implementation or private member of the upstream library is replaced.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <SdfLib/OctreeSdf.h>
#include <SdfLib/InterpolationMethods.h>
#include <SdfLib/utils/Mesh.h>

namespace {
using Sdf = sdflib::OctreeSdf;
using Interpolation = sdflib::TriCubicInterpolation;
using Coefficients = std::array<float, 64>;
static_assert(Interpolation::NUM_COEFFICIENTS == 64, "Tricubic coefficient layout changed");
static_assert(sizeof(Sdf::OctreeNode) == sizeof(float), "Packed upstream tree layout changed");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "The export format requires IEEE-754 binary32");
constexpr std::uint32_t kExactFloatIntegerLimit = 1u << 24;

struct Options {
    std::string mesh_path;
    std::string prefix;
    std::uint32_t start_depth;
    std::uint32_t max_depth;
    float error;
    bool continuity = true;
    glm::vec3 lo{-0.8f};
    glm::vec3 hi{0.8f};
    std::uint32_t query_count = 4096;
    std::uint32_t threads = 1;
    bool morton_index = false;
};

std::uint32_t unsigned_arg(const std::string& value) {
    std::size_t consumed = 0;
    const auto result = std::stoull(value, &consumed);
    if (value.empty() || value.front() == '-' || consumed != value.size() ||
        result > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Invalid unsigned integer: " + value);
    }
    return static_cast<std::uint32_t>(result);
}

float float_arg(const std::string& value) {
    std::size_t consumed = 0;
    const float result = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result)) {
        throw std::runtime_error("Invalid finite float: " + value);
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    if (argc < 6) {
        throw std::runtime_error(
            "Usage: export_sdflib mesh.obj output_prefix startDepth maxDepth error "
            "[continuity=1|0] [--bounds minx miny minz maxx maxy maxz] "
            "[--query-count N] [--threads N] [--morton-index]");
    }
    Options options{argv[1], argv[2], unsigned_arg(argv[3]), unsigned_arg(argv[4]), float_arg(argv[5])};
    int argument = 6;
    if (argument < argc && argv[argument][0] != '-') {
        std::string mode = argv[argument++];
        if (mode.rfind("continuity=", 0) == 0) mode.erase(0, 11);
        if (mode != "0" && mode != "1") throw std::runtime_error("Continuity must be 0 or 1");
        options.continuity = mode == "1";
    }
    while (argument < argc) {
        const std::string option = argv[argument++];
        if (option == "--bounds" && argument + 6 <= argc) {
            for (int axis = 0; axis < 3; ++axis) options.lo[axis] = float_arg(argv[argument++]);
            for (int axis = 0; axis < 3; ++axis) options.hi[axis] = float_arg(argv[argument++]);
        } else if (option == "--query-count" && argument < argc) {
            options.query_count = unsigned_arg(argv[argument++]);
        } else if (option == "--threads" && argument < argc) {
            options.threads = unsigned_arg(argv[argument++]);
        } else if (option == "--morton-index") {
            options.morton_index = true;
        } else {
            throw std::runtime_error("Unknown option or missing arguments: " + option);
        }
    }
    const glm::vec3 extent = options.hi - options.lo;
    if (!(extent.x > 0.0f) || !std::isfinite(extent.x) ||
        std::abs(extent.y - extent.x) > 1e-6f * extent.x ||
        std::abs(extent.z - extent.x) > 1e-6f * extent.x) {
        throw std::runtime_error("Bounds must describe a finite positive cube");
    }
    if (options.start_depth > 10 || options.max_depth < options.start_depth ||
        options.max_depth > 30 || options.error < 0.0f || options.threads == 0) {
        throw std::runtime_error("Require startDepth <= 10, startDepth <= maxDepth <= 30, error >= 0, threads > 0");
    }
    if (options.morton_index && options.max_depth > 20) {
        throw std::runtime_error("Morton uint64 export requires maxDepth <= 20");
    }
    return options;
}

sdflib::Mesh load_triangle_obj(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open mesh: " + path);
    std::vector<glm::vec3> vertices;
    std::vector<std::uint32_t> indices;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream words(line);
        std::string kind;
        words >> kind;
        if (kind == "v") {
            glm::vec3 point;
            if (!(words >> point.x >> point.y >> point.z) ||
                !std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                throw std::runtime_error("Invalid OBJ vertex at line " + std::to_string(line_number));
            }
            vertices.push_back(point);
        } else if (kind == "f") {
            std::vector<std::uint32_t> face;
            std::string token;
            while (words >> token && token.front() != '#') {
                const std::string index_token = token.substr(0, token.find('/'));
                std::size_t consumed = 0;
                const auto obj_index = std::stoll(index_token, &consumed);
                const auto index = obj_index > 0 ? obj_index - 1 :
                                   static_cast<std::int64_t>(vertices.size()) + obj_index;
                if (consumed != index_token.size() || obj_index == 0 || index < 0 ||
                    static_cast<std::uint64_t>(index) >= vertices.size()) {
                    throw std::runtime_error("Invalid OBJ index at line " + std::to_string(line_number));
                }
                face.push_back(static_cast<std::uint32_t>(index));
            }
            if (face.size() != 3) {
                throw std::runtime_error("Only triangulated OBJ faces are accepted, line " + std::to_string(line_number));
            }
            const glm::dvec3 edge_a = glm::dvec3(vertices[face[1]]) - glm::dvec3(vertices[face[0]]);
            const glm::dvec3 edge_b = glm::dvec3(vertices[face[2]]) - glm::dvec3(vertices[face[0]]);
            if (!(glm::length(glm::cross(edge_a, edge_b)) > 0.0)) {
                throw std::runtime_error("Degenerate OBJ triangle at line " + std::to_string(line_number));
            }
            indices.insert(indices.end(), face.begin(), face.end());
        }
    }
    if (vertices.empty() || indices.empty() ||
        vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
        indices.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Mesh must be nonempty and fit the SdfLib uint32 index API");
    }
    sdflib::Mesh mesh(vertices.data(), static_cast<std::uint32_t>(vertices.size()),
                     indices.data(), static_cast<std::uint32_t>(indices.size()));
    mesh.computeBoundingBox();
    return mesh;
}

struct Leaf {
    glm::vec3 lo;
    float size;
    std::uint32_t depth;
    std::uint32_t node_index;
    Coefficients coefficients;
    glm::uvec3 position;
};

Coefficients read_coefficients(const Sdf& sdf, std::uint32_t node_index) {
    const auto& data = sdf.getOctreeData();
    const auto offset = data.at(node_index).getChildrenIndex();
    if (static_cast<std::uint64_t>(offset) + 64 > data.size()) {
        throw std::runtime_error("Leaf coefficient pointer is outside the public tree array");
    }
    Coefficients result;
    // Copy object bytes rather than aliasing the union array as std::array<float>.
    std::memcpy(result.data(), &data[offset], sizeof(result));
    return result;
}

std::vector<Leaf> collect_leaves(const Sdf& sdf, std::uint32_t start_depth, bool morton_index) {
    const auto& data = sdf.getOctreeData();
    const auto& bounds = sdf.getGridBoundingBox();
    const auto grid = sdf.getStartGridSize();
    if (morton_index && data.size() > (std::uint64_t{1} << 30)) {
        throw std::runtime_error("Upstream packed tree exceeds its 30-bit address capacity");
    }
    std::vector<Leaf> leaves;
    std::function<void(std::uint32_t, std::uint32_t, glm::uvec3)> visit;
    visit = [&](std::uint32_t node_index, std::uint32_t depth, glm::uvec3 position) {
        const auto& node = data.at(node_index);
        if (depth > sdf.getOctreeMaxDepth()) throw std::runtime_error("Tree exceeds declared maxDepth");
        if (node.isLeaf()) {
            if (!morton_index && (node_index >= kExactFloatIntegerLimit || leaves.size() >= kExactFloatIntegerLimit)) {
                throw std::runtime_error("Float32 row format cannot represent this tree's ids exactly");
            }
            if (leaves.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("Leaf rows exceed uint32 capacity");
            }
            const double size = std::ldexp(static_cast<double>(bounds.getSize().x), -static_cast<int>(depth));
            // Integer dyadic coordinates give shared planes identical float32 lo.
            const glm::vec3 lo(glm::dvec3(bounds.min) + glm::dvec3(position) * size);
            leaves.push_back({lo, static_cast<float>(size), depth, node_index,
                              read_coefficients(sdf, node_index), position});
            return;
        }
        const auto children = node.getChildrenIndex();
        if (static_cast<std::uint64_t>(children) + 8 > data.size()) {
            throw std::runtime_error("Child pointer is outside the public tree array");
        }
        for (std::uint32_t child = 0; child < 8; ++child) {
            visit(children + child, depth + 1,
                  position * 2u + glm::uvec3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u));
        }
    };
    for (int z = 0; z < grid.z; ++z) {
        for (int y = 0; y < grid.y; ++y) {
            for (int x = 0; x < grid.x; ++x) {
                visit(static_cast<std::uint32_t>((z * grid.y + y) * grid.x + x),
                      start_depth, glm::uvec3(x, y, z));
            }
        }
    }
    return leaves;
}

std::uint64_t morton_code(glm::uvec3 position, std::uint32_t depth) {
    std::uint64_t code = 0;
    for (std::uint32_t bit = 0; bit < depth; ++bit) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            code |= std::uint64_t{(position[axis] >> bit) & 1u} << (3 * bit + axis);
        }
    }
    return code;
}

template<typename T>
void write_integer(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("Failed writing integer export");
}

void export_morton_index(const Options& options, const std::vector<Leaf>& leaves) {
    struct Interval { std::uint64_t start, end; std::uint32_t row; };
    std::vector<Interval> intervals;
    intervals.reserve(leaves.size());
    std::ofstream nodes(options.prefix + ".node_indices.bin", std::ios::binary);
    for (std::size_t row = 0; row < leaves.size(); ++row) {
        const Leaf& leaf = leaves[row];
        const auto shift = 3 * (options.max_depth - leaf.depth);
        const auto start = morton_code(leaf.position, leaf.depth) << shift;
        intervals.push_back({start, start + (std::uint64_t{1} << shift), static_cast<std::uint32_t>(row)});
        write_integer(nodes, leaf.node_index);
    }
    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) { return a.start < b.start; });
    std::ofstream keys(options.prefix + ".morton.bin", std::ios::binary);
    std::ofstream rows(options.prefix + ".morton_leaf_ids.bin", std::ios::binary);
    std::uint64_t previous_end = 0;
    for (const auto& interval : intervals) {
        if (interval.start != previous_end || interval.end <= interval.start) {
            throw std::runtime_error("Morton leaf intervals overlap or leave a domain gap");
        }
        write_integer(keys, interval.start);
        write_integer(keys, interval.end);
        write_integer(rows, interval.row);
        previous_end = interval.end;
    }
    if (previous_end != (std::uint64_t{1} << (3 * options.max_depth))) {
        throw std::runtime_error("Morton leaf intervals do not cover the complete cube");
    }
    nodes.close(); keys.close(); rows.close();
    if (!nodes || !keys || !rows) throw std::runtime_error("Failed closing Morton export files");
}

struct Location {
    std::uint32_t node_index;
    glm::vec3 local;
    float size;
};

Location locate(const Sdf& sdf, glm::vec3 point) {
    const auto& data = sdf.getOctreeData();
    const auto grid = sdf.getStartGridSize();
    float size = sdf.getGridBoundingBox().getSize().x / static_cast<float>(grid.x);
    glm::vec3 local = (point - sdf.getGridBoundingBox().min) / size;
    const glm::ivec3 root(glm::floor(local));
    if (root.x < 0 || root.y < 0 || root.z < 0 || root.x >= grid.x || root.y >= grid.y || root.z >= grid.z) {
        throw std::runtime_error("Native comparison queries must be strictly inside the exported domain");
    }
    local = glm::fract(local);
    std::uint32_t node_index = (root.z * grid.y + root.y) * grid.x + root.x;
    while (!data.at(node_index).isLeaf()) {
        const std::uint32_t child = (local.x >= 0.5f ? 1u : 0u) +
                                    (local.y >= 0.5f ? 2u : 0u) +
                                    (local.z >= 0.5f ? 4u : 0u);
        node_index = data.at(node_index).getChildrenIndex() + child;
        local = glm::fract(2.0f * local);
        size *= 0.5f;
    }
    return {node_index, local, size};
}

double radical_inverse(std::uint32_t index, std::uint32_t base) {
    double value = 0.0;
    double weight = 1.0 / base;
    while (index) {
        value += (index % base) * weight;
        index /= base;
        weight /= base;
    }
    return value;
}

template<std::size_t Columns>
void write_row(std::ofstream& stream, const std::array<float, Columns>& row) {
    stream.write(reinterpret_cast<const char*>(row.data()), sizeof(row));
    if (!stream) throw std::runtime_error("Failed writing binary export");
}

bool finite(glm::vec3 vector) {
    return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') output << '\\' << character;
        else if (character < 0x20) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character) << std::dec;
        else output << character;
    }
    output << '"';
    return output.str();
}

void write_vector(std::ostream& output, glm::vec3 vector) {
    output << '[' << vector.x << ',' << vector.y << ',' << vector.z << ']';
}

void export_result(const Options& options, const sdflib::Mesh& mesh, const Sdf& sdf, double build_seconds) {
    const auto leaves = collect_leaves(sdf, options.start_depth, options.morton_index);
    std::unordered_map<std::uint32_t, std::uint32_t> leaf_rows;
    const auto directory = std::filesystem::path(options.prefix).parent_path();
    if (!directory.empty()) std::filesystem::create_directories(directory);
    if (options.morton_index) export_morton_index(options, leaves);
    std::ofstream leaf_file(options.prefix + ".leaves.bin", std::ios::binary);
    std::ofstream query_file(options.prefix + ".queries.bin", std::ios::binary);
    std::ofstream query_ids;
    if (options.morton_index) query_ids.open(options.prefix + ".query_leaf_ids.bin", std::ios::binary);
    if (!leaf_file || !query_file) throw std::runtime_error("Cannot create output files");
    std::size_t nonfinite_coefficients = 0;
    std::vector<std::size_t> depth_counts(options.max_depth + 1, 0);
    for (std::size_t index = 0; index < leaves.size(); ++index) {
        const Leaf& leaf = leaves[index];
        leaf_rows.emplace(leaf.node_index, static_cast<std::uint32_t>(index));
        std::array<float, 70> row{leaf.lo.x, leaf.lo.y, leaf.lo.z, leaf.size,
                                  static_cast<float>(leaf.depth), options.morton_index ? -1.0f : static_cast<float>(leaf.node_index)};
        std::copy(leaf.coefficients.begin(), leaf.coefficients.end(), row.begin() + 6);
        for (float coefficient : leaf.coefficients) nonfinite_coefficients += !std::isfinite(coefficient);
        ++depth_counts[leaf.depth];
        write_row(leaf_file, row);
    }
    const auto& bounds = sdf.getGridBoundingBox();
    std::size_t nonfinite_native_gradients = 0, nonfinite_raw_gradients = 0, nonfinite_values = 0;
    for (std::uint32_t index = 0; index < options.query_count; ++index) {
        // The same strict-interior Halton points are used for all tree settings.
        glm::vec3 point;
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint32_t base = axis == 0 ? 2 : (axis == 1 ? 3 : 5);
            const double unit = 0.0001 + 0.9998 * radical_inverse(index + 1, base);
            point[axis] = static_cast<float>(bounds.min[axis] + unit * static_cast<double>(bounds.getSize()[axis]));
        }
        const Location location = locate(sdf, point);
        const auto row_index = leaf_rows.at(location.node_index);
        if (options.morton_index) write_integer(query_ids, row_index);
        glm::vec3 native_gradient;
        const float distance = sdf.getDistance(point, native_gradient);
        const glm::vec3 raw_gradient = Interpolation::interpolateGradient(
            leaves[row_index].coefficients, location.local) / location.size;
        nonfinite_native_gradients += !finite(native_gradient);
        nonfinite_raw_gradients += !finite(raw_gradient);
        nonfinite_values += !std::isfinite(distance);
        write_row(query_file, std::array<float, 11>{point.x, point.y, point.z, distance,
            native_gradient.x, native_gradient.y, native_gradient.z,
            raw_gradient.x, raw_gradient.y, raw_gradient.z,
            row_index < kExactFloatIntegerLimit ? static_cast<float>(row_index) : -1.0f});
    }
    leaf_file.close();
    query_file.close();
    if (options.morton_index) {
        query_ids.close();
        if (!query_ids) throw std::runtime_error("Failed closing integer query leaf ids");
    }
    if (!leaf_file || !query_file) throw std::runtime_error("Failed closing binary export files");

    std::ofstream metadata(options.prefix + ".json");
    if (!metadata) throw std::runtime_error("Cannot create metadata file");
    metadata << std::setprecision(std::numeric_limits<double>::max_digits10);
    metadata << "{\n  \"format_version\": " << (options.morton_index ? 2 : 1)
             << ",\n  \"upstream_revision_expected\": " << json_string(SDFLIB_EXPECTED_REVISION)
             << ",\n  \"octree_source_sha256\": " << json_string(SDFLIB_OCTREE_SOURCE_SHA256)
             << ",\n  \"interpolation_source_sha256\": " << json_string(SDFLIB_INTERPOLATION_SOURCE_SHA256)
             << ",\n  \"mesh_path\": " << json_string(options.mesh_path)
             << ",\n  \"vertex_count\": " << mesh.getVertices().size()
             << ",\n  \"triangle_count\": " << mesh.getIndices().size() / 3
             << ",\n  \"bounds\": [";
    write_vector(metadata, bounds.min); metadata << ','; write_vector(metadata, bounds.max);
    metadata << "],\n  \"start_depth\": " << options.start_depth
             << ",\n  \"max_depth\": " << options.max_depth
             << ",\n  \"start_cell_size\": " << bounds.getSize().x / sdf.getStartGridSize().x
             << ",\n  \"termination_error\": " << options.error
             << ",\n  \"termination_rule\": \"TRAPEZOIDAL_RULE\",\n  \"initialization\": "
             << json_string(options.continuity ? "CONTINUITY" : "NO_CONTINUITY")
             << ",\n  \"threads\": " << options.threads
             << ",\n  \"build_seconds\": " << build_seconds
             << ",\n  \"leaf_count\": " << leaves.size()
             << ",\n  \"tree_storage_words\": " << sdf.getOctreeData().size()
             << ",\n  \"leaves_per_depth\": [";
    for (std::size_t depth = 0; depth < depth_counts.size(); ++depth) {
        if (depth) metadata << ',';
        metadata << depth_counts[depth];
    }
    metadata << ']';
    if (options.morton_index) {
        metadata << ",\n  \"morton_index\": {\"depth\":" << options.max_depth
                 << ",\"interval_dtype\":\"<u8\",\"interval_columns\":2,\"interval_suffix\":\".morton.bin\","
                 << "\"interval_semantics\":\"sorted [start,end), complete cube, x bit first then y,z\","
                 << "\"leaf_ids_dtype\":\"<u4\",\"leaf_ids_suffix\":\".morton_leaf_ids.bin\","
                 << "\"leaf_ids_semantics\":\"interval row to original leaves.bin row\"},"
                 << "\n  \"node_indices_dtype\":\"<u4\",\"node_indices_suffix\":\".node_indices.bin\","
                 << "\n  \"node_indices_semantics\":\"original leaf row order; float leaf column 5 is always -1\","
                 << "\n  \"query_leaf_ids_dtype\":\"<u4\",\"query_leaf_ids_suffix\":\".query_leaf_ids.bin\","
                 << "\n  \"query_leaf_ids_semantics\":\"exact original leaf row per native query; float query column 10 is -1 if not exactly representable\"";
    }
    metadata << ",\n  \"interpolation\": \"TriCubicInterpolation\","
             << "\n  \"coefficient_order\": \"x_fastest: i+4*j+16*k\","
             << "\n  \"polynomial\": \"sum c[i+4*j+16*k] * u_x^i * u_y^j * u_z^k, u=(point-lo)/size, i,j,k=0..3\","
             << "\n  \"dtype\": \"<f4\",\n  \"leaf_columns\": 70,"
             << "\n  \"leaf_layout\": [\"lo_x\",\"lo_y\",\"lo_z\",\"size\",\"depth\",\"node_index\",\"coefficient[0:64]\"],"
             << "\n  \"leaf_order\": \"start grid z,y,x then depth first child x+2*y+4*z\","
             << "\n  \"query_count\": " << options.query_count
             << ",\n  \"query_columns\": 11,"
             << "\n  \"query_layout\": [\"x\",\"y\",\"z\",\"native_value\",\"native_normalized_gradient_x\",\"native_normalized_gradient_y\",\"native_normalized_gradient_z\",\"raw_world_gradient_x\",\"raw_world_gradient_y\",\"raw_world_gradient_z\",\"leaf_row_index\"],"
             << "\n  \"query_points\": \"Halton bases 2,3,5, indices 1..query_count, u=0.0001+0.9998*Halton; strictly inside bounds\","
             << "\n  \"native_gradient_semantics\": \"glm::normalize(local polynomial gradient); may be nonfinite at zero gradient\","
             << "\n  \"raw_gradient_semantics\": \"TriCubicInterpolation::interpolateGradient at native traversal local coordinates divided by leaf size\","
             << "\n  \"nonfinite_coefficients\": " << nonfinite_coefficients
             << ",\n  \"nonfinite_values\": " << nonfinite_values
             << ",\n  \"nonfinite_native_gradient_rows\": " << nonfinite_native_gradients
             << ",\n  \"nonfinite_raw_gradient_rows\": " << nonfinite_raw_gradients << "\n}\n";
    metadata.close();
    if (!metadata) throw std::runtime_error("Failed writing metadata");
    std::cout << "Exported " << leaves.size() << " leaves and " << options.query_count
              << " queries to " << options.prefix << ".{json,leaves.bin,queries.bin}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t one = 1;
        if (*reinterpret_cast<const unsigned char*>(&one) != 1) {
            throw std::runtime_error("This minimal exporter supports little-endian machines only");
        }
        const Options options = parse_options(argc, argv);
        const sdflib::Mesh mesh = load_triangle_obj(options.mesh_path);
        const auto begin = std::chrono::steady_clock::now();
        const Sdf sdf(mesh, sdflib::BoundingBox(options.lo, options.hi),
                      options.max_depth, options.start_depth, options.error,
                      options.continuity ? Sdf::CONTINUITY : Sdf::NO_CONTINUITY, options.threads);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        export_result(options, mesh, sdf, seconds);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "export_sdflib: " << error.what() << '\n';
        return 1;
    }
}
