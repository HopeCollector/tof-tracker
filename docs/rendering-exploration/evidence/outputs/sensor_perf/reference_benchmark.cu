// Temporary performance harness. Build against the unchanged C5 reference
// sensor_simulator.cu and headers mounted read-only. This is not product code.
#include "sensor_simulator.cuh"
#include "tof_renderer.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
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
using Clock = std::chrono::steady_clock;
constexpr double pi = 3.14159265358979323846;

struct Box {
    std::string name;
    std::array<double, 3> low{{INFINITY, INFINITY, INFINITY}};
    std::array<double, 3> high{{-INFINITY, -INFINITY, -INFINITY}};
    int vertices = 0;
};

struct Voxelized {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    std::array<double, 3> size;
    std::size_t boxes = 0, occupied = 0;
};

struct Pose {
    Eigen::Vector3f position;
    Eigen::Quaternionf rotation;
};

void cudaCheck(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

Voxelized loadMap(const std::string& path, double resolution) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open map " + path);
    std::vector<Box> objects;
    std::array<double, 3> size{{0, 0, 0}};
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("# size_xyz ", 0) == 0) {
            std::istringstream dimensions(line.substr(11));
            dimensions >> size[0] >> size[1] >> size[2];
        }
        std::istringstream values(line);
        std::string token;
        values >> token;
        if (token == "o") {
            Box box;
            values >> box.name;
            objects.push_back(box);
        } else if (token == "v") {
            if (objects.empty()) throw std::runtime_error("vertex before OBJ object");
            std::array<double, 3> point;
            if (!(values >> point[0] >> point[1] >> point[2]))
                throw std::runtime_error("invalid OBJ vertex");
            Box& box = objects.back();
            for (int a = 0; a < 3; ++a) {
                if (!std::isfinite(point[a])) throw std::runtime_error("nonfinite OBJ vertex");
                box.low[a] = std::min(box.low[a], point[a]);
                box.high[a] = std::max(box.high[a], point[a]);
            }
            ++box.vertices;
        }
    }
    for (double value : size)
        if (!(value > 0) || !std::isfinite(value))
            throw std::runtime_error("mapgen OBJ must declare finite positive # size_xyz");
    const std::array<double, 3> origin{{-size[0] / 2, -size[1] / 2, 0}};
    std::array<int, 3> dimensions;
    std::size_t cells = 1;
    for (int a = 0; a < 3; ++a) {
        dimensions[a] = static_cast<int>(std::ceil(size[a] / resolution));
        cells *= dimensions[a];
    }
    if (cells > 200000000) throw std::runtime_error("temporary harness voxel limit exceeded");
    std::vector<std::uint8_t> occupied(cells, 0);
    std::size_t box_count = 0;
    // Benchmark adapter only: represent the exact mapgen AABBs by occupied
    // 0.1 m global-grid centers. This deliberately retains voxel discretization.
    for (const Box& box : objects) {
        if (box.name == "ground") continue;
        if (box.vertices != 8) throw std::runtime_error("expected mapgen eight-vertex pillar box");
        ++box_count;
        std::array<int, 3> low, high;
        for (int a = 0; a < 3; ++a) {
            low[a] = std::max(0, static_cast<int>(std::ceil((box.low[a] - origin[a]) / resolution - 0.5)));
            high[a] = std::min(dimensions[a] - 1,
                static_cast<int>(std::floor((box.high[a] - origin[a]) / resolution - 0.5)));
        }
        for (int x = low[0]; x <= high[0]; ++x)
            for (int y = low[1]; y <= high[1]; ++y)
                for (int z = low[2]; z <= high[2]; ++z)
                    occupied[(static_cast<std::size_t>(x) * dimensions[1] + y) * dimensions[2] + z] = 1;
    }
    const std::size_t count = std::count(occupied.begin(), occupied.end(), std::uint8_t(1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(count + 2);
    // Old GridMap infers its domain from cloud extrema. These two occupied
    // corner anchors stabilize identical 60x60x15 bounds even with zero pillars.
    cloud->push_back(pcl::PointXYZ(origin[0], origin[1], origin[2]));
    cloud->push_back(pcl::PointXYZ(origin[0] + size[0], origin[1] + size[1], size[2]));
    for (int x = 0; x < dimensions[0]; ++x)
        for (int y = 0; y < dimensions[1]; ++y)
            for (int z = 0; z < dimensions[2]; ++z)
                if (occupied[(static_cast<std::size_t>(x) * dimensions[1] + y) * dimensions[2] + z])
                    cloud->push_back(pcl::PointXYZ(origin[0] + (x + 0.5) * resolution,
                                                 origin[1] + (y + 0.5) * resolution,
                                                 origin[2] + (z + 0.5) * resolution));
    return {cloud, size, box_count, count};
}

std::vector<Pose> loadPoses(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open poses " + path);
    std::vector<Pose> poses;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream values(line);
        std::array<double, 7> numbers;
        for (double& value : numbers)
            if (!(values >> value) || !std::isfinite(value))
                throw std::runtime_error("pose rows must be x y z qx qy qz qw, finite numbers");
        double norm = std::sqrt(numbers[3]*numbers[3] + numbers[4]*numbers[4] +
                                numbers[5]*numbers[5] + numbers[6]*numbers[6]);
        if (std::abs(norm - 1) > 1e-6) throw std::runtime_error("pose quaternion must be unit length");
        poses.push_back({Eigen::Vector3f(numbers[0], numbers[1], numbers[2]),
                         Eigen::Quaternionf(numbers[6]/norm, numbers[3]/norm,
                                            numbers[4]/norm, numbers[5]/norm)});
    }
    if (poses.empty()) throw std::runtime_error("at least one pose is required");
    return poses;
}

raycast::CameraParams camera(int width, int height, double hfov, double vfov, double maximum) {
    raycast::CameraParams result;
    result.image_width = width;
    result.image_height = height;
    result.cx = 0.5f * (width - 1);
    result.cy = 0.5f * (height - 1);
    result.fx = 0.5 * width / std::tan(hfov * pi / 360);
    result.fy = 0.5 * height / std::tan(vfov * pi / 360);
    result.max_depth_dist = maximum;
    result.normalize_depth = false;
    return result;
}

std::string quoted(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char character : value) {
        if (character == '"' || character == '\\') result << '\\' << character;
        else if (character < 32) result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(character);
        else result << character;
    }
    return result.str() + '"';
}

double percentile(const std::vector<double>& sorted, double fraction) {
    const double index = fraction * (sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(index);
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string map_path, poses_path, mode = "matched_rig";
        int warmup = 20, iterations = 200;
        double tof_hfov = 60, tof_vfov = 60, tof_max = 12, resolution = 0.1;
        for (int i = 1; i < argc; ++i) {
            const std::string key(argv[i]);
            const auto next = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("missing argument after " + key);
                return argv[i];
            };
            if (key == "--map") map_path = next();
            else if (key == "--poses") poses_path = next();
            else if (key == "--mode") mode = next();
            else if (key == "--warmup") warmup = std::stoi(next());
            else if (key == "--iterations") iterations = std::stoi(next());
            else if (key == "--tof-hfov") tof_hfov = std::stod(next());
            else if (key == "--tof-vfov") tof_vfov = std::stod(next());
            else if (key == "--tof-max") tof_max = std::stod(next());
            else throw std::runtime_error("unknown option " + key);
        }
        if (map_path.empty() || poses_path.empty() || warmup < 1 || iterations < 1 ||
            !(tof_hfov > 0 && tof_hfov < 179 && tof_vfov > 0 && tof_vfov < 179 &&
              tof_max > 0 && std::isfinite(tof_max)))
            throw std::runtime_error("required: --map MAP.obj --poses POSES.txt [--mode matched_rig|subray_rig] [--warmup N] [--iterations N]");
        if (mode != "matched_rig" && mode != "subray_rig") throw std::runtime_error("invalid --mode");
        const auto poses = loadPoses(poses_path);
        const auto begin = Clock::now();
        Voxelized map = loadMap(map_path, resolution);
        const auto voxelized = Clock::now();
        int device = 0;
        cudaCheck(cudaGetDevice(&device), "get device");
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties, device), "device properties");
        raycast::GridMap grid(map.cloud, resolution, 0);
        cudaCheck(cudaGetLastError(), "construct GridMap");
        const auto uploaded = Clock::now();
        raycast::CameraParams tof = camera(8, 8, tof_hfov, tof_vfov, tof_max);
        raycast::CameraParams rgb = camera(160, 120, 80, 60, 30);
        std::default_random_engine generator(0);
        std::vector<double> times;
        double checksum = 0;
        std::size_t validated_pixels = 0;
        float observed_min = std::numeric_limits<float>::infinity();
        float observed_max = -std::numeric_limits<float>::infinity();
        const std::array<float, 4> yaws{{-float(pi / 2), 0, float(pi / 2), float(pi)}};
        for (int index = -warmup; index < iterations; ++index) {
            const std::size_t pose_index = static_cast<std::size_t>(index >= 0 ? index : index + warmup) % poses.size();
            const Pose& body = poses[pose_index];
            std::array<cv::Mat, 5> images;
            const auto start = Clock::now();
            for (std::size_t view = 0; view < yaws.size(); ++view) {
                const Eigen::Quaternionf rotation = body.rotation * Eigen::Quaternionf(Eigen::AngleAxisf(yaws[view], Eigen::Vector3f::UnitZ()));
                cudaMat::SE3<float> transform(rotation.w(), rotation.x(), rotation.y(), rotation.z(),
                                              body.position.x(), body.position.y(), body.position.z());
                if (mode == "matched_rig")
                    raycast::renderDepthImage(&grid, &tof, transform, images[view]);
                else
                    renderToFSenseMImage(&grid, tof, transform, 4, 0.2f, 0.0f, 0.0f,
                                        0.08f, 0.015f, generator, images[view]);
            }
            if (mode == "matched_rig") {
                cudaMat::SE3<float> transform(body.rotation.w(), body.rotation.x(), body.rotation.y(), body.rotation.z(),
                                              body.position.x(), body.position.y(), body.position.z());
                raycast::renderDepthImage(&grid, &rgb, transform, images[4]);
            }
            const auto stop = Clock::now();
            cudaCheck(cudaGetLastError(), "render iteration");
            cudaCheck(cudaDeviceSynchronize(), "validate completed render iteration");
            // Validate every returned value outside the timed interval. Original
            // renderDepthImage synchronizes and copies each output to CPU.
            for (std::size_t view = 0; view < (mode == "matched_rig" ? 5 : 4); ++view) {
                const cv::Mat& image = images[view];
                const int width = view == 4 ? 160 : 8;
                const int height = view == 4 ? 120 : 8;
                if (image.type() != CV_32FC1 || image.cols != width || image.rows != height)
                    throw std::runtime_error("reference renderer returned incorrect image shape/type");
                for (int row = 0; row < image.rows; ++row) {
                    const float* values = image.ptr<float>(row);
                    for (int col = 0; col < image.cols; ++col) {
                        if (!std::isfinite(values[col]))
                            throw std::runtime_error("reference renderer returned a nonfinite depth");
                        observed_min = std::min(observed_min, values[col]);
                        observed_max = std::max(observed_max, values[col]);
                        checksum += values[col];
                        ++validated_pixels;
                    }
                }
            }
            if (index >= 0) times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
        grid.freeGridMap();
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        double mean = 0;
        for (double value : sorted) mean += value / sorted.size();
        std::cout << std::setprecision(12)
            << "{\"reference_commit\":\"ab9cbfaef8216b8a47a952b40d199fd7e5cef134\","
            << "\"mode\":" << quoted(mode) << ",\"map\":" << quoted(map_path)
            << ",\"gpu\":" << quoted(properties.name) << ",\"compute_capability\":" << quoted(std::to_string(properties.major) + "." + std::to_string(properties.minor))
            << ",\"map_size_xyz_m\":[" << map.size[0] << ',' << map.size[1] << ',' << map.size[2] << ']'
            << ",\"pillar_count\":" << map.boxes << ",\"cloud_points\":" << map.cloud->size()
            << ",\"occupied_voxel_centers\":" << map.occupied << ",\"resolution_m\":0.1,\"occupancy_threshold\":0"
            << ",\"poses\":" << poses.size() << ",\"warmup\":" << warmup << ",\"iterations\":" << iterations
            << ",\"tof_size\":[8,8],\"tof_fov_deg\":[" << tof_hfov << ',' << tof_vfov << "],\"tof_max_axis_depth_m\":" << tof_max
            << ",\"camera_depth\":" << (mode == "matched_rig" ? "{\"size\":[160,120],\"fov_deg\":[80,60],\"max_axis_depth_m\":30}" : "null")
            << ",\"kernel_launches_per_iteration\":" << (mode == "matched_rig" ? 5 : 4)
            << ",\"rays_per_iteration\":" << (mode == "matched_rig" ? 4*8*8 + 160*120 : 4*32*32)
            << ",\"prepare_seconds\":{\"obj_to_cloud\":" << std::chrono::duration<double>(voxelized - begin).count()
            << ",\"grid_construct_and_upload\":" << std::chrono::duration<double>(uploaded - voxelized).count() << '}'
            << ",\"latency_ms\":{\"mean\":" << mean << ",\"p50\":" << percentile(sorted, 0.5)
            << ",\"p95\":" << percentile(sorted, 0.95) << ",\"min\":" << sorted.front() << ",\"max\":" << sorted.back() << '}'
            << ",\"checksum\":" << checksum
            << ",\"output_validation\":{\"all_finite\":true,\"pixels_including_warmup\":" << validated_pixels
            << ",\"observed_min_axis_depth_m\":" << observed_min << ",\"observed_max_axis_depth_m\":" << observed_max << '}'
            << ",\"limitations\":[\"matched means image dimensions and API call counts, not identical geometric semantics\",\"optical-axis depth and cutoff, not Euclidean range: oblique reference rays travel farther at the same numeric max\",\"voxelized background with legacy XY mirror and ground semantics\",\"two occupied corner points anchor the declared domain\",\"no target geometry, mask, visibility, hint, coverage checks or RGB\",\"original synchronous allocations, launches, readback and CPU aggregation retained\"]"
            << ",\"samples_ms\":[";
        for (std::size_t i = 0; i < times.size(); ++i) std::cout << (i ? "," : "") << times[i];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reference benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
