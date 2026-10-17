// Isolated benchmark of Omni_ToF-YOPO cf0b7f4b17712667a8bbdb1126f5e914904ece9d.
// Compile with its original sensor_simulator.cu and include/ directory.
#include "sensor_simulator.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace raycast;

// BEGIN VERBATIM: original test_simulator_cuda.cpp, weightedQuantileDepth and
// renderToFSenseMImage. These are experiment-only reference code, not a sensor
// implementation installed into the project. See README for provenance.
float weightedQuantileDepth(std::vector<std::pair<float, float>> depth_weights, float quantile)
{
    if (depth_weights.empty())
        return 0.0f;
    std::sort(depth_weights.begin(), depth_weights.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    float total_weight = 0.0f;
    for (const auto &item : depth_weights)
        total_weight += item.second;
    if (total_weight <= 1e-6f)
        return depth_weights.front().first;

    float target = std::clamp(quantile, 0.0f, 1.0f) * total_weight;
    float accum = 0.0f;
    for (const auto &item : depth_weights)
    {
        accum += item.second;
        if (accum >= target)
            return item.first;
    }
    return depth_weights.back().first;
}

void renderToFSenseMImage(GridMap *grid_map,
                          const CameraParams &tof_camera,
                          cudaMat::SE3<float> &T_wc,
                          int zone_subsample,
                          float depth_quantile,
                          float noise_std,
                          float far_noise_std,
                          float signal_floor,
                          float min_depth,
                          std::default_random_engine &generator,
                          cv::Mat &tof_image)
{
    const int subsample = std::max(1, zone_subsample);
    CameraParams sub_camera = tof_camera;
    const float tan_half_x = (0.5f * tof_camera.image_width) / tof_camera.fx;
    const float tan_half_y = (0.5f * tof_camera.image_height) / tof_camera.fy;
    sub_camera.image_width = tof_camera.image_width * subsample;
    sub_camera.image_height = tof_camera.image_height * subsample;
    sub_camera.fx = (0.5f * sub_camera.image_width) / tan_half_x;
    sub_camera.fy = (0.5f * sub_camera.image_height) / tan_half_y;
    sub_camera.cx = 0.5f * (sub_camera.image_width - 1);
    sub_camera.cy = 0.5f * (sub_camera.image_height - 1);

    cv::Mat sub_depth;
    renderDepthImage(grid_map, &sub_camera, T_wc, sub_depth);

    tof_image.create(tof_camera.image_height, tof_camera.image_width, CV_32FC1);
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f);
    const float max_depth = tof_camera.max_depth_dist;

    for (int zone_v = 0; zone_v < tof_camera.image_height; ++zone_v)
        for (int zone_u = 0; zone_u < tof_camera.image_width; ++zone_u)
        {
            std::vector<std::pair<float, float>> depth_weights;
            depth_weights.reserve(subsample * subsample);
            for (int sv = 0; sv < subsample; ++sv)
                for (int su = 0; su < subsample; ++su)
                {
                    float depth = sub_depth.at<float>(zone_v * subsample + sv, zone_u * subsample + su);
                    depth = std::clamp(depth, min_depth, max_depth);
                    float normalized = depth / max_depth;
                    float signal = depth >= max_depth - 1e-4f
                                       ? signal_floor
                                       : std::max(signal_floor, 1.0f / (depth * depth + 0.05f));
                    signal *= std::max(0.05f, 1.0f - 0.35f * normalized * normalized);
                    depth_weights.emplace_back(depth, signal);
                }

            float depth = weightedQuantileDepth(depth_weights, depth_quantile);
            if (depth < max_depth - 1e-4f)
            {
                float normalized = depth / max_depth;
                float sigma = noise_std + far_noise_std * normalized * normalized;
                depth += sigma * normal_distribution(generator);
            }
            depth = std::clamp(depth, min_depth, max_depth);
            depth = std::round(depth * 1000.0f) / 1000.0f;
            tof_image.at<float>(zone_v, zone_u) = depth;
        }
}
// END VERBATIM.

// Split only at the raw-image / CPU-post boundary. The function body from
// tof_image.create through the loops is the identical original body above.
void postprocessToF(const CameraParams &tof_camera, const cv::Mat &sub_depth,
                    int subsample, float depth_quantile, float noise_std,
                    float far_noise_std, float signal_floor, float min_depth,
                    std::default_random_engine &generator, cv::Mat &tof_image)
{
    tof_image.create(tof_camera.image_height, tof_camera.image_width, CV_32FC1);
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f);
    const float max_depth = tof_camera.max_depth_dist;

    for (int zone_v = 0; zone_v < tof_camera.image_height; ++zone_v)
        for (int zone_u = 0; zone_u < tof_camera.image_width; ++zone_u)
        {
            std::vector<std::pair<float, float>> depth_weights;
            depth_weights.reserve(subsample * subsample);
            for (int sv = 0; sv < subsample; ++sv)
                for (int su = 0; su < subsample; ++su)
                {
                    float depth = sub_depth.at<float>(zone_v * subsample + sv, zone_u * subsample + su);
                    depth = std::clamp(depth, min_depth, max_depth);
                    float normalized = depth / max_depth;
                    float signal = depth >= max_depth - 1e-4f
                                       ? signal_floor
                                       : std::max(signal_floor, 1.0f / (depth * depth + 0.05f));
                    signal *= std::max(0.05f, 1.0f - 0.35f * normalized * normalized);
                    depth_weights.emplace_back(depth, signal);
                }

            float depth = weightedQuantileDepth(depth_weights, depth_quantile);
            if (depth < max_depth - 1e-4f)
            {
                float normalized = depth / max_depth;
                float sigma = noise_std + far_noise_std * normalized * normalized;
                depth += sigma * normal_distribution(generator);
            }
            depth = std::clamp(depth, min_depth, max_depth);
            depth = std::round(depth * 1000.0f) / 1000.0f;
            tof_image.at<float>(zone_v, zone_u) = depth;
        }
}

namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t code, const char *where) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
}
struct Options {
    std::string cloud, poses, output;
    int batch = 1, warmup = 5, repeats = 30;
    float fov = 45, max_range = 4, resolution = .1f;
};
Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("every CLI option needs a value");
        std::string key = argv[i], value = argv[i + 1];
        if (key == "--cloud") o.cloud = value;
        else if (key == "--poses") o.poses = value;
        else if (key == "--output") o.output = value;
        else if (key == "--batch") o.batch = std::stoi(value);
        else if (key == "--warmup") o.warmup = std::stoi(value);
        else if (key == "--repeats") o.repeats = std::stoi(value);
        else if (key == "--fov") o.fov = std::stof(value);
        else if (key == "--max-range") o.max_range = std::stof(value);
        else if (key == "--resolution") o.resolution = std::stof(value);
        else throw std::runtime_error("unknown option " + key);
    }
    if (o.cloud.empty() || o.poses.empty() || o.output.empty() || o.batch < 1 ||
        o.warmup < 0 || o.repeats < 1 || !std::isfinite(o.fov) || o.fov <= 0 ||
        o.fov >= 180 || !std::isfinite(o.max_range) || o.max_range <= .015f ||
        !std::isfinite(o.resolution) || o.resolution <= 0)
        throw std::runtime_error("invalid/missing cloud, poses, output or numeric argument");
    return o;
}
std::vector<float> readFloats(const std::string &path, std::size_t width) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    const std::streamoff bytes = f.tellg();
    if (bytes <= 0 || bytes % (width * sizeof(float)) != 0)
        throw std::runtime_error("invalid float32 row size: " + path);
    std::vector<float> data(static_cast<std::size_t>(bytes) / sizeof(float));
    f.seekg(0); f.read(reinterpret_cast<char *>(data.data()), bytes);
    if (!f) throw std::runtime_error("short binary read: " + path);
    for (float x : data) if (!std::isfinite(x)) throw std::runtime_error("nonfinite input: " + path);
    return data;
}
CameraParams subCamera(const CameraParams &tof) {
    CameraParams sub = tof;
    const float tan_x = .5f * tof.image_width / tof.fx;
    const float tan_y = .5f * tof.image_height / tof.fy;
    sub.image_width = tof.image_width * 4; sub.image_height = tof.image_height * 4;
    sub.fx = .5f * sub.image_width / tan_x; sub.fy = .5f * sub.image_height / tan_y;
    sub.cx = .5f * (sub.image_width - 1); sub.cy = .5f * (sub.image_height - 1);
    return sub;
}
std::string quoted(const std::string &s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : .5 * (values[n / 2 - 1] + values[n / 2]);
}
double percentile95(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(.95 * values.size())) - 1];
}
void series(std::ostream &out, const std::vector<double> &values, int batch) {
    const double med = median(values);
    out << "{\"median_ms\":" << med << ",\"p95_ms\":" << percentile95(values)
        << ",\"views_per_s\":" << batch * 1000.0 / med << ",\"samples_ms\":[";
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << "]}";
}
template<class F> std::vector<double> measure(F &&f, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) f();
    std::vector<double> samples;
    for (int i = 0; i < repeats; ++i) {
        const auto start = Clock::now(); f();
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    return samples;
}
struct ImageStats {
    std::size_t elements = 0, finite = 0, nan = 0, inf = 0;
    double sum = 0; float lo = std::numeric_limits<float>::infinity();
    float hi = -std::numeric_limits<float>::infinity();
    std::uint64_t hash = 1469598103934665603ULL;
    void add(const cv::Mat &image) {
        if (!image.isContinuous()) throw std::runtime_error("expected contiguous output");
        for (std::size_t i = 0; i < image.total(); ++i) {
            const float x = image.ptr<float>()[i]; ++elements;
            if (std::isfinite(x)) { ++finite; sum += x; lo = std::min(lo, x); hi = std::max(hi, x); }
            else if (std::isnan(x)) ++nan; else ++inf;
        }
        for (std::size_t i = 0; i < image.total() * image.elemSize(); ++i) {
            hash ^= image.data[i]; hash *= 1099511628211ULL;
        }
    }
    void write(std::ostream &out) const {
        out << "{\"elements\":" << elements << ",\"finite\":" << finite
            << ",\"nan\":" << nan << ",\"inf\":" << inf << ",\"sum\":" << sum
            << ",\"min\":" << (finite ? std::to_string(lo) : "null")
            << ",\"max\":" << (finite ? std::to_string(hi) : "null")
            << ",\"byte_checksum_u64\":" << quoted(std::to_string(hash)) << '}';
    }
};
}

int main(int argc, char **argv) {
    try {
        const auto o = parse(argc, argv);
        const auto xyz = readFloats(o.cloud, 3), pose_rows = readFloats(o.poses, 7);
        if (pose_rows.size() / 7 < static_cast<std::size_t>(o.batch)) throw std::runtime_error("not enough poses");
        auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
        cloud->points.resize(xyz.size() / 3);
        for (std::size_t i = 0; i < cloud->points.size(); ++i) {
            cloud->points[i].x = xyz[3 * i]; cloud->points[i].y = xyz[3 * i + 1]; cloud->points[i].z = xyz[3 * i + 2];
        }
        std::vector<cudaMat::SE3<float>> poses;
        for (int i = 0; i < o.batch; ++i) {
            const auto *p = pose_rows.data() + 7 * i;
            const float norm2 = p[3]*p[3] + p[4]*p[4] + p[5]*p[5] + p[6]*p[6];
            if (std::abs(norm2 - 1) > 1e-4f) throw std::runtime_error("input quaternion not normalized");
            poses.emplace_back(p[6], p[3], p[4], p[5], p[0], p[1], p[2]);
        }
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
        int runtime = 0, driver = 0; cudaRuntimeGetVersion(&runtime); cudaDriverGetVersion(&driver);
        const auto setup_start = Clock::now();
        GridMap map(cloud, o.resolution, 0);
        check(cudaDeviceSynchronize(), "GridMap initialization");
        const double map_ms = std::chrono::duration<double, std::milli>(Clock::now() - setup_start).count();
        CameraParams tof;
        tof.image_width = 8; tof.image_height = 8;
        tof.fx = tof.fy = 4.0f / std::tan(.5f * o.fov * M_PI / 180.0f);
        tof.cx = tof.cy = 3.5f; tof.max_depth_dist = o.max_range; tof.normalize_depth = false;
        auto sub = subCamera(tof);
        std::vector<cv::Mat> raw(o.batch), full(o.batch), post(o.batch);
        for (int i = 0; i < o.batch; ++i) renderDepthImage(&map, &sub, poses[i], raw[i]);
        check(cudaGetLastError(), "initial raw rendering");

        // Independent split-vs-original check, including RNG progression.
        std::default_random_engine oracle_rng(3), split_rng(3);
        for (int i = 0; i < o.batch; ++i) {
            renderToFSenseMImage(&map, tof, poses[i], 4, .2f, .01f, 0, .08f, .015f, oracle_rng, full[i]);
            postprocessToF(tof, raw[i], 4, .2f, .01f, 0, .08f, .015f, split_rng, post[i]);
            if (full[i].total() != post[i].total() ||
                std::memcmp(full[i].data, post[i].data, full[i].total() * sizeof(float)) != 0)
                throw std::runtime_error("CPU post split does not match original helper");
        }
        if (!(oracle_rng == split_rng)) throw std::runtime_error("CPU post RNG does not match original helper");

        float *device_output = nullptr;
        check(cudaMalloc(&device_output, 32 * 32 * sizeof(float)), "cudaMalloc kernel-only output");
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin), "cudaEventCreate begin"); check(cudaEventCreate(&end), "cudaEventCreate end");
        std::vector<double> kernel_event;
        auto kernel_call = [&] {
            check(cudaEventRecord(begin), "record begin");
            for (int i = 0; i < o.batch; ++i)
                cameraRaycastKernel<<<sub.image_height, sub.image_width>>>(device_output, map, sub, poses[i]);
            check(cudaEventRecord(end), "record end"); check(cudaEventSynchronize(end), "synchronize kernel");
            float ms = 0; check(cudaEventElapsedTime(&ms, begin, end), "kernel elapsed"); kernel_event.push_back(ms);
        };
        std::cerr << "stage kernel\n";
        const auto kernel_wall = measure(kernel_call, o.warmup, o.repeats);
        kernel_event.erase(kernel_event.begin(), kernel_event.begin() + o.warmup);
        check(cudaGetLastError(), "kernel-only launch");
        cv::Mat kernel_image(32, 32, CV_32FC1);
        check(cudaMemcpy(kernel_image.data, device_output, 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost), "kernel output download");
        const bool raw_kernel_parity = std::memcmp(kernel_image.data, raw.back().data, 32 * 32 * sizeof(float)) == 0;
        if (!raw_kernel_parity) throw std::runtime_error("kernel-only output differs from original wrapper");
        for (int i = 0; i < o.batch; ++i) {
            cameraRaycastKernel<<<sub.image_height, sub.image_width>>>(device_output, map, sub, poses[i]);
            check(cudaMemcpy(kernel_image.data, device_output, 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost), "per-pose kernel output download");
            if (std::memcmp(kernel_image.data, raw[i].data, 32 * 32 * sizeof(float)) != 0)
                throw std::runtime_error("per-pose kernel output differs from original wrapper");
        }

        std::cerr << "stage wrapper\n";
        const auto wrapper = measure([&] {
            for (int i = 0; i < o.batch; ++i) renderDepthImage(&map, &sub, poses[i], raw[i]);
            check(cudaGetLastError(), "wrapper rendering");
        }, o.warmup, o.repeats);
        std::default_random_engine post_rng(3), full_rng(3);
        std::cerr << "stage postprocess\n";
        const auto post_time = measure([&] {
            for (int i = 0; i < o.batch; ++i)
                postprocessToF(tof, raw[i], 4, .2f, .01f, 0, .08f, .015f, post_rng, post[i]);
        }, o.warmup, o.repeats);
        std::cerr << "stage complete ToF\n";
        const auto full_time = measure([&] {
            for (int i = 0; i < o.batch; ++i)
                renderToFSenseMImage(&map, tof, poses[i], 4, .2f, .01f, 0, .08f, .015f, full_rng, full[i]);
            check(cudaGetLastError(), "complete ToF rendering");
        }, o.warmup, o.repeats);
        ImageStats raw_stats, post_stats, full_stats;
        for (int i = 0; i < o.batch; ++i) {
            raw_stats.add(raw[i]); post_stats.add(post[i]); full_stats.add(full[i]);
            if (std::memcmp(full[i].data, post[i].data, full[i].total() * sizeof(float)) != 0)
                throw std::runtime_error("measured post/full output parity failed");
        }
        if (!(post_rng == full_rng)) throw std::runtime_error("measured post/full RNG parity failed");
        check(cudaEventDestroy(begin), "destroy begin"); check(cudaEventDestroy(end), "destroy end");
        check(cudaFree(device_output), "free kernel output"); map.freeGridMap();

        std::ofstream out(o.output);
        if (!out) throw std::runtime_error("cannot write " + o.output);
        out << std::setprecision(10) << "{\n\"status\":\"ok\",\"source_commit\":\"cf0b7f4b17712667a8bbdb1126f5e914904ece9d\","
            << "\n\"device\":" << quoted(props.name) << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
            << ",\"cloud\":" << quoted(o.cloud) << ",\"poses\":" << quoted(o.poses)
            << ",\"point_count\":" << cloud->points.size() << ",\"batch\":" << o.batch
            << ",\"native_batch\":false,\"views_processed_serially\":true,\"warmup\":" << o.warmup << ",\"repeats\":" << o.repeats
            << ",\"fov_deg\":" << o.fov << ",\"max_range_m\":" << o.max_range << ",\"resolution_m\":" << o.resolution
            << ",\"tof_shape\":[8,8],\"subray_shape\":[32,32],\"subrays_per_zone\":16"
            << ",\"depth_quantile\":0.2,\"noise_std_m\":0.01,\"far_noise_std_m\":0,\"signal_floor\":0.08,\"min_depth_m\":0.015"
            << ",\"map_setup_ms\":" << map_ms << ",\"output_residency\":\"CPU cv::Mat\",\"stages\":{\n\"kernel_cuda_event\":";
        series(out, kernel_event, o.batch); out << ",\n\"kernel_wall\":"; series(out, kernel_wall, o.batch);
        out << ",\n\"raw_wrapper_wall\":"; series(out, wrapper, o.batch);
        out << ",\n\"postprocess_wall\":"; series(out, post_time, o.batch);
        out << ",\n\"end_to_end_wall\":"; series(out, full_time, o.batch);
        out << "},\n\"parity\":{\"original_vs_split_byte_exact\":true,\"rng_state_equal\":true,\"kernel_vs_wrapper_byte_exact\":true},"
            << "\n\"raw_output\":"; raw_stats.write(out); out << ",\"post_output\":"; post_stats.write(out);
        out << ",\"full_output\":"; full_stats.write(out); out << "\n}\n";
        if (!out) throw std::runtime_error("output write failed");
        std::cerr << "complete " << o.output << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n'; return 1;
    }
}
