#ifndef TOF_TRACKER_OFFLINE_MAPS_HPP
#define TOF_TRACKER_OFFLINE_MAPS_HPP

// Selected generation mechanisms adapted from YOPO-tracker / mockamap.
// See repository-root docs/origins.md and tools/mapgen/third_party_notices.
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstddef>
#include <cstdint>
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace offline_map {
using Cloud = pcl::PointCloud<pcl::PointXYZ>;

struct Config {
  std::string type = "pillars";
  double size_x = 8.0;
  double size_y = 6.0;
  double size_z = 3.0;
  double resolution = 0.2;
  std::uint32_t seed = 3;
  std::size_t max_points = 5000000;
  int count = 12;
  double width_min = 0.6;
  double width_max = 1.2;
  std::optional<double> height_min;
  std::optional<double> height_max;
  double wall_thickness = 0.2;
  bool ceiling = false;
  double complexity = 0.02;
  double fill = 0.1;
  int fractal = 1;
  double attenuation = 0.1;
};

// One fresh background map per call. No target/entity, renderer, training,
// ROS, dataset-layout or globally compiled path settings are involved.
Cloud::Ptr generate(const Config& config);

// Vertices are in world metres. Faces index only this object's own vertices;
// the writer translates them to OBJ's file-wide one-based indices.
struct MeshObject {
  std::string name;
  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<std::size_t, 3>> triangles;
};
using MeshMap = std::vector<MeshObject>;
MeshMap generateMesh(const Config& config);
} // namespace offline_map
#endif
