#ifndef TOF_TRACKER_OFFLINE_MAPS_HPP
#define TOF_TRACKER_OFFLINE_MAPS_HPP

// Physical mesh generation for this project. Reference distributions and
// parameter origins are recorded in repository-root docs/origins.md.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace offline_map {
struct Config {
  std::string type = "pillars";
  double size_x = 8.0;
  double size_y = 6.0;
  double size_z = 3.0;
  std::uint32_t seed = 3;
  int count = 12;
  double width_min = 0.6;
  double width_max = 1.2;
  std::optional<double> height_min;
  std::optional<double> height_max;
};

// Vertices are in world metres. Faces index only this object's own vertices;
// the writer translates them to OBJ's file-wide one-based indices.
struct MeshObject {
  std::string name;
  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<std::size_t, 3>> triangles;
};
using MeshMap = std::vector<MeshObject>;
// One fresh pillar field and separate ground plane per call, in physical metres.
MeshMap generateMesh(const Config& config);
} // namespace offline_map
#endif
