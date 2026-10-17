#include "maps.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace offline_map {
namespace {
bool positiveFinite(double value) { return std::isfinite(value) && value > 0; }
bool fitsPositiveFloat(double value) {
  return positiveFinite(value) && value <= std::numeric_limits<float>::max() &&
         value >= std::numeric_limits<float>::denorm_min();
}
} // namespace

MeshMap generateMesh(const Config& config) {
  if (config.type != "pillars")
    throw std::invalid_argument("OBJ mesh output supports only pillars");
  if (config.count < 0 || config.count > 100000)
    throw std::invalid_argument("count must be between 0 and 100000");
  for (const double size : {config.size_x, config.size_y, config.size_z})
    if (!fitsPositiveFloat(size))
      throw std::invalid_argument("mesh sizes must fit positive finite float metres");
  const double h_min = config.height_min.value_or(0.1 * config.size_z);
  const double h_max = config.height_max.value_or(config.size_z);
  if (!fitsPositiveFloat(config.width_min) || !fitsPositiveFloat(config.width_max) ||
      config.width_max < config.width_min)
    throw std::invalid_argument("mesh widths must be positive finite and ordered");
  if (!fitsPositiveFloat(h_min) || !fitsPositiveFloat(h_max) || h_max < h_min)
    throw std::invalid_argument("mesh heights must be positive finite and ordered");
  if (config.count > 0) {
    for (const double extent : {config.size_x / 2, config.size_y / 2}) {
      const double spacing = std::nextafter(extent, std::numeric_limits<double>::infinity()) - extent;
      if (config.width_min <= 4 * spacing)
        throw std::invalid_argument("mesh width is finer than XYZ precision at map extent");
    }
  }
  MeshMap objects;
  objects.reserve(static_cast<std::size_t>(config.count) + 1);
  std::default_random_engine engine(config.seed);
  std::uniform_real_distribution<double> x(-config.size_x / 2, config.size_x / 2);
  std::uniform_real_distribution<double> y(-config.size_y / 2, config.size_y / 2);
  std::uniform_real_distribution<double> width(config.width_min, config.width_max);
  std::uniform_real_distribution<double> height(h_min, h_max);
  for (int index = 0; index < config.count; ++index) {
    const double cx = x(engine), cy = y(engine), w = width(engine), h = height(engine);
    // Intersect the physical box with map bounds, then emit all six faces of
    // that intersection. Clipping vertices alone would leave open surfaces.
    const double x0 = std::max(-config.size_x / 2, cx - w / 2);
    const double x1 = std::min(config.size_x / 2, cx + w / 2);
    const double y0 = std::max(-config.size_y / 2, cy - w / 2);
    const double y1 = std::min(config.size_y / 2, cy + w / 2);
    const double z1 = std::min(config.size_z, h);
    if (!(x0 < x1 && y0 < y1 && 0 < z1))
      throw std::invalid_argument("mesh box collapsed at coordinate precision");
    objects.push_back({"pillar_" + std::to_string(index),
        {{{x0, y0, 0}}, {{x1, y0, 0}}, {{x1, y1, 0}}, {{x0, y1, 0}},
         {{x0, y0, z1}}, {{x1, y0, z1}}, {{x1, y1, z1}}, {{x0, y1, z1}}},
        {{{0, 2, 1}}, {{0, 3, 2}}, {{4, 5, 6}}, {{4, 6, 7}},
         {{0, 1, 5}}, {{0, 5, 4}}, {{1, 2, 6}}, {{1, 6, 5}},
         {{2, 3, 7}}, {{2, 7, 6}}, {{3, 0, 4}}, {{3, 4, 7}}}});
  }
  const double x0 = -config.size_x / 2, x1 = config.size_x / 2;
  const double y0 = -config.size_y / 2, y1 = config.size_y / 2;
  objects.push_back({"ground", {{{x0, y0, 0}}, {{x1, y0, 0}}, {{x1, y1, 0}}, {{x0, y1, 0}}},
                     {{{0, 1, 2}}, {{0, 2, 3}}}});
  return objects;
}

} // namespace offline_map
