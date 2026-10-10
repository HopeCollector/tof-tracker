#include "maps.hpp"
#include "perlinnoise.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

// Selected mechanisms from YOPO-tracker / mockamap; see docs/origins.md.
namespace offline_map {
namespace {
constexpr double pi = 3.14159265358979323846;

bool positiveFinite(double value) { return std::isfinite(value) && value > 0; }
bool fitsPositiveFloat(double value) {
  return positiveFinite(value) && value <= std::numeric_limits<float>::max() &&
         value >= std::numeric_limits<float>::denorm_min();
}

int gridCells(double size, double resolution) {
  if (!fitsPositiveFloat(size) || !fitsPositiveFloat(resolution))
    throw std::invalid_argument("size and resolution must fit positive finite float metres");
  const double cells = size / resolution;
  if (!std::isfinite(cells) || cells < 1 || cells > std::numeric_limits<int>::max() - 2)
    throw std::invalid_argument("size/resolution exceeds supported grid dimensions");
  const double rounded = std::round(cells);
  if (std::abs(cells - rounded) > 1e-7)
    throw std::invalid_argument("size must be an integral multiple of resolution");
  return static_cast<int>(rounded);
}

int sampleSteps(double length, double resolution) {
  const double steps = std::ceil(length / resolution);
  if (!std::isfinite(steps) || steps < 0 || steps > std::numeric_limits<int>::max())
    throw std::invalid_argument("sample steps exceed supported integer range");
  return static_cast<int>(steps);
}

std::size_t boundedProduct(std::size_t a, std::size_t b, std::size_t limit) {
  if (b && a > limit / b)
    throw std::invalid_argument("requested samples exceed max-points");
  return a * b;
}

class Generator {
public:
  explicit Generator(const Config& config);
  Cloud::Ptr generate();

private:
  Config config_;
  int nx_, ny_, nz_;
  Cloud::Ptr cloud_{new Cloud()};
  std::size_t samples_ = 0;

  void reserveSamples(std::size_t count);
  void append(double x, double y, double z);
  void plane(double z);
  void pillars();
  void walls();
  void perlin();
};

Generator::Generator(const Config& config) : config_(config) {
  if (config_.type != "pillars" && config_.type != "walls" && config_.type != "perlin")
    throw std::invalid_argument("type must be pillars, walls, or perlin");
  if (config_.max_points == 0 || config_.max_points > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("max-points must be from 1 to UINT32_MAX");
  nx_ = gridCells(config_.size_x, config_.resolution);
  ny_ = gridCells(config_.size_y, config_.resolution);
  nz_ = gridCells(config_.size_z, config_.resolution);
  for (const double extent : {config_.size_x / 2, config_.size_y / 2, config_.size_z}) {
    const float coordinate = static_cast<float>(extent);
    const double spacing = static_cast<double>(std::nextafter(
        coordinate, std::numeric_limits<float>::infinity())) - coordinate;
    if (config_.resolution < spacing)
      throw std::invalid_argument("resolution is finer than float XYZ precision at map extent");
  }
  const auto xy = boundedProduct(nx_, ny_, config_.max_points);
  if (config_.type == "perlin") boundedProduct(xy, nz_, config_.max_points);
  if (config_.type != "perlin") {
    if (config_.count < 0 || config_.count > 100000)
      throw std::invalid_argument("count must be between 0 and 100000");
    if (!fitsPositiveFloat(config_.width_min) || !fitsPositiveFloat(config_.width_max) ||
        config_.width_min < config_.resolution || config_.width_max < config_.width_min ||
        config_.width_max > std::max(config_.size_x, config_.size_y))
      throw std::invalid_argument("widths must be ordered, >= resolution and fit map span");
  }
  if (config_.type == "walls" && (!fitsPositiveFloat(config_.wall_thickness) ||
      config_.wall_thickness > std::min(config_.size_x, config_.size_y)))
    throw std::invalid_argument("wall thickness must be positive and fit the map");
  if (config_.type == "perlin" && (!positiveFinite(config_.complexity) ||
      !positiveFinite(config_.attenuation) || !positiveFinite(config_.fill) ||
      config_.complexity > 0.5 || config_.fill >= 1 || config_.fractal < 1 || config_.fractal > 16))
    throw std::invalid_argument("perlin needs 0 < complexity <= 0.5, attenuation > 0, 0 < fill < 1, 1 <= fractal <= 16");
}

void Generator::reserveSamples(std::size_t count) {
  if (count > config_.max_points - samples_)
    throw std::invalid_argument("requested samples exceed max-points");
  samples_ += count;
}

void Generator::append(double x, double y, double z) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    throw std::runtime_error("generator returned a non-finite point");
  // Clip in metres before converting to the PCL float representation.
  if (x < -config_.size_x / 2 || x > config_.size_x / 2 ||
      y < -config_.size_y / 2 || y > config_.size_y / 2 || z < 0 || z > config_.size_z)
    return;
  if (cloud_->size() >= config_.max_points)
    throw std::runtime_error("map exceeded max-points");
  cloud_->emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

void Generator::plane(double z) {
  reserveSamples(boundedProduct(nx_, ny_, config_.max_points));
  for (int i = 0; i < nx_; ++i)
    for (int j = 0; j < ny_; ++j)
      append(i * config_.resolution - config_.size_x / 2,
             j * config_.resolution - config_.size_y / 2, z);
}

Cloud::Ptr Generator::generate() {
  if (config_.type == "pillars") pillars();
  else if (config_.type == "walls") walls();
  else perlin();
  plane(0.0);
  if (config_.type == "walls" && config_.ceiling) plane(config_.size_z);
  cloud_->width = static_cast<std::uint32_t>(cloud_->size());
  cloud_->height = 1;
  cloud_->is_dense = true;
  return cloud_;
}

void Generator::pillars() {
  std::default_random_engine engine(config_.seed);
  std::uniform_real_distribution<double> x(-config_.size_x / 2, config_.size_x / 2);
  std::uniform_real_distribution<double> y(-config_.size_y / 2, config_.size_y / 2);
  std::uniform_real_distribution<double> width(config_.width_min, config_.width_max);
  std::uniform_real_distribution<double> height(0, config_.size_z);
  for (int obstacle = 0; obstacle < config_.count; ++obstacle) {
    const double cx = x(engine), cy = y(engine);
    const int w = sampleSteps(width(engine), config_.resolution);
    const int h = sampleSteps(height(engine), config_.resolution);
    const auto square = boundedProduct(w, w, config_.max_points);
    const auto inner = static_cast<std::size_t>(std::max(w - 2, 0));
    const auto edge = square - inner * inner;
    const auto caps = boundedProduct(square, std::min(h, 2), config_.max_points);
    const auto sides = boundedProduct(edge, std::max(h - 2, 0), config_.max_points);
    reserveSamples(caps);
    reserveSamples(sides);
    const int low = -w / 2, high = low + w;
    for (int r = low; r < high; ++r)
      for (int s = low; s < high; ++s) {
        const auto emit = [&](int t) {
          append(cx + r * config_.resolution, cy + s * config_.resolution,
                 t * config_.resolution);
        };
        if (r == low || r == high - 1 || s == low || s == high - 1) {
          for (int t = 0; t < h; ++t) emit(t);
        } else if (h > 0) {
          // Same shell and traversal order, without scanning its empty interior.
          emit(0);
          if (h > 1) emit(h - 1);
        }
      }
  }
}

void Generator::walls() {
  std::default_random_engine engine(config_.seed);
  std::uniform_real_distribution<double> x(-config_.size_x / 2, config_.size_x / 2);
  std::uniform_real_distribution<double> y(-config_.size_y / 2, config_.size_y / 2);
  std::uniform_real_distribution<double> width(config_.width_min, config_.width_max);
  std::uniform_real_distribution<double> height(0.3 * config_.size_z, config_.size_z);
  std::uniform_real_distribution<float> yaw(-pi, pi), pitch(-pi / 10, pi / 10);
  for (int wall = 0; wall < config_.count; ++wall) {
    const double cx = x(engine), cy = y(engine);
    const double angle_yaw = yaw(engine), angle_pitch = pitch(engine);
    const double w = width(engine), h = height(engine), t = config_.wall_thickness;
    const int nw = sampleSteps(w, config_.resolution);
    const int nh = sampleSteps(h, config_.resolution);
    const int nt = sampleSteps(t, config_.resolution);
    reserveSamples(boundedProduct(boundedProduct(nw, nh, config_.max_points),
                                  nt, config_.max_points));
    const double cosy = std::cos(angle_yaw), siny = std::sin(angle_yaw);
    const double cosp = std::cos(angle_pitch), sinp = std::sin(angle_pitch);
    for (int i = 0; i < nw; ++i)
      for (int j = 0; j < nh; ++j)
        for (int k = 0; k < nt; ++k) {
          const double local_x = -w / 2 + i * config_.resolution;
          const double local_y = -t / 2 + k * config_.resolution;
          const double local_z = j * config_.resolution;
          // Preserve upstream yaw-then-X-tilt rotation, including its convention.
          const double x1 = cosy * local_x - siny * local_y;
          const double y1 = siny * local_x + cosy * local_y;
          append(cx + x1, cy + cosp * y1 - sinp * local_z,
                 sinp * y1 + cosp * local_z);
        }
  }
}

void Generator::perlin() {
  const auto count = boundedProduct(boundedProduct(nx_, ny_, config_.max_points),
                                    nz_, config_.max_points);
  PerlinNoise noise(config_.seed);
  std::vector<double> values;
  values.reserve(count);
  for (int i = 0; i < nx_; ++i)
    for (int j = 0; j < ny_; ++j)
      for (int k = 0; k < nz_; ++k) {
        double value = 0;
        for (int octave = 1; octave <= config_.fractal; ++octave) {
          const double frequency = std::ldexp(1.0, octave);
          value += config_.attenuation / octave * noise.noise(
              frequency * i * config_.complexity, frequency * j * config_.complexity,
              frequency * k * config_.complexity);
        }
        if (!std::isfinite(value)) throw std::runtime_error("non-finite perlin noise");
        values.push_back(value);
      }
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const auto index = std::min(count - 1, static_cast<std::size_t>(count * (1 - config_.fill)));
  const double threshold = sorted[index];
  std::size_t position = 0;
  for (int i = 0; i < nx_; ++i)
    for (int j = 0; j < ny_; ++j)
      for (int k = 0; k < nz_; ++k)
        if (values[position++] > threshold) {
          reserveSamples(1);
          append(i * config_.resolution - config_.size_x / 2,
                 j * config_.resolution - config_.size_y / 2, k * config_.resolution);
        }
}
} // namespace

Cloud::Ptr generate(const Config& config) { return Generator(config).generate(); }
} // namespace offline_map
