#include "maps.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {
void usage(std::ostream& stream) {
  stream << "Usage: offline_mapgen --output MAP.obj|MAP.ply [options]\n"
         << "  --type pillars|walls|perlin  --seed UINT32\n"
         << "  --size-x M --size-y M --size-z M\n"
         << "  --count N --width-min M --width-max M (pillars/walls)\n"
         << "  --height-min M --height-max M (OBJ pillars only)\n"
         << "  --resolution M (PLY only)\n"
         << "  --wall-thickness M --ceiling 0|1 (walls)\n"
         << "  --complexity V --fill V --fractal N --attenuation V (perlin)\n"
         << "  --max-points N (PLY only)\n"
         << "OBJ: closed pillars and a separate ground plane; physical dimensions.\n"
         << "PLY: legacy sampled maps; sizes must be multiples of resolution.\n"
         << "XYZ metres;\n"
         << "X/Y are centred on zero; Z is from zero to size-z.\n"
         << "Existing output is refused. Parent directory must exist.\n";
}

double number(const std::string& value) {
  std::size_t end = 0;
  const double parsed = std::stod(value, &end);
  if (end != value.size() || !std::isfinite(parsed))
    throw std::invalid_argument("not a finite number: " + value);
  return parsed;
}

std::uint64_t unsignedNumber(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("not an unsigned integer: " + value);
  return std::stoull(value);
}

int intNumber(const std::string& value) {
  const auto parsed = unsignedNumber(value);
  if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("integer too large: " + value);
  return static_cast<int>(parsed);
}

template <typename Writer>
void publish(const std::filesystem::path& path, Writer writer) {
  // Write a unique temporary file beside the destination. Publishing via link()
  // is atomic and refuses any existing directory entry, including a symlink.
  auto pattern = (path.parent_path() / ("." + path.filename().string() + ".tmp-XXXXXX")).string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  const int descriptor = ::mkstemp(name.data());
  if (descriptor < 0) throw std::system_error(errno, std::generic_category(), "create output temporary file");
  struct Temporary {
    std::string path;
    FILE* stream = nullptr;
    ~Temporary() {
      if (stream) std::fclose(stream);
      ::unlink(path.c_str());
    }
  } temporary{name.data(), ::fdopen(descriptor, "w")};
  if (!temporary.stream) {
    const int error = errno;
    ::close(descriptor);
    throw std::system_error(error, std::generic_category(), "open output stream");
  }
  const auto write = [&](const char* format, auto... values) {
    if (std::fprintf(temporary.stream, format, values...) < 0)
      throw std::system_error(errno, std::generic_category(), "write map");
  };
  writer(write);
  const int result = std::fclose(temporary.stream);
  temporary.stream = nullptr;
  if (result != 0) throw std::system_error(errno, std::generic_category(), "close map");
  if (::link(temporary.path.c_str(), path.c_str()) != 0)
    throw std::system_error(errno, std::generic_category(), "publish map without replacing output");
}

void savePly(const std::filesystem::path& path, const offline_map::Config& config,
             const offline_map::Cloud& cloud) {
  publish(path, [&](const auto& write) {
    write("%s", "ply\nformat ascii 1.0\ncomment unit metre\n");
    write("comment offline_mapgen type %s seed %u\n", config.type.c_str(), config.seed);
    write("comment size_xyz %.17g %.17g %.17g resolution %.17g\n",
          config.size_x, config.size_y, config.size_z, config.resolution);
    write("comment count %d width_min %.17g width_max %.17g wall_thickness %.17g ceiling %d\n",
          config.count, config.width_min, config.width_max, config.wall_thickness, int(config.ceiling));
    write("comment complexity %.17g fill %.17g fractal %d attenuation %.17g\n",
          config.complexity, config.fill, config.fractal, config.attenuation);
    write("element vertex %zu\nproperty float x\nproperty float y\nproperty float z\nend_header\n", cloud.size());
    for (const auto& point : cloud)
      write("%.*g %.*g %.*g\n", std::numeric_limits<float>::max_digits10, double(point.x),
            std::numeric_limits<float>::max_digits10, double(point.y),
            std::numeric_limits<float>::max_digits10, double(point.z));
  });
}

void saveObj(const std::filesystem::path& path, const offline_map::Config& config,
             const offline_map::MeshMap& objects) {
  publish(path, [&](const auto& write) {
    write("%s", "# unit metre\n");
    write("# offline_mapgen type pillars seed %u\n", config.seed);
    write("# size_xyz %.17g %.17g %.17g\n", config.size_x, config.size_y, config.size_z);
    write("# count %d width_min %.17g width_max %.17g height_min %.17g height_max %.17g\n",
          config.count, config.width_min, config.width_max,
          config.height_min.value_or(0.1 * config.size_z), config.height_max.value_or(config.size_z));
    std::size_t first_vertex = 1;
    for (const auto& object : objects) {
      write("o %s\n", object.name.c_str());
      for (const auto& vertex : object.vertices)
        write("v %.17g %.17g %.17g\n", vertex[0], vertex[1], vertex[2]);
      for (const auto& triangle : object.triangles)
        write("f %zu %zu %zu\n", first_vertex + triangle[0], first_vertex + triangle[1],
              first_vertex + triangle[2]);
      first_vertex += object.vertices.size();
    }
  });
}
} // namespace

int main(int argc, char** argv) {
  try {
    offline_map::Config config;
    std::filesystem::path output;
    std::set<std::string> specified;
    for (int index = 1; index < argc; ++index) {
      const std::string key(argv[index]);
      if (key == "--help") { usage(std::cout); return 0; }
      if (index + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      const std::string value(argv[++index]);
      specified.insert(key);
      if (key == "--output") output = value;
      else if (key == "--type") config.type = value;
      else if (key == "--seed") {
        const auto seed = unsignedNumber(value);
        if (seed > std::numeric_limits<std::uint32_t>::max())
          throw std::invalid_argument("seed exceeds uint32");
        config.seed = static_cast<std::uint32_t>(seed);
      }
      else if (key == "--size-x") config.size_x = number(value);
      else if (key == "--size-y") config.size_y = number(value);
      else if (key == "--size-z") config.size_z = number(value);
      else if (key == "--resolution") config.resolution = number(value);
      else if (key == "--count") config.count = intNumber(value);
      else if (key == "--width-min") config.width_min = number(value);
      else if (key == "--width-max") config.width_max = number(value);
      else if (key == "--height-min") config.height_min = number(value);
      else if (key == "--height-max") config.height_max = number(value);
      else if (key == "--wall-thickness") config.wall_thickness = number(value);
      else if (key == "--ceiling") {
        if (value != "0" && value != "1") throw std::invalid_argument("ceiling requires 0 or 1");
        config.ceiling = value == "1";
      }
      else if (key == "--complexity") config.complexity = number(value);
      else if (key == "--fill") config.fill = number(value);
      else if (key == "--fractal") config.fractal = intNumber(value);
      else if (key == "--attenuation") config.attenuation = number(value);
      else if (key == "--max-points") {
        const auto maximum = unsignedNumber(value);
        if (maximum > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument("max-points exceeds size_t");
        config.max_points = static_cast<std::size_t>(maximum);
      }
      else throw std::invalid_argument("unknown option " + key);
    }
    if (output.empty()) throw std::invalid_argument("--output is required");
    if (std::filesystem::exists(output) || std::filesystem::is_symlink(output))
      throw std::invalid_argument("output already exists");
    if (output.extension() == ".obj") {
      for (const std::string key : {"--resolution", "--max-points", "--wall-thickness", "--ceiling",
                                   "--complexity", "--fill", "--fractal", "--attenuation"})
        if (specified.count(key))
          throw std::invalid_argument(key + " applies only to PLY output");
      const auto objects = offline_map::generateMesh(config);
      saveObj(output, config, objects);
      std::cout << "wrote " << config.count << " closed pillars and ground to " << output.string() << '\n';
    } else if (output.extension() == ".ply") {
      const auto cloud = offline_map::generate(config);
      savePly(output, config, *cloud);
      std::cout << "wrote " << cloud->size() << " points to " << output.string() << '\n';
    } else {
      throw std::invalid_argument("output extension must be .obj or .ply");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "offline_mapgen: " << error.what() << '\n';
    return 1;
  }
}
