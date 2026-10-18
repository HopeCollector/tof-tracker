// CPU mesh-distance C ABI. All coordinates and residual calculations are double.
// Public input/lifetime contracts are documented in mesh_distance.py.
//
// Vendored TriangleMeshDistance.h: Jose Antonio Fernandez Fernandez, 2021, MIT.
// Source: https://github.com/UPC-ViRVIG/SdfLib commit
// 8db373ef71d65be24badf6ae10750a932bbc223b, file
// libs/InteractiveComputerGraphics/InteractiveComputerGraphics/TriangleMeshDistance.h.
// Original SHA256:
// c9a4fabb0e54a3b2d4ef4eb9deb941387d9916f8bd138d0ed5ff7c329175c58c.
// Vendored SHA256:
// e1ff3c81ffb856a8d53a2576704cc83b55bfd63f8bcc14ca8ae06cb8838886f3.
// The header adds <stdexcept> and signed_distance guards for a missing/nonfinite
// nearest triangle before indexing, an invalid nearest-entity enum, and
// nonfinite pseudonormals. It also normalizes trailing whitespace/newline.
// Pre-whitespace-normalization patched SHA256 (older validation snapshots):
// 1c631199a4462c3443aac0ef9647b007c38dfff874e518d1dd6f40a0c50e752a.
// Successful finite BVH/query arithmetic is unchanged; these guards turn
// overflow/error paths into C ABI failures. MIT notice and separate LICENSE
// are retained. No SdfLib runtime, octree or Hermite data is linked.
//
// Below, the BVH selects a nearest triangle and its pseudonormal sign. We
// independently recompute closest projection/three clamped-segment residuals
// to avoid cancellation in the header's expanded quadratic. The selected face
// still comes from the BVH: this is not a Boolean or exact-arithmetic proof.
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <set>
#include <map>
#include <string>
#include "third_party/TriangleMeshDistance.h"

namespace {
using Vec = tmd::Vec3d;
using Face = std::array<int, 3>;

void fail(const char* message) { throw std::invalid_argument(message); }
bool finite(const Vec& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

struct Query {
    std::vector<std::array<double, 3>> vertices;
    std::vector<Face> faces;
    tmd::TriangleMeshDistance bvh;

    Query(const double* xyz, std::uint64_t nv, const std::int32_t* ijk, std::uint64_t nf) {
        if (!xyz || !ijk || nv < 4 || nf < 4 ||
            nv > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            nf > static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 2))
            fail("mesh needs nonempty finite vertices and int32-sized triangle indices");
        vertices.resize(nv);
        for (std::size_t i = 0; i < nv; ++i) {
            vertices[i] = {xyz[3*i], xyz[3*i+1], xyz[3*i+2]};
            if (!finite(Vec(vertices[i][0], vertices[i][1], vertices[i][2])))
                fail("mesh vertices must be finite");
        }
        faces.resize(nf);
        std::set<Face> unique_faces;
        std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> edges;
        std::vector<std::vector<int>> incident(nv);
        for (std::size_t i = 0; i < nf; ++i) {
            Face f{ijk[3*i], ijk[3*i+1], ijk[3*i+2]};
            for (int v : f)
                if (v < 0 || static_cast<std::uint64_t>(v) >= nv) fail("triangle index out of range");
            if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) fail("repeated triangle vertex");
            Face key = f;
            std::sort(key.begin(), key.end());
            if (!unique_faces.insert(key).second) fail("duplicate triangle");
            const Vec a = vertex(f[0]), b = vertex(f[1]), c = vertex(f[2]);
            const Vec normal = (b-a).cross(c-a);
            if (!finite(normal) || !(normal.squaredNorm() > 0.0) || !std::isfinite(normal.squaredNorm()))
                fail("triangle is degenerate or exceeds numerical range");
            faces[i] = f;
            for (int k = 0; k < 3; ++k) {
                const int v = f[k], w = f[(k+1)%3];
                edges[{std::min(v,w), std::max(v,w)}].push_back({static_cast<int>(i), v < w ? 1 : -1});
                incident[v].push_back(static_cast<int>(i));
            }
        }
        std::vector<std::vector<int>> adjacent(nf);
        for (const auto& entry : edges) {
            const auto& edge = entry.second;
            if (edge.size() != 2 || edge[0].second + edge[1].second != 0)
                fail("mesh must be closed with opposite winding on each shared edge");
            adjacent[edge[0].first].push_back(edge[1].first);
            adjacent[edge[1].first].push_back(edge[0].first);
        }
        // Each vertex must have one connected incident-face fan. Edge incidence
        // alone would permit two solids touching only at a nonmanifold vertex.
        for (const auto& fan : incident) {
            if (fan.empty()) fail("unused mesh vertex");
            std::set<int> remaining(fan.begin(), fan.end());
            std::vector<int> pending{*remaining.begin()};
            remaining.erase(pending.back());
            while (!pending.empty()) {
                const int face = pending.back(); pending.pop_back();
                for (int neighbour : adjacent[face])
                    if (remaining.erase(neighbour)) pending.push_back(neighbour);
            }
            if (!remaining.empty()) fail("mesh has a nonmanifold vertex");
        }
        std::vector<bool> visited(nf, false);
        for (std::size_t seed = 0; seed < nf; ++seed) {
            if (visited[seed]) continue;
            const Vec origin = vertex(faces[seed][0]);
            long double volume6 = 0;
            std::vector<int> pending{static_cast<int>(seed)};
            visited[seed] = true;
            while (!pending.empty()) {
                const int i = pending.back(); pending.pop_back();
                const Vec a = vertex(faces[i][0])-origin;
                const Vec b = vertex(faces[i][1])-origin;
                const Vec c = vertex(faces[i][2])-origin;
                volume6 += static_cast<long double>(a.dot(b.cross(c)));
                for (int neighbour : adjacent[i])
                    if (!visited[neighbour]) { visited[neighbour] = true; pending.push_back(neighbour); }
            }
            if (!std::isfinite(volume6) || volume6 <= 0)
                fail("each connected component must enclose positive volume with outward winding");
        }
        bvh.construct(vertices, faces);
    }

    Vec vertex(int i) const { const auto& v = vertices[i]; return Vec(v[0], v[1], v[2]); }

    double distance(const Vec& p) const {
        const tmd::Result result = bvh.signed_distance(std::array<double,3>{p[0],p[1],p[2]});
        if (result.triangle_id < 0 || static_cast<std::size_t>(result.triangle_id) >= faces.size() ||
            !std::isfinite(result.distance))
            throw std::runtime_error("mesh BVH returned an invalid nearest face or sign");
        const Face& f = faces[result.triangle_id];
        const Vec a = vertex(f[0]), b = vertex(f[1]), c = vertex(f[2]);
        const Vec normal = (b-a).cross(c-a);
        const Vec unit = normal / normal.norm();
        const double height = (p-a).dot(unit);
        const Vec projection = p-unit*height;
        const std::array<double,3> side = {(b-a).cross(projection-a).dot(normal),
            (c-b).cross(projection-b).dot(normal), (a-c).cross(projection-c).dot(normal)};
        const bool in_face = side[0] >= 0 && side[1] >= 0 && side[2] >= 0;
        const bool face_interior = side[0] > 0 && side[1] > 0 && side[2] > 0;
        double best_squared = (p-a).squaredNorm();
        const auto consider = [&](const Vec& q) { best_squared = std::min(best_squared, (p-q).squaredNorm()); };
        if (in_face) consider(projection);
        const auto edge = [&](const Vec& start, const Vec& end) {
            const Vec v = end-start;
            const double t = std::clamp((p-start).dot(v) / v.squaredNorm(), 0.0, 1.0);
            consider(start+v*t);
        };
        edge(a,b); edge(b,c); edge(c,a);
        const double scale = std::max({p.norm(),a.norm(),b.norm(),c.norm(),(b-a).norm(),(c-b).norm(),(a-c).norm()});
        double magnitude = std::sqrt(best_squared);
        if (!std::isfinite(magnitude)) throw std::runtime_error("distance exceeds numerical range");
        if (magnitude <= 64.0 * std::numeric_limits<double>::epsilon() * scale) magnitude = 0.0;
        // Stable oriented-plane sign at a strict face interior; at edges and
        // vertices keep the mesh pseudonormal sign, including signed zero.
        return std::copysign(magnitude, face_interior ? height : result.distance);
    }
};

void set_error(char* output, std::size_t capacity, const char* message) noexcept {
    if (!output || capacity == 0) return;
    const std::size_t count = std::min(capacity-1, std::strlen(message));
    std::memcpy(output, message, count); output[count] = '\0';
}
} // namespace

extern "C" {
int mdq_abi_version() noexcept { return 1; }

int mdq_create(const double* xyz, std::uint64_t nv, const std::int32_t* ijk, std::uint64_t nf,
               void** output, char* error, std::size_t error_capacity) noexcept {
    if (output) *output = nullptr;
    try {
        if (!output) fail("missing output handle");
        *output = new Query(xyz, nv, ijk, nf);
        return 0;
    } catch (const std::exception& exception) { set_error(error,error_capacity,exception.what()); }
      catch (...) { set_error(error,error_capacity,"unknown mesh construction failure"); }
    return 1;
}

int mdq_query(const void* handle, const double* xyz, std::uint64_t count, double* distances,
              char* error, std::size_t error_capacity) noexcept {
    try {
        if (!handle || (count && (!xyz || !distances))) fail("invalid query pointer");
        const Query& query = *static_cast<const Query*>(handle);
        for (std::uint64_t i = 0; i < count; ++i) {
            const Vec p(xyz[3*i],xyz[3*i+1],xyz[3*i+2]);
            if (!finite(p)) fail("query positions must be finite");
            distances[i] = query.distance(p);
        }
        return 0;
    } catch (const std::exception& exception) { set_error(error,error_capacity,exception.what()); }
      catch (...) { set_error(error,error_capacity,"unknown mesh query failure"); }
    return 1;
}

void mdq_destroy(void* handle) noexcept { delete static_cast<Query*>(handle); }
} // extern C
