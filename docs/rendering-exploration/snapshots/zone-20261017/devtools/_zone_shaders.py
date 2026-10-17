"""Native angular-cell/triangle intersection shaders for idealized ToF zones.

One geometry primitive emits a conservative cell bounding rectangle. Each
fragment clips that original triangle against its cell cone, requires positive
projected area, and minimizes Euclidean distance on the clipped convex face.
No image supersampling, center-ray approximation, or minimum-range culling is
used. Clipping and minimization use float64 after the shared float32 world/view
transform; outputs and the normalized depth32 attachment use float32. Equal
representable depths retain the first triangle in the stable scene draw order.
"""

VERTEX_SHADER = r"""
#version 450 core
layout(location=0) in vec3 position;
layout(location=2) in float object_id;
layout(std430, binding=0) readonly buffer PoseData { float pose_data[]; };
layout(std430, binding=1) readonly buffer ObjectData { mat4 object_to_world[]; };
out VertexData {
    vec3 point_sensor;
    flat float identity;
    flat float render_radius;
    flat int layer;
    flat int triangle_index;
} vertex_out;
void main() {
    int i = gl_InstanceID * 17;
    mat3 sensor_to_world = mat3(
        vec3(pose_data[i], pose_data[i+4], pose_data[i+8]),
        vec3(pose_data[i+1], pose_data[i+5], pose_data[i+9]),
        vec3(pose_data[i+2], pose_data[i+6], pose_data[i+10]));
    vec3 origin = vec3(pose_data[i+3], pose_data[i+7], pose_data[i+11]);
    mat4 object_pose = object_to_world[int(object_id) - 1];
    vec3 point_world = (object_pose * vec4(position, 1.0)).xyz;
    vertex_out.point_sensor = transpose(sensor_to_world) * (point_world - origin);
    vertex_out.identity = object_id;
    vertex_out.render_radius = pose_data[i+16];
    vertex_out.layer = gl_InstanceID;
    // gl_VertexID includes the first-vertex offset for selected-object draws.
    vertex_out.triangle_index = gl_VertexID / 3;
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

_CLIP_FUNCTION = r"""
// A convex triangle clipped by five planes has at most eight vertices.
int clip_plane(in dvec3 source[12], int count, dvec3 plane,
               out dvec3 destination[12]) {
    if (count == 0) return 0;
    int result_count = 0;
    dvec3 previous = source[count - 1];
    double previous_distance = dot(plane, previous);
    bool previous_inside = previous_distance >= 0.0;
    for (int i = 0; i < count; ++i) {
        dvec3 current = source[i];
        double current_distance = dot(plane, current);
        bool current_inside = current_distance >= 0.0;
        if (previous_inside != current_inside) {
            double fraction = previous_distance / (previous_distance - current_distance);
            destination[result_count++] = previous + fraction * (current - previous);
        }
        if (current_inside) destination[result_count++] = current;
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    return result_count;
}

int clip_cone(dvec3 a, dvec3 b, dvec3 c, dvec2 low, dvec2 high,
              dvec2 tangent, out dvec3 polygon[12]) {
    dvec3 temporary[12];
    polygon[0] = a; polygon[1] = b; polygon[2] = c;
    int count = clip_plane(polygon, 3, dvec3(1.0, 0.0, 0.0), temporary);
    count = clip_plane(temporary, count, dvec3(-low.x * tangent.x, -1.0, 0.0), polygon);
    count = clip_plane(polygon, count, dvec3(high.x * tangent.x, 1.0, 0.0), temporary);
    count = clip_plane(temporary, count, dvec3(-low.y * tangent.y, 0.0, 1.0), polygon);
    count = clip_plane(polygon, count, dvec3(high.y * tangent.y, 0.0, -1.0), temporary);
    for (int i = 0; i < count; ++i) polygon[i] = temporary[i];
    return count;
}

dvec2 project_direction(dvec3 point, dvec2 tangent) {
    return dvec2(-point.y / tangent.x, point.z / tangent.y) / point.x;
}
"""

GEOMETRY_SHADER = r"""
#version 450 core
layout(triangles) in;
layout(triangle_strip, max_vertices=4) out;
in VertexData {
    vec3 point_sensor;
    flat float identity;
    flat float render_radius;
    flat int layer;
    flat int triangle_index;
} vertex_in[];
uniform vec2 tangent_fov;
uniform vec2 image_size;
flat out vec3 triangle_a;
flat out vec3 triangle_b;
flat out vec3 triangle_c;
flat out float identity;
flat out float render_radius;
flat out int triangle_index;
""" + _CLIP_FUNCTION + r"""
void emit_corner(vec2 corner) {
    triangle_a = vertex_in[0].point_sensor;
    triangle_b = vertex_in[1].point_sensor;
    triangle_c = vertex_in[2].point_sensor;
    identity = vertex_in[0].identity;
    render_radius = vertex_in[0].render_radius;
    triangle_index = vertex_in[0].triangle_index;
    gl_Layer = vertex_in[0].layer;
    gl_Position = vec4(corner, 0.0, 1.0);
    EmitVertex();
}
void main() {
    dvec3 polygon[12];
    dvec2 tangent = dvec2(tangent_fov);
    int count = clip_cone(dvec3(vertex_in[0].point_sensor),
                          dvec3(vertex_in[1].point_sensor),
                          dvec3(vertex_in[2].point_sensor),
                          dvec2(-1.0), dvec2(1.0), tangent, polygon);
    dvec2 low = dvec2(1.0), high = dvec2(-1.0);
    bool has_forward_point = false;
    for (int i = 0; i < count; ++i) {
        if (polygon[i].x <= 0.0) continue;
        dvec2 projected = project_direction(polygon[i], tangent);
        low = min(low, projected);
        high = max(high, projected);
        has_forward_point = true;
    }
    if (!has_forward_point || count < 3) return;
    // Expand by one cell to make the broad phase conservative at float edges.
    // The fragment stage rejects cells with only point/line contact.
    vec2 first = max(vec2(0.0), floor(vec2((low + 1.0) * 0.5) * image_size) - 1.0);
    vec2 last = min(image_size - 1.0, floor(vec2((high + 1.0) * 0.5) * image_size) + 1.0);
    if (any(greaterThan(first, last))) return;
    vec2 lower = 2.0 * first / image_size - 1.0;
    vec2 upper = 2.0 * (last + 1.0) / image_size - 1.0;
    emit_corner(vec2(lower.x, lower.y));
    emit_corner(vec2(upper.x, lower.y));
    emit_corner(vec2(lower.x, upper.y));
    emit_corner(vec2(upper.x, upper.y));
    EndPrimitive();
}
"""

FRAGMENT_SHADER = r"""
#version 450 core
flat in vec3 triangle_a;
flat in vec3 triangle_b;
flat in vec3 triangle_c;
flat in float identity;
flat in float render_radius;
flat in int triangle_index;
uniform vec2 tangent_fov;
uniform vec2 image_size;
layout(location=0) out vec2 measurement;
layout(location=1) out vec4 hit;
""" + _CLIP_FUNCTION + r"""
void main() {
    dvec2 low = 2.0 * floor(dvec2(gl_FragCoord.xy)) / dvec2(image_size) - 1.0;
    dvec2 high = low + 2.0 / dvec2(image_size);
    dvec2 tangent = dvec2(tangent_fov);
    dvec3 polygon[12];
    int count = clip_cone(dvec3(triangle_a), dvec3(triangle_b), dvec3(triangle_c),
                          low, high, tangent, polygon);
    if (count < 3) discard;
    // Translation before the shoelace sum avoids cancellation for thin faces
    // far from the optical axis. Exactly zero projected area never measures.
    dvec2 projected[12];
    int projected_count = 0;
    for (int i = 0; i < count; ++i) {
        if (polygon[i].x > 0.0) projected[projected_count++] = project_direction(polygon[i], tangent);
    }
    if (projected_count < 3) discard;
    double twice_area = 0.0;
    for (int i = 1; i + 1 < projected_count; ++i) {
        dvec2 a = projected[i] - projected[0];
        dvec2 b = projected[i+1] - projected[0];
        twice_area += a.x * b.y - a.y * b.x;
    }
    if (twice_area == 0.0) discard;
    dvec3 a = dvec3(triangle_a), b = dvec3(triangle_b), c = dvec3(triangle_c);
    dvec3 normal = cross(b-a, c-a);
    double norm_squared = dot(normal, normal);
    double offset = dot(normal, a);
    if (norm_squared == 0.0 || offset == 0.0) discard;
    dvec3 foot = normal * (offset / norm_squared);
    bool foot_inside = true;
    dvec3 nearest = polygon[0];
    double nearest_squared = dot(nearest, nearest);
    for (int i = 0; i < count; ++i) {
        dvec3 start = polygon[i], end = polygon[(i+1) % count];
        dvec3 edge = end - start;
        if (dot(cross(edge, foot - start), normal) < 0.0) foot_inside = false;
        double edge_squared = dot(edge, edge);
        double fraction = edge_squared > 0.0 ? clamp(-dot(start, edge) / edge_squared, 0.0, 1.0) : 0.0;
        dvec3 candidate = start + fraction * edge;
        double squared = dot(candidate, candidate);
        if (squared < nearest_squared) { nearest_squared = squared; nearest = candidate; }
    }
    if (foot_inside) { nearest = foot; nearest_squared = dot(foot, foot); }
    double radius = sqrt(nearest_squared);
    if (radius > double(render_radius)) discard;
    gl_FragDepth = float(radius / double(render_radius));
    measurement = vec2(float(radius), identity);
    hit = vec4(vec3(nearest), float(triangle_index + 1));
}
"""
