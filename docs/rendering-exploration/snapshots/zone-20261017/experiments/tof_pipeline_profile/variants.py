"""Experimental shader substitutions for profiling the angular-zone renderer.

This module does not alter the production renderer.  An ablation measures the
cost of the remaining program, not an additive attribution of the removed code:
the GLSL compiler may make different scheduling/register-allocation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from devtools import _zone_shaders as source


@dataclass(frozen=True)
class ShaderVariant:
    name: str
    vertex_shader: str
    geometry_shader: str
    fragment_shader: str
    semantic_status: str
    description: str
    caveats: tuple[str, ...] = ()
    rasterizer_discard: bool = False

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "semantic_status": self.semantic_status,
            "description": self.description,
            "caveats": list(self.caveats),
            "rasterizer_discard": self.rasterizer_discard,
        }


_CHEAP_FRAGMENT = r"""
#version 450 core
flat in vec3 triangle_a;
flat in vec3 triangle_b;
flat in vec3 triangle_c;
flat in float identity;
flat in float render_radius;
flat in int triangle_index;
layout(location=0) out vec2 measurement;
layout(location=1) out vec4 hit;
void main() {
    // Deliberately invalid measurements. Keep all primitive data live while
    // removing per-cell clipping, coverage, nearest-point and range work.
    gl_FragDepth = 0.5;
    measurement = vec2(render_radius * 0.5, identity);
    hit = vec4((triangle_a + triangle_b + triangle_c) / 3.0,
               float(triangle_index + 1));
}
"""


_CLIPPED_POLYGON_PLACEHOLDER = r"""
    // Invalid sensor output, solely to keep every clipped vertex observable.
    // The same sum is used by both clip and clip+area ablations so the added
    // placeholder work is controlled, though not zero-cost.
    dvec3 diagnostic_sum = dvec3(0.0);
    for (int i = 0; i < count; ++i) diagnostic_sum += polygon[i];
    gl_FragDepth = 0.5;
    measurement = vec2(render_radius * 0.5, identity);
    hit = vec4(vec3(diagnostic_sum / double(count)), float(triangle_index + 1));
}
"""


def _fragment_through(marker: str) -> str:
    shader = source.FRAGMENT_SHADER
    if shader.count(marker) != 1:
        raise RuntimeError("production fragment shader shape changed")
    prefix, _ = shader.split(marker, 1)
    return prefix + marker + "\n" + _CLIPPED_POLYGON_PLACEHOLDER


_PLANE_FAST_FUNCTION = r"""
// Same per-vertex plane distances and intersection expressions as production.
// Avoid the branchy intersection loop when a whole convex polygon lies on one
// side of a plane; cached distances also avoid reevaluating the final vertex.
int clip_plane(in dvec3 source[12], int count, dvec3 plane,
               out dvec3 destination[12]) {
    if (count == 0) return 0;
    double distances[12];
    bool all_inside = true;
    bool any_inside = false;
    for (int i = 0; i < count; ++i) {
        distances[i] = dot(plane, source[i]);
        bool inside = distances[i] >= 0.0;
        all_inside = all_inside && inside;
        any_inside = any_inside || inside;
    }
    if (!any_inside) return 0;
    if (all_inside) {
        for (int i = 0; i < count; ++i) destination[i] = source[i];
        return count;
    }
    int result_count = 0;
    dvec3 previous = source[count - 1];
    double previous_distance = distances[count - 1];
    bool previous_inside = previous_distance >= 0.0;
    for (int i = 0; i < count; ++i) {
        dvec3 current = source[i];
        double current_distance = distances[i];
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

"""


_CONE_TRIVIAL_PREFIX = r"""
    // A linear half-space contains the complete triangle iff it contains all
    // three vertices. Strictly outside all three vertices is an exact reject.
    // Plane coefficients and dot expressions match clip_plane below. Boundary
    // vertices (distance == 0) remain inside, exactly as in production.
    dvec3 planes[5] = dvec3[5](
        dvec3(1.0, 0.0, 0.0),
        dvec3(-low.x * tangent.x, -1.0, 0.0),
        dvec3(high.x * tangent.x, 1.0, 0.0),
        dvec3(-low.y * tangent.y, 0.0, 1.0),
        dvec3(high.y * tangent.y, 0.0, -1.0));
    bool all_inside = true;
    for (int p = 0; p < 5; ++p) {
        double da = dot(planes[p], a);
        double db = dot(planes[p], b);
        double dc = dot(planes[p], c);
        if (da < 0.0 && db < 0.0 && dc < 0.0) return 0;
        all_inside = all_inside && da >= 0.0 && db >= 0.0 && dc >= 0.0;
    }
    if (all_inside) {
        polygon[0] = a; polygon[1] = b; polygon[2] = c;
        return 3;
    }
"""


def _fast_clip(plane_fast: bool, cone_fast: bool) -> str:
    clip = source._CLIP_FUNCTION
    if plane_fast:
        start = clip.index("int clip_plane(")
        end = clip.index("int clip_cone(")
        clip = clip[:start] + _PLANE_FAST_FUNCTION + clip[end:]
    if cone_fast:
        needle = "    dvec3 temporary[12];"
        if clip.count(needle) != 1:
            raise RuntimeError("production clip_cone shape changed")
        clip = clip.replace(needle, _CONE_TRIVIAL_PREFIX + needle, 1)
    return clip


def _replace_clip(shader: str, clip: str) -> str:
    if shader.count(source._CLIP_FUNCTION) != 1:
        raise RuntimeError("production shader no longer embeds one shared clip helper")
    return shader.replace(source._CLIP_FUNCTION, clip, 1)


def _float32(shader: str) -> str:
    return re.sub(r"\bdvec([234])\b", r"vec\1", re.sub(r"\bdouble\b", "float", shader))


_SINGLE_CELL_FUNCTION = r"""
bool single_cell_strict(dvec2 tangent, out vec2 cell) {
    dvec3 a = dvec3(vertex_in[0].point_sensor);
    dvec3 b = dvec3(vertex_in[1].point_sensor);
    dvec3 c = dvec3(vertex_in[2].point_sensor);
    if (!(a.x > 0.0 && b.x > 0.0 && c.x > 0.0)) return false;
    dvec2 pa = project_direction(a, tangent);
    dvec2 pb = project_direction(b, tangent);
    dvec2 pc = project_direction(c, tangent);
    dvec2 image = dvec2(image_size);
    dvec2 first_cell = floor((min(pa, min(pb, pc)) + 1.0) * 0.5 * image);
    dvec2 last_cell = floor((max(pa, max(pb, pc)) + 1.0) * 0.5 * image);
    if (any(notEqual(first_cell, last_cell)) ||
        any(lessThan(first_cell, dvec2(0.0))) ||
        any(greaterThanEqual(first_cell, image))) return false;
    dvec2 low = 2.0 * first_cell / image - 1.0;
    dvec2 high = low + 2.0 / image;
    dvec3 planes[4] = dvec3[4](
        dvec3(-low.x * tangent.x, -1.0, 0.0),
        dvec3(high.x * tangent.x, 1.0, 0.0),
        dvec3(-low.y * tangent.y, 0.0, 1.0),
        dvec3(high.y * tangent.y, 0.0, -1.0));
    double scale_a = max(1.0, max(abs(a.x), max(abs(a.y), abs(a.z))));
    double scale_b = max(1.0, max(abs(b.x), max(abs(b.y), abs(b.z))));
    double scale_c = max(1.0, max(abs(c.x), max(abs(c.y), abs(c.z))));
    for (int p = 0; p < 4; ++p) {
        double plane_scale = abs(planes[p].x) + abs(planes[p].y) + abs(planes[p].z);
        double margin = 1e-10 * plane_scale;
        if (!(dot(planes[p], a) > margin * scale_a &&
              dot(planes[p], b) > margin * scale_b &&
              dot(planes[p], c) > margin * scale_c)) return false;
    }
    cell = vec2(first_cell);
    return true;
}
"""


def _single_cell_geometry() -> str:
    # Leave the original full-view clip, projected bounds, and conservative
    # rectangle computation in place. Override only an independently certified
    # one-cell case, otherwise fall through to the byte-identical original box.
    main_needle = "void main() {\n    dvec3 polygon[12];"
    bbox_needle = "    if (any(greaterThan(first, last))) return;"
    shader = source.GEOMETRY_SHADER
    if shader.count(main_needle) != 1 or shader.count(bbox_needle) != 1:
        raise RuntimeError("production geometry shader shape changed")
    shader = shader.replace(main_needle, _SINGLE_CELL_FUNCTION + "\n" + main_needle, 1)
    return shader.replace(bbox_needle, bbox_needle + r"""
    vec2 certified_cell;
    if (single_cell_strict(tangent, certified_cell)) {
        first = certified_cell;
        last = certified_cell;
    }
""", 1)


_RANGE_AABB_FUNCTION = r"""
bool triangle_beyond_range() {
    dvec3 a = dvec3(vertex_in[0].point_sensor);
    dvec3 b = dvec3(vertex_in[1].point_sensor);
    dvec3 c = dvec3(vertex_in[2].point_sensor);
    dvec3 low = min(a, min(b, c));
    dvec3 high = max(a, max(b, c));
    dvec3 nearest_box_point = clamp(dvec3(0.0), low, high);
    double lower_squared = dot(nearest_box_point, nearest_box_point);
    double radius = double(vertex_in[0].render_radius);
    double radius_squared = radius * radius;
    double guard = 1e-10 * max(1.0, max(lower_squared, radius_squared));
    return lower_squared > radius_squared + guard;
}
"""


def _range_aabb_geometry(shader: str) -> str:
    main_needle = "void main() {\n    dvec3 polygon[12];"
    if shader.count(main_needle) != 1:
        raise RuntimeError("production geometry shader shape changed")
    return shader.replace(main_needle, _RANGE_AABB_FUNCTION + r"""
void main() {
    if (triangle_beyond_range()) return;
    dvec3 polygon[12];""", 1)


def get_variants() -> dict[str, ShaderVariant]:
    """Return isolated variants; baseline shader strings are byte-identical.

    ``rasterizer_discard`` is a requested GL state, not a shader rewrite. The
    driver must restore GL_RASTERIZER_DISCARD after that draw. Outputs from any
    diagnostic ablation must never be accepted as valid sensor measurements.
    """
    result: dict[str, ShaderVariant] = {}

    def add(name: str, *, geometry: str | None = None,
            fragment: str | None = None, status: str, description: str,
            caveats: tuple[str, ...] = (), discard: bool = False) -> None:
        result[name] = ShaderVariant(
            name=name, vertex_shader=source.VERTEX_SHADER,
            geometry_shader=source.GEOMETRY_SHADER if geometry is None else geometry,
            fragment_shader=source.FRAGMENT_SHADER if fragment is None else fragment,
            semantic_status=status, description=description,
            caveats=caveats, rasterizer_discard=discard)

    add("baseline", status="production-identical",
        description="Unmodified production vertex/geometry/fragment shaders.")
    add("cheap_fragment", fragment=_CHEAP_FRAGMENT, status="diagnostic-non-equivalent",
        description="Keep conservative GS rectangles; replace exact cell work with invalid cheap output.",
        caveats=("Measures a remaining-program cost, not an additive FS cost.",
                 "Coverage, distance, ID selection and visibility are invalid.",
                 "Compiler register pressure and depth-test behavior may differ."))
    for name, marker, description in (
        ("clip_only_fragment", "    if (count < 3) discard;",
         "Original FS through exact cell-cone clip and polygon vertex-count rejection; invalid placeholder afterwards."),
        ("clip_area_only_fragment", "    if (twice_area == 0.0) discard;",
         "Original FS through exact cell-cone clip, projected-area construction and zero-area rejection; invalid placeholder afterwards."),
    ):
        add(name, fragment=_fragment_through(marker), status="diagnostic-non-equivalent",
            description=description,
            caveats=("Remaining-program timing only; differences are not additive isolated phase costs.",
                     "A shared extra polygon-vertex sum keeps every clipped point live and adds a small nonzero cost.",
                     "Nearest-point minimization, maximum-range rejection, depth and ID selection are invalid.",
                     "Compiler scheduling, local-array allocation and depth-test behavior may differ from production."))
    add("rasterizer_discard", status="diagnostic-non-equivalent",
        description="Use unchanged program and GL_RASTERIZER_DISCARD; no fragment stage or outputs.",
        caveats=("Deleting EmitVertex would let the compiler remove all unused GS clip work.",
                 "Runtime discard avoids that source rewrite, but hardware may still skip work.",
                 "This is a front-end lower-bound comparison, not exact isolated GS attribution."),
        discard=True)

    add("single_cell_tight_gs", geometry=_single_cell_geometry(),
        status="geometry-equivalent-candidate",
        description=("Keep original FP64 full-view GS clip and original FS; emit one cell only "
                     "when the original triangle is certified strictly inside that cell cone."),
        caveats=("All original vertices must be forward, project into the same legal cell, and be strictly inside all four cone sides.",
                 "Each side requires dot > 1e-10 * L1(plane) * max(1, max_abs(vertex)); uncertain/boundary cases use the original expanded rectangle.",
                 "A triangle is convex; strict vertex containment proves no positive-area overlap with another cell, in exact arithmetic.",
                 "The margin is an engineering guard, not a formal bound on every GPU compiler transformation; validate boundary cases and full-output parity.",
                 "No geometry, arithmetic precision, primitive order, fragment coverage test, nearest hit or depth/ID tie rule is changed."))

    range_caveats = (
        "A triangle is contained in its sensor-space AABB; distance to that AABB is a lower bound on distance to every triangle point.",
        "Reject only lower_sq > max_range_sq + 1e-10 * max(1, lower_sq, max_range_sq); boundary cases fall through.",
        "Only maximum range is used; minimum-range occluders are not removed.",
        "FP64 margin guards ordinary rounding but is not a formal GPU-compiler error bound; validate exact outputs and range-boundary cases.",
        "No mesh, transform, primitive order, surviving fragment algorithm, nearest hit or depth/ID tie rule is changed.",
    )
    add("range_aabb_gs", geometry=_range_aabb_geometry(source.GEOMETRY_SHADER),
        status="geometry-equivalent-candidate",
        description="Conservatively reject triangles whose AABB is strictly outside the maximum-range sphere; original FS.",
        caveats=range_caveats)
    add("range_aabb_single_cell_gs", geometry=_range_aabb_geometry(_single_cell_geometry()),
        status="geometry-equivalent-candidate",
        description="Combine maximum-range AABB rejection with strictly certified one-cell rectangles; original FS.",
        caveats=range_caveats + ("Also uses the single_cell_tight_gs containment certificate and its conservative boundary fallback.",))

    for stages in ("gs", "fs", "both"):
        add(f"fp32_{stages}",
            geometry=_float32(source.GEOMETRY_SHADER) if stages in ("gs", "both") else None,
            fragment=_float32(source.FRAGMENT_SHADER) if stages in ("fs", "both") else None,
            status="diagnostic-precision-changed",
            description=f"Keep the same geometric algorithm; change FP64 to FP32 in {stages}.",
            caveats=("Floating-point boundary, thin-area, distance and tie behavior may change.",
                     "Performance evidence only; not an approved production-equivalent optimization."))

    for family, plane_fast, cone_fast in (
        ("clip_plane_fast", True, False),
        ("clip_cone_fast", False, True),
        ("clip_fast", True, True),
    ):
        clip = _fast_clip(plane_fast, cone_fast)
        for stages in ("gs", "fs", "both"):
            add(f"{family}_{stages}",
                geometry=_replace_clip(source.GEOMETRY_SHADER, clip) if stages in ("gs", "both") else None,
                fragment=_replace_clip(source.FRAGMENT_SHADER, clip) if stages in ("fs", "both") else None,
                status="geometry-equivalent-candidate",
                description=(f"FP64 {family} trivial accept/reject in {stages}; "
                             "unchanged partial-clip intersection, coverage, nearest point and winners."),
                caveats=("Same half-space inequalities; strict negative whole-triangle reject is convexity-safe.",
                         "Partially clipped intersections keep production arithmetic order.",
                         "Compiler code generation can still change rounding; verify full output parity and boundary cases.",
                         "Extra pretests and local arrays can cost more than they save; measure before selection."))
    return result


STATIC_ANALYSIS = {
    "invariant": "Every positive-area triangle/cell-cone overlap participates; choose nearest Euclidean hit and its ID.",
    "front_end": (
        "All scene triangles are transformed and processed once for every pose; there is no range/spatial broad phase.",
        "Each GS invocation FP64-clips a triangle against five view planes before its projected cell bounding box.",
        "The cell rectangle expands one cell on every side; a triangle contained in one interior cell can generate nine cell fragments.",
    ),
    "fragment": (
        "Every candidate triangle/cell pair repeats five-plane FP64 clipping, projected-area filtering and exact nearest-point work.",
        "Local fixed arrays and variable loops may produce register pressure or spills; timing alone cannot establish actual spills.",
        "Depth is explicitly written only after nearest-radius calculation; regular early-Z rejection must not be assumed.",
        "FP64 cost is a hypothesis; compare isolated GS/FS precision ablations before assigning causality.",
    ),
    "safe_direction": (
        "Use convex half-space trivial accept/reject before full clipping, keeping the current partial clip algorithm.",
        "A conservative triangle/object/BVH broad phase can reject triangles outside the cell cone or max-range sphere without approximating surviving intersections.",
        "Tightening the expanded bbox needs a proof and robust outward rounding; simply deleting the one-cell guard is not accepted.",
        "Nearest-point and projected-area algorithm simplification requires preserving zero-area rejection, within-cell minimum and stable depth/ID ties.",
    ),
}
