"""Untimed shader counters; their contended atomics are never performance data."""
import struct
from devtools import _zone_shaders as original

NAMES = ("fs_entries", "fs_polygon_count_ge3", "fs_positive_projected_area",
         "fs_nondegenerate_plane", "fs_within_render_radius", "gs_input_triangles",
         "gs_emitted_rectangles", "gs_rectangle_cell_sum")
DECLARATION = "\nlayout(std430, binding=2) buffer ProfileCounts { uint profile_counts[16]; };\n"


def _add(shader, needle, replacement):
    assert shader.count(needle) == 1, needle
    return shader.replace(needle, replacement)


def sources():
    gs, fs = original.GEOMETRY_SHADER, original.FRAGMENT_SHADER
    for name, text in (("gs", gs), ("fs", fs)):
        text = _add(text, "#version 450 core", "#version 450 core" + DECLARATION)
        if name == "gs":
            gs = text
        else:
            fs = text
    # Buckets: all inputs plus target only. IDs keep their original scene order.
    gs = _add(gs, "void main() {", "void main() {\n    atomicAdd(profile_counts[5], 1u);\n    if (int(vertex_in[0].identity) == profile_target_id) atomicAdd(profile_counts[13], 1u);")
    gs = _add(gs, "    vec2 lower =", "    atomicAdd(profile_counts[6], 1u);\n    atomicAdd(profile_counts[7], uint((last.x-first.x+1.0)*(last.y-first.y+1.0)));\n    if (int(vertex_in[0].identity) == profile_target_id) {\n        atomicAdd(profile_counts[14], 1u);\n        atomicAdd(profile_counts[15], uint((last.x-first.x+1.0)*(last.y-first.y+1.0)));\n    }\n    vec2 lower =")
    gs = _add(gs, DECLARATION, DECLARATION + "uniform int profile_target_id;\n")
    fs = _add(fs, DECLARATION, DECLARATION + "uniform int profile_target_id;\nvoid count_stage(int index) {\n    atomicAdd(profile_counts[index], 1u);\n    if (int(identity) == profile_target_id) atomicAdd(profile_counts[index+8], 1u);\n}\n")
    # identity must be declared before count_stage is parsed by GLSL.
    start = fs.index("void count_stage(")
    stop = fs.index("}\n", start) + 2
    function = fs[start:stop]
    fs = fs[:start] + fs[stop:]
    fs = _add(fs, "void main() {", function + "\nvoid main() {\n    count_stage(0);")
    fs = _add(fs, "    if (count < 3) discard;", "    if (count < 3) discard;\n    count_stage(1);")
    fs = _add(fs, "    if (twice_area == 0.0) discard;", "    if (twice_area == 0.0) discard;\n    count_stage(2);")
    fs = _add(fs, "    if (norm_squared == 0.0 || offset == 0.0) discard;", "    if (norm_squared == 0.0 || offset == 0.0) discard;\n    count_stage(3);")
    fs = _add(fs, "    if (radius > double(render_radius)) discard;", "    if (radius > double(render_radius)) discard;\n    count_stage(4);")
    return {"vertex_shader": original.VERTEX_SHADER, "geometry_shader": gs, "fragment_shader": fs}


def decode(data):
    values = struct.unpack("16I", data)
    all_values = dict(zip(NAMES, values[:8]))
    target = dict(zip(NAMES, values[8:]))
    return {"all": all_values, "target": target,
            "map": {key: all_values[key]-target[key] for key in NAMES},
            "scope": "Untimed shader atomics count non-helper side effects. GL FS invocation statistics are separate and can include helper/quad execution. No pixel/area resampling."}
