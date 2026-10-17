cmake_minimum_required(VERSION 3.16)
project(adaptive_distance_export LANGUAGES CXX)

set(SDFLIB_SOURCE_DIR "/opt/SdfLib" CACHE PATH "Read-only SdfLib checkout")
if(NOT EXISTS "${SDFLIB_SOURCE_DIR}/include/SdfLib/OctreeSdf.h")
    message(FATAL_ERROR "Set SDFLIB_SOURCE_DIR to the pinned SdfLib checkout")
endif()

# This revision selects interpolation with a source alias, not a CMake option.
# Fail explicitly if another checkout would silently export a different layout.
file(STRINGS "${SDFLIB_SOURCE_DIR}/src/sdf/OctreeSdf.cpp" interpolation_alias
     REGEX "^[ \t]*typedef[ \t]+[A-Za-z0-9_:]+[ \t]+InterpolationMethod[ \t]*;")
list(LENGTH interpolation_alias interpolation_alias_count)
if(NOT interpolation_alias_count EQUAL 1)
    message(FATAL_ERROR "Expected exactly one active interpolation alias")
endif()
# file(STRINGS) escapes C++ semicolons to preserve the CMake list; list(GET)
# decodes that element before the ordinary string comparison.
list(GET interpolation_alias 0 interpolation_alias)
string(STRIP "${interpolation_alias}" interpolation_alias)
if(NOT interpolation_alias STREQUAL "typedef TriCubicInterpolation InterpolationMethod;")
    message(FATAL_ERROR "Expected active TriCubicInterpolation alias; found: ${interpolation_alias}")
endif()
file(SHA256 "${SDFLIB_SOURCE_DIR}/src/sdf/OctreeSdf.cpp" octree_source_sha256)
file(SHA256 "${SDFLIB_SOURCE_DIR}/include/SdfLib/InterpolationMethods.h" interpolation_source_sha256)

# The Docker image supplies these three dependencies. No FetchContent downloads
# are needed, and the small OBJ reader below avoids an Assimp dependency.
set(SDFLIB_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(SDFLIB_BUILD_DEBUG_APPS OFF CACHE BOOL "" FORCE)
set(SDFLIB_USE_ASSIMP OFF CACHE BOOL "" FORCE)
set(SDFLIB_USE_ENOKI OFF CACHE BOOL "" FORCE)
set(SDFLIB_USE_SYSTEM_ASSIMP OFF CACHE BOOL "" FORCE)
set(SDFLIB_USE_SYSTEM_GLM ON CACHE BOOL "" FORCE)
set(SDFLIB_USE_SYSTEM_SPDLOG ON CACHE BOOL "" FORCE)
set(SDFLIB_USE_SYSTEM_CEREAL ON CACHE BOOL "" FORCE)
set(SDFLIB_USE_OPENMP ON CACHE BOOL "" FORCE)
add_subdirectory("${SDFLIB_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/sdflib" EXCLUDE_FROM_ALL)

add_executable(export_sdflib export_sdflib.cpp)
target_compile_features(export_sdflib PRIVATE cxx_std_17)
target_link_libraries(export_sdflib PRIVATE SdfLib)
target_compile_definitions(export_sdflib PRIVATE
    SDFLIB_EXPECTED_REVISION="8db373ef71d65be24badf6ae10750a932bbc223b"
    SDFLIB_OCTREE_SOURCE_SHA256="${octree_source_sha256}"
    SDFLIB_INTERPOLATION_SOURCE_SHA256="${interpolation_source_sha256}")

# Raw-value regression for the actual-map adaptation; no octree or GPU needed.
add_executable(mesh_value_probe mesh_value_probe.cpp)
target_compile_features(mesh_value_probe PRIVATE cxx_std_17)
target_link_libraries(mesh_value_probe PRIVATE SdfLib)
target_compile_definitions(mesh_value_probe PRIVATE
    SDFLIB_INTERPOLATION_SOURCE_SHA256="${interpolation_source_sha256}")
