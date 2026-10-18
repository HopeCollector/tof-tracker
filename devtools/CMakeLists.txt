cmake_minimum_required(VERSION 3.16)
project(mesh_distance LANGUAGES CXX)

add_library(mesh_distance SHARED _mesh_distance.cpp)
target_compile_features(mesh_distance PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(mesh_distance PRIVATE -Wall -Wextra -Wpedantic)
endif()
