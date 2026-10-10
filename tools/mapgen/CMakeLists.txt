cmake_minimum_required(VERSION 3.16)
project(offline_mapgen LANGUAGES CXX)

find_package(PCL REQUIRED COMPONENTS common)

add_executable(offline_mapgen src/main.cpp src/maps.cpp src/perlinnoise.cpp)
target_compile_features(offline_mapgen PRIVATE cxx_std_17)
target_include_directories(offline_mapgen PRIVATE include ${PCL_INCLUDE_DIRS})
target_compile_options(offline_mapgen PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(offline_mapgen PRIVATE ${PCL_LIBRARIES})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(offline_mapgen PRIVATE -Wall -Wextra -Wpedantic)
endif()
