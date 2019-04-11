project(dylatentstruct)

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-pedantic")
add_definitions("-O3")
add_definitions("-Wfatal-errors")
add_definitions("-g")
add_definitions("-std=c++11")
add_definitions("-funroll-loops")
add_definitions("-DEIGEN_FAST_MATH")
add_definitions("-march=native")


set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} CACHE STRING "" FORCE)
if(NOT EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIR /home/vlad/src/eigen)
endif()

set(DYNET_DIR ${DYNET_DIR} CACHE STRING "" FORCE)
if(NOT DYNET_DIR)
    set(DYNET_DIR /home/vlad/src/dynet)
endif()


message("-- Eigen dir is " ${EIGEN3_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

message("-- dynet dir is " ${DYNET_DIR})
include_directories(${DYNET_DIR})

find_library(DYNET_LIBRARY NAMES dynet HINTS "${DYNET_DIR}/build/dynet")

include_directories("include")


add_library(
    libdylatentstruct

    src/builders/gcn.cpp
    src/builders/bilstm.cpp
)
target_link_libraries(libdylatentstruct ${DYNET_LIBRARY})

add_executable(sentclf src/sentclf.cpp)

target_link_libraries(sentclf ${DYNET_LIBRARY} libdylatentstruct)
