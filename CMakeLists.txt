project(dylatentstruct)

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  FIND_PACKAGE(Git)
  if(GIT_FOUND)
    EXECUTE_PROCESS(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE "GIT_COMMIT"
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    MESSAGE( STATUS "Git commit: ${GIT_COMMIT}" )
  else(GIT_FOUND)
      SET(GIT_COMMIT 0)
  endif(GIT_FOUND)
endif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

configure_file("include/git_commit.h.in" "include/git_commit.h")

add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-pedantic")
# add_definitions("-O3")
add_definitions("-Wfatal-errors")
add_definitions("-g")
add_definitions("-std=c++11")
add_definitions("-funroll-loops")
# add_definitions("-DEIGEN_FAST_MATH")
add_definitions("-march=native")

add_subdirectory(opt)

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

include_directories(${JSON_INCLUDE_DIRS})
include_directories(${CPR_INCLUDE_DIRS})
include_directories("include")#
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")#

add_library(
    libdylatentstruct

    src/builders/gcn.cpp
    src/builders/bilstm.cpp
)
target_link_libraries(libdylatentstruct ${DYNET_LIBRARY} ${CPR_LIBRARIES})

set(ALL_LIBS
    libdylatentstruct
    ${DYNET_LIBRARY}
    ${CPR_LIBRARIES}
    ${JSON_LIBRARIES}
)

add_executable(sentclf src/sentclf.cpp)
add_executable(check_gcn_nondet src/check_gcn_nondet.cpp)
add_executable(mlflowtest src/mlflowtest.cpp)

target_link_libraries(mlflowtest ${ALL_LIBS})
target_link_libraries(sentclf ${ALL_LIBS})
target_link_libraries(check_gcn_nondet ${ALL_LIBS})
