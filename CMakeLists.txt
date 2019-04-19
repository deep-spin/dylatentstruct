project(dylatentstruct)

cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  execute_process( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
endif()


# Store git commit in header file
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

set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} CACHE STRING "" FORCE)
if(NOT EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIR /home/vlad/src/eigen)
endif()

find_package(Dynet REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(DySparseMAP REQUIRED)

add_subdirectory(opt)

add_library(
    dylatentstruct

    include/args.h
    include/crayon.h
    include/mlflow.h
    include/utils.h
    include/data.h

    include/models/basemodel.h
    include/models/sentclf_model.h
    include/models/basenli.h
    include/models/esim.h

    include/builders/adjmatrix.h
    include/builders/arcscorers.h
    include/builders/bilstm.h
    include/builders/gcn.h
    include/builders/biattn.h

    src/data.cpp
    src/builders/bilstm.cpp
    src/builders/gcn.cpp
    src/builders/biattn.cpp
)

target_link_libraries(dylatentstruct
    DySparseMAP::ad3qp
    DySparseMAP::dysparsemap
    Eigen3::Eigen
    Dynet::dynet
    nlohmann_json::nlohmann_json
    cpr
)

target_include_directories(
    dylatentstruct

    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>

    PRIVATE
    src
)

target_compile_options(dylatentstruct PUBLIC
    -Ofast
#     -Werror
    -Wall
    -Wextra
    -pedantic
    -funroll-loops
    -march=native)

add_executable(sentclf src/sentclf.cpp)
add_executable(esim src/esim.cpp)
# add_executable(check_gcn_nondet src/check_gcn_nondet.cpp)
# add_executable(mlflowtest src/mlflowtest.cpp)

target_link_libraries(sentclf PUBLIC dylatentstruct)
target_link_libraries(esim PUBLIC dylatentstruct)
#target_link_libraries(mlflowtest PUBLIC dylatentstruct)
#target_link_libraries(check_gcn_nondet PUBLIC dylatentstruct)
