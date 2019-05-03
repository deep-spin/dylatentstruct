cmake_minimum_required(VERSION 3.1)
project(lap LANGUAGES CXX)

add_library(lap lapjv.cpp lapmod.cpp)
target_include_directories(lap
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
