#pragma once

// #include <Eigen/Eigen>
#include <fstream>
#include <vector>
#include <cassert>

/*
 * Generic utils
 */

void normalize_vector(std::vector<float> & v);
unsigned line_count(const std::string filename);
