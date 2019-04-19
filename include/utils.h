#pragma once

// #include <Eigen/Eigen>
#include <fstream>
#include <cassert>

/*
 * Generic utils
 */


unsigned line_count(const std::string filename)
{
    std::ifstream in(filename);
    assert(in);
    std::string line;

    unsigned lines = 0;
    while (getline(in, line))
        ++lines;

    return lines;
}
