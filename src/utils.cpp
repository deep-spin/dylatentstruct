#include <Eigen/Eigen>
#include "utils.h"


void normalize_vector(std::vector<float>& v)
{
    Eigen::Map<Eigen::RowVectorXf> v_map(v.data(), v.size());
    v_map.normalize();
}


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
