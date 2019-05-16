#include "builders/distance-bias.h"
#include <vector>

#include <dynet/param-init.h>


std::vector<unsigned>
bias_indices(const unsigned size,
             const unsigned bucket,
             const bool dir_sensitive)
{
    std::vector<unsigned> v;
    v.reserve(size * size);
    for (unsigned i = 0u; i < size; ++i) {
        for (unsigned j = 0u; j < size; ++j) {
            if (i <= j) {
                const unsigned d = std::min(bucket, j - i);
                v.push_back(d);
            } else {
                const unsigned d = std::min(bucket, i - j);
                if (dir_sensitive)
                    v.push_back(bucket + d);
                else
                    v.push_back(d);
            }
        }
    }
    return v;
}

DistanceBiasBuilder::DistanceBiasBuilder(dy::ParameterCollection& pc,
                                         bool active,
                                         unsigned bucket,
                                         bool dir_sensitive)
  : active(active)
  , bucket(bucket)
  , dir_sensitive(dir_sensitive)
{
    if (active)
        lp_distance =
          pc.add_lookup_parameters(1 + bucket * (dir_sensitive ? 2 : 1),
                                   { 1 },
                                   dy::ParameterInitConst(0.f));
}

dy::Expression
DistanceBiasBuilder::compute(dy::Expression& input)
{
    unsigned size = input.dim()[0];
    dy::Expression bias;
    if (active) {
        auto v = bias_indices(size, bucket, dir_sensitive);

        std::cout << std::endl;
        bias = dy::lookup(*input.pg, lp_distance, v);
        bias = dy::reshape(bias, { size, size });
    }

    auto output = input;

    if (active)
        output = output + bias;

    return output;
}

