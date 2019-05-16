#pragma once

#include "dynet/expr.h"

namespace dy = dynet;

struct DistanceBiasBuilder
{

    DistanceBiasBuilder(dy::ParameterCollection& pc,
            bool active,
            unsigned bucket = 10,
            bool dir_sensitive = true);

    dy::Expression compute(dy::Expression& input);

    dy::LookupParameter lp_distance;
    bool active;
    unsigned bucket;
    unsigned dir_sensitive;
};

