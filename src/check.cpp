#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/globals.h>
#include <dynet/io.h>
#include <dynet/timing.h>
#include <dynet/training.h>

#include <vector>
#include <iostream>
//#include <lapjv.h>

namespace dy = dynet;


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dyparams.random_seed = 42;
    dy::initialize(dyparams);

    dy::ParameterCollection p;

    const int dim = 4;
    const int vocab = 10;
    auto Wemb = p.add_lookup_parameters(vocab, {dim});

    dy::ComputationGraph cg;
    std::vector<dy::Expression> x1{ dy::lookup(cg, Wemb, 2), dy::lookup(cg, Wemb, 4) };
    auto x1avg = dy::average(x1);

    std::vector<dy::Expression> x2{ dy::lookup(cg, Wemb, 1), dy::lookup(cg, Wemb, 3) };
    auto x2avg = dy::average(x2);

    std::vector<dy::Expression> x3{ dy::lookup(cg, Wemb, 0u), dy::lookup(cg, Wemb, 5) };
    auto x3avg = dy::average(x3);

    std::cout << dy::sum_elems(x1avg + x2avg + x3avg).value();
    return 0;
}
