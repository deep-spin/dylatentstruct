#include <vector>

#include <dynet/dynet.h>
#include <dynet/expr.h>

#include "builders/biattn.h"
#include "sparsemap.h"
#include "data.h"

namespace dy = dynet;

int main(int argc, char** argv)
{
    dy::initialize(argc, argv);

    dy::SparseMAPOpts opts;
    opts.eta = 0.5;
    opts.max_iter = 10;
    opts.log_stream = std::make_shared<std::ofstream>("log.txt");

    std::vector<int> pp = {-1};
    std::vector<int> ph = {-1};

    auto match_builder = XORMatchingBuilder{ opts };

    {
        dy::ComputationGraph cg;
        match_builder.new_graph(cg, true);

        //auto dim = dy::Dim{2, 3};
        auto dim = dy::Dim{3, 2};
        auto data = std::vector<float>{1, 0, 0, 0, 0, 3};
        auto x = dy::input(cg, dim, data);
        std::cout << x.value() << std::endl << std::endl;

        auto y = match_builder.attend(x, pp, ph);
        auto sum = dy::reshape(dy::sum_dim(y, { 0 }), {1, 2});
        std::cout << y.value() << std::endl << std::endl;
        y = dy::cdiv(y, sum);
        std::cout << y.value() << std::endl << std::endl;

    }
    return 0;
}


