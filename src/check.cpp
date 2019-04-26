#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/globals.h>
#include <dynet/io.h>
#include <dynet/timing.h>
#include <dynet/training.h>

//#include "builders/gcn.h"
//#include "builders/adjmatrix.h"
#include <vector>


namespace dy = dynet;

int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    dyparams.random_seed = 42;
    dyparams.autobatch = true;
    dy::initialize(dyparams);

    dy::ComputationGraph cg;
    auto x = dy::random_normal(cg, { 6 });
    auto z = dy::zeros(cg, { 1 });

    auto xx = dy::concatenate({z, x});
    std::cout << xx.value() << std::endl << std::endl;

    std::vector<unsigned> ix = {1, 2, 0, 3, 4, 0, 5, 6, 0};
    auto out = dy::select_rows(xx, ix);
    std::cout << out.value() << std::endl << std::endl;

    std::cout << dy::reshape(out, {3, 3}).value() << std::endl;


    /*
    dy::ParameterCollection p;
    const size_t dim = 300;

    GCNSettings gcn_settings{dim, 1, false};
    GCNBuilder gcn(p, gcn_settings, dim);
    auto tree = FlatAdjacency();


    std::vector<int> heads = {-1, 3, 2, 3, 4};

    for (int i = 0; i < 3; ++i)
    {
        dy::ComputationGraph cg;
        tree.new_graph(cg);
        gcn.new_graph(cg, true, true);

        auto X = dy::random_uniform(cg, {dim, 5}, -1, 1);
        auto z = dy::random_uniform(cg, {5}, -1, 1);

        auto G = tree.make_adj({}, heads);
        //std::cout << X.value() << std::endl << std::endl;
        //std::cout << G.value() << std::endl << std::endl;

        auto Y = gcn.apply(X, G);

        auto ls = dy::mean_dim(Y * z, {0});
        cg.backward(ls);
        std::cout << ls.value() << std::endl;
        std::cout << p.gradient_l2_norm() << std::endl;

    }
    */

    return 0;
}
