#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>

#include <iostream>

#include "layers/arcs-to-adj.h"

namespace dy = dynet;


void test_arcs_to_adj(unsigned int size)
{
    dy::ParameterCollection m;

    unsigned input_dim = (size - 1) * (size - 1);
    auto u_p = m.add_parameters({input_dim}, 0, "X");

    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
        {
            dy::ComputationGraph cg;
            auto u = dy::parameter(cg, u_p);
            auto adj = dy::arcs_to_adj(u, size);
            auto z = dy::pick(dy::pick(adj, i), j);
            // cg.forward(z);
            cg.backward(z);
            dy::check_grad(m, z, 1);
        }
}

void test_adj_to_arcs(unsigned int size)
{
    dy::ParameterCollection m;

    auto Gp = m.add_parameters({size, size}, 0, "X");

    unsigned output_dim = (size - 1) * (size - 1);

    for (size_t i = 0; i < output_dim; ++i)
    {
        dy::ComputationGraph cg;
        auto G = dy::parameter(cg, Gp);
        auto u = dy::adj_to_arcs(G);
        auto z = dy::pick(u, i);
        // cg.forward(z);
        cg.backward(z);
        dy::check_grad(m, z, 1);
    }
}


int main(int argc, char** argv)
{
    dy::initialize(argc, argv);

    const unsigned int size = 7;
    std::cout << "arcs to adj" << std::endl;
    test_arcs_to_adj(size);
    std::cout << "adj to arcs" << std::endl;
    test_adj_to_arcs(size);
}
