#pragma once

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/nodes-def-macros.h>
#include <dynet/nodes.h>

namespace dynet {

dynet::Expression
arcs_to_adj(const dynet::Expression& eta_u, unsigned size);

dynet::Expression
adj_to_arcs(const dynet::Expression& G);

struct ArcsToAdj : public dynet::Node
{
    explicit ArcsToAdj(const std::initializer_list<dynet::VariableIndex>&,
                       unsigned size);

    DYNET_NODE_DEFINE_DEV_IMPL()

    unsigned size;
};

struct AdjToArcs : public dynet::Node
{
    explicit AdjToArcs(const std::initializer_list<dynet::VariableIndex>&);
    DYNET_NODE_DEFINE_DEV_IMPL()

    unsigned size;
};

}
