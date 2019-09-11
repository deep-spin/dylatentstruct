#pragma once

/* Author: Caio Corro
 * License: MIT
 * Part of https://github.com/FilippoC/dynet-tools/
 */

#include <vector>

#include <dynet/model.h>
#include <dynet/expr.h>

namespace dy = dynet;

struct GCNSettings
{
    unsigned dim;
    unsigned layers;
    bool dense;
};

struct GCNParams {

    GCNParams(dy::ParameterCollection& pc, unsigned dim_in, unsigned dim_out);

    dy::Parameter W_parents;
    dy::Parameter b_parents;
    dy::Parameter W_children;
    dy::Parameter b_children;
    dy::Parameter W_self;
    dy::Parameter b_self;
};

struct GCNExprs {

    GCNExprs() = default;
    GCNExprs(dy::ComputationGraph& cg, GCNParams params);

    dy::Expression W_parents;
    dy::Expression b_parents;
    dy::Expression W_children;
    dy::Expression b_children;
    dy::Expression W_self;
    dy::Expression b_self;
};


struct GCNBuilder
{
    GCNBuilder(dy::ParameterCollection& pc,
               unsigned n_layers,
               unsigned dim_in,
               unsigned dim_out,
               bool dense);

    void new_graph(dy::ComputationGraph& cg, bool training);
    dy::Expression apply(const dy::Expression &input, const dy::Expression& graph);

    void set_dropout(float value);

    dy::ParameterCollection local_pc;
    std::vector<GCNParams> params;
    std::vector<GCNExprs> exprs;
    unsigned n_layers;
    bool dense;
    bool _training = true;
    float dropout_rate = 0.f;
};
