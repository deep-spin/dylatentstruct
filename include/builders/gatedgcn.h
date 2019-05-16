#pragma once

#include <vector>

#include <dynet/expr.h>
#include <dynet/model.h>

namespace dy = dynet;

struct GatedGCNParams {

    GatedGCNParams(dy::ParameterCollection& pc, unsigned dim);

    dy::Parameter W_parent;
    dy::Parameter b_parent;
    dy::Parameter W_children;
    dy::Parameter b_children;
    dy::Parameter W_gate_old;
    dy::Parameter W_gate_new;
    dy::Parameter b_gate;
};

struct GatedGCNExprs {

    GatedGCNExprs() = default;
    GatedGCNExprs(dy::ComputationGraph& cg, GatedGCNParams params);

    dy::Expression W_parent;
    dy::Expression b_parent;
    dy::Expression W_children;
    dy::Expression b_children;
    dy::Expression W_gate_old;
    dy::Expression W_gate_new;
    dy::Expression b_gate;
};

struct GatedGCNBuilder
{
    dy::ParameterCollection local_pc;

    GatedGCNBuilder(dy::ParameterCollection& pc,
                    unsigned n_layers,
                    unsigned n_iter,
                    unsigned dim);

    void new_graph(dy::ComputationGraph& cg, bool training);
    dy::Expression apply(const dy::Expression& input,
                         const dy::Expression& graph);

    void set_dropout(float value);

    unsigned n_iter;
    unsigned n_layers;
    std::vector<GatedGCNParams> params;
    std::vector<GatedGCNExprs> exprs;

    bool _training = true;
    float dropout_rate = 0.f;
};
