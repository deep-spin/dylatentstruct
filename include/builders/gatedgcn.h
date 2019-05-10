#pragma once

#include <vector>

#include <dynet/expr.h>
#include <dynet/model.h>

namespace dy = dynet;

struct GatedGCNBuilder
{
    dy::ParameterCollection local_pc;

    dy::Parameter p_W_parent;
    dy::Parameter p_b_parent;
    dy::Parameter p_W_children;
    dy::Parameter p_b_children;
    dy::Parameter p_W_gate_old;
    dy::Parameter p_W_gate_new;
    dy::Parameter p_b_gate;

    dy::Expression e_W_parent;
    dy::Expression e_b_parent;
    dy::Expression e_W_children;
    dy::Expression e_b_children;
    dy::Expression e_W_gate_old;
    dy::Expression e_W_gate_new;
    dy::Expression e_b_gate;

    unsigned n_iter;

    bool _training = true;
    float dropout_rate = 0.f;

    GatedGCNBuilder(dy::ParameterCollection& pc,
                    unsigned n_iter,
                    unsigned dim);

    void new_graph(dy::ComputationGraph& cg, bool training);
    dy::Expression apply(const dy::Expression& input,
                         const dy::Expression& graph);

    void set_dropout(float value);
};
