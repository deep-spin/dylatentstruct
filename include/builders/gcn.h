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


struct GCNBuilder
{
    const GCNSettings settings;
    dy::ParameterCollection local_pc;

    std::vector<dy::Parameter> p_W_parents, p_b_parents, p_W_children, p_b_children, p_W_self, p_b_self;
    std::vector<dy::Expression> e_W_parents, e_b_parents, e_W_children, e_b_children, e_W_self, e_b_self;

    unsigned _output_rows;
    bool _training = true;
    float dropout_rate = 0.f;

    GCNBuilder(dy::ParameterCollection& pc, const GCNSettings& settings, unsigned dim_input);

    void new_graph(dy::ComputationGraph& cg, bool training, bool update);
    dy::Expression apply(const dy::Expression &input, const dy::Expression& graph);

    void set_dropout(float value);
    unsigned output_rows() const;
};
