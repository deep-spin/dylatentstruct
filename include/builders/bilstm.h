
#pragma once
/* Author: Caio Corro
 * License: MIT
 * Part of https://github.com/FilippoC/dynet-tools/
 */

#include <dynet/model.h>
#include <dynet/lstm.h>

namespace dy = dynet;

struct BiLSTMSettings
{
    unsigned stacks;
    unsigned layers;
    unsigned dim;

    unsigned int output_rows(const unsigned input_dim) const;
};


struct BiLSTMBuilder
{
    const BiLSTMSettings settings;
    dy::ParameterCollection local_pc;
    const unsigned input_dim;

    std::vector<std::pair<dy::VanillaLSTMBuilder, dy::VanillaLSTMBuilder>> builders;

    BiLSTMBuilder(dy::ParameterCollection& pc, const BiLSTMSettings& settings, unsigned input_dim);

    void new_graph(dy::ComputationGraph& cg, bool training, bool update);
    std::vector<dy::Expression> operator()(const std::vector<dy::Expression>& embeddings);

    void set_dropout(float value);
    void disable_dropout();

    unsigned output_rows() const;
};
