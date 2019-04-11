/* Author: Caio Corro
 * License: MIT
 * Part of https://github.com/FilippoC/dynet-tools/
 */

#include "builders/bilstm.h"

namespace dy = dynet;


unsigned int BiLSTMSettings::output_rows(const unsigned input_dim) const
{
    if (stacks == 0)
        return input_dim;
    else
        return 2 * dim;
}


BiLSTMBuilder::BiLSTMBuilder(dy::ParameterCollection &pc, const BiLSTMSettings &settings, unsigned input_dim) :
        settings(settings),
        local_pc(pc.add_subcollection("bilstm")),
        input_dim(input_dim)
{
    for (unsigned stack = 0; stack < settings.stacks; ++stack)
        builders.emplace_back(
                dy::VanillaLSTMBuilder(
                        settings.layers,
                        (stack == 0 ? input_dim : 2 * settings.dim),
                        settings.dim,
                        local_pc),
                dy::VanillaLSTMBuilder(
                        settings.layers,
                        (stack == 0 ? input_dim : 2 * settings.dim),
                        settings.dim,
                        local_pc
                )
        );

    std::cerr
            << "BiLSTM\n"
            << " input dim: " << input_dim << "\n"
            << " hidden dim: " << settings.dim << "\n"
            << " layers: " << settings.layers << "\n"
            << " stacks: " << settings.stacks << "\n"
            << "\n";
}

unsigned BiLSTMBuilder::output_rows() const
{
    return settings.output_rows(input_dim);
}

void BiLSTMBuilder::new_graph(dy::ComputationGraph &cg, bool, bool update)
{
    for (unsigned stack = 0; stack < settings.stacks; ++stack)
    {
        builders.at(stack).first.new_graph(cg, update);
        builders.at(stack).second.new_graph(cg, update);
    }
}


void BiLSTMBuilder::set_dropout(float value)
{
    for (unsigned stack = 0; stack < settings.stacks; ++stack)
    {
        builders.at(stack).first.set_dropout(value);
        builders.at(stack).second.set_dropout(value);
    }
}

void BiLSTMBuilder::disable_dropout()
{
    set_dropout(0);
}

std::vector<dy::Expression> BiLSTMBuilder::operator()(const std::vector<dy::Expression>& embeddings)
{
    const unsigned size = embeddings.size();
    std::vector<dy::Expression> ret(embeddings);

    std::vector<dy::Expression> e_forward(embeddings.size());
    std::vector<dy::Expression> e_backward(embeddings.size());
    for (unsigned stack = 0; stack < settings.stacks; ++stack)
    {
        builders.at(stack).first.start_new_sequence();
        builders.at(stack).second.start_new_sequence();

        for (unsigned i = 0u ; i < size ; ++i)
            e_forward.at(i) = builders.at(stack).first.add_input(ret.at(i));
        for (int i = size - 1 ; i >= 0 ; --i)
            e_backward.at(i) = builders.at(stack).second.add_input(ret.at(i));
        for (unsigned i = 0u ; i < size ; ++i)
            ret.at(i) = dy::concatenate({e_forward.at(i), e_backward.at(i)});
    }
    return ret;
}
