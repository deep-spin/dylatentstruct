#include "builders/gatedgcn.h"
#include <dynet/param-init.h>

namespace dy = dynet;

GatedGCNParams::GatedGCNParams(dy::ParameterCollection& pc, unsigned dim)
  : W_parent{ pc.add_parameters({ dim, dim }, 0, "W-parent") }
  , b_parent{ pc.add_parameters({ dim }, 0, "b-parent") }
  , W_children{ pc.add_parameters({ dim, dim }, 0, "W-children") }
  , b_children{ pc.add_parameters({ dim }, 0, "b-children") }
  , W_gate_old{ pc.add_parameters({ dim, dim }, 0, "W-gate-old") }
  , W_gate_new{ pc.add_parameters({ dim, dim }, 0, "W-gate-new") }
  , b_gate{ pc.add_parameters({ dim }, 0, "b-gate") }
{}

GatedGCNExprs::GatedGCNExprs(dy::ComputationGraph& cg, GatedGCNParams params)
{
    W_parent = dy::parameter(cg, params.W_parent);
    b_parent = dy::parameter(cg, params.b_parent);
    W_children = dy::parameter(cg, params.W_children);
    b_children = dy::parameter(cg, params.b_children);
    W_gate_old = dy::parameter(cg, params.W_gate_old);
    W_gate_new = dy::parameter(cg, params.W_gate_new);
    b_gate = dy::parameter(cg, params.b_gate);
}

GatedGCNBuilder::GatedGCNBuilder(dy::ParameterCollection& pc,
                                 unsigned n_layers,
                                 unsigned n_iter,
                                 unsigned dim)
  : local_pc{ pc.add_subcollection("gcn") }
  , n_iter(n_iter)
  , n_layers(n_layers)
  , exprs(n_layers)
{
    params.reserve(n_layers);
    for (unsigned i = 0; i < n_layers; ++i)
        params.push_back(GatedGCNParams(local_pc, dim));

    std::cerr << "Gated Graph Convolutional Network \n"
              << "n_layers=" << n_layers << "n_iter=" << n_iter << std::endl;
}

void
GatedGCNBuilder::new_graph(dy::ComputationGraph& cg, bool training)
{
    _training = training;
    for (unsigned i = 0; i < n_layers; ++i)
        exprs.at(i) = GatedGCNExprs(cg, params.at(i));
}

dy::Expression
GatedGCNBuilder::apply(const dy::Expression& input, const dy::Expression& graph)
{
    using dy::affine_transform;

    auto t_graph = dy::transpose(graph);
    auto h_old = input;

    for (size_t k = 0; k < n_layers; ++k) {
        auto ex = exprs.at(k);

        for (size_t i = 0; i < n_iter; ++i) {
            auto parents =
              affine_transform({ ex.b_parent, ex.W_parent, h_old });

            auto children =
              affine_transform({ ex.b_children, ex.W_children, h_old });

            auto h_new = parents * graph + children * t_graph;

            if (_training)
                h_new = dy::dropout(h_new, dropout_rate);

            auto gate = dy::logistic(affine_transform(
              { ex.b_gate, ex.W_gate_old, h_old, ex.W_gate_new, h_new }));

            h_old = dy::cmult(gate, h_old) + dy::cmult(1 - gate, h_new);
        }
    }

    return h_old;
}

void
GatedGCNBuilder::set_dropout(float value)
{
    dropout_rate = value;
}
