#include "builders/gatedgcn.h"
#include <dynet/param-init.h>

namespace dy = dynet;

GatedGCNBuilder::GatedGCNBuilder(dy::ParameterCollection& pc,
                                 unsigned n_iter,
                                 unsigned dim)
  : local_pc(pc.add_subcollection("gcn"))
  , p_W_parent{ local_pc.add_parameters({ dim, dim }) }
  , p_b_parent{ local_pc.add_parameters({ dim }) }
  , p_W_children{ local_pc.add_parameters({ dim, dim }) }
  , p_b_children{ local_pc.add_parameters({ dim }) }
  , p_W_gate_old{ local_pc.add_parameters({ dim, dim }) }
  , p_W_gate_new{ local_pc.add_parameters({ dim, dim }) }
  , p_b_gate{ local_pc.add_parameters({ dim }) }
  , n_iter(n_iter)
{

    std::cerr << "Gated Graph Convolutional Network \n"
              << "n_iter=" << n_iter << std::endl;
}

void
GatedGCNBuilder::new_graph(dy::ComputationGraph& cg, bool training)
{
    _training = training;
    e_W_parent = dy::parameter(cg, p_W_parent);
    e_b_parent = dy::parameter(cg, p_b_parent);
    e_W_children = dy::parameter(cg, p_W_children);
    e_b_children = dy::parameter(cg, p_b_children);
    e_W_gate_old = dy::parameter(cg, p_W_gate_old);
    e_W_gate_new = dy::parameter(cg, p_W_gate_new);
    e_b_gate = dy::parameter(cg, p_b_gate);
}

dy::Expression
GatedGCNBuilder::apply(const dy::Expression& input, const dy::Expression& graph)
{
    auto t_graph = dy::transpose(graph);
    auto h_old = input;

    for(size_t i = 0; i < n_iter; ++i) {
        auto parents = dy::affine_transform({ e_b_parent, e_W_parent, h_old });
        auto children = dy::affine_transform({ e_b_children, e_W_children, h_old });
        auto h_new = parents * graph + children * t_graph;

        auto gate = dy::affine_transform({e_b_gate,
                                          e_W_gate_old, h_old,
                                          e_W_gate_new, h_new});
        gate = dy::logistic(gate);
        h_old = dy::cmult(gate, h_old) + dy::cmult(1 - gate, h_new);
    }

    return h_old;
}

void
GatedGCNBuilder::set_dropout(float value)
{
    dropout_rate = value;
}
