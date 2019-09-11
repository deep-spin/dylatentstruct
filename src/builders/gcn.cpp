#include <dynet/param-init.h>

#include "builders/gcn.h"

namespace dy = dynet;

GCNParams::GCNParams(dy::ParameterCollection& pc, unsigned dim_in, unsigned dim_out)
  : W_parents{ pc.add_parameters({ dim_out, dim_in }, 0, "W-parents") }
  , b_parents{ pc.add_parameters({ dim_out }, 0, "b-parents") }
  , W_children{ pc.add_parameters({ dim_out, dim_in }, 0, "W-children") }
  , b_children{ pc.add_parameters({ dim_out }, 0, "b-children") }
  , W_self{ pc.add_parameters({ dim_out, dim_in }, 0, "W-self") }
  , b_self{ pc.add_parameters({ dim_out }, 0, "b-self") }
{}

GCNExprs::GCNExprs(dy::ComputationGraph& cg, GCNParams params)
{
    W_parents = dy::parameter(cg, params.W_parents);
    b_parents = dy::parameter(cg, params.b_parents);
    W_children = dy::parameter(cg, params.W_children);
    b_children = dy::parameter(cg, params.b_children);
    W_self = dy::parameter(cg, params.W_self);
    b_self = dy::parameter(cg, params.b_self);
}

GCNBuilder::GCNBuilder(dy::ParameterCollection& pc,
                       unsigned n_layers,
                       unsigned dim_in,
                       unsigned dim_out,
                       bool dense)
  : local_pc(pc.add_subcollection("gcn"))
  , exprs{ n_layers }
  , n_layers{ n_layers }
  , dense{ dense }
{
    params.reserve(n_layers);
    unsigned dim = dim_in;
    for (unsigned i = 0; i < n_layers; ++i) {
        params.push_back(GCNParams(local_pc, dim, dim_out));

        if (dense)
            dim = dim_in + dim_out;
        else
            dim = dim_out;
    }

    std::cerr << "Graph Convolutional Network\n"
              << " layers: " << n_layers << "\n"
              << std::endl;
}

void
GCNBuilder::new_graph(dy::ComputationGraph& cg, bool training)
{
    _training = training;
    for (unsigned i = 0; i < n_layers; ++i)
        exprs.at(i) = GCNExprs(cg, params.at(i));
}

dy::Expression
GCNBuilder::apply(const dy::Expression& input, const dy::Expression& graph)
{
    using dy::affine_transform;

    if (n_layers == 0)
        return input;

    auto t_graph = dy::transpose(graph);
    auto h = input;
    for (auto i = 0u; i < n_layers; ++i) {
        auto ex = exprs.at(i);

        auto self = affine_transform({ ex.b_self, ex.W_self, h });
        auto parents = affine_transform({ ex.b_parents, ex.W_parents, h });
        auto children = affine_transform({ ex.b_children, ex.W_children, h });
        auto h_next = self + parents * graph + children * t_graph;

        if (_training)
            h_next = dy::dropout(h_next, dropout_rate);

        h_next = dy::rectify(h_next);

        if (dense)
            h = dy::concatenate({ h, h_next });
        else
            h = h_next;
    }

    return h;
}

void
GCNBuilder::set_dropout(float value)
{
    dropout_rate = value;
}
