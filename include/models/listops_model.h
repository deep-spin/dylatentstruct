#pragma once

#include <dynet/lstm.h>
#include <dynet/rnn.h>
#include <dynet/tensor-eigen.h>
#include <dynet/tensor.h>

#include <string>
#include <tuple>

#include "args.h"
#include "basesent.h"
#include "builders/adjmatrix.h"

namespace dy = dynet;

struct ListOps : public BaseSentClf
{
    explicit ListOps(dy::ParameterCollection& params,
                     ListOpOpts::Tree tree_type,
                     size_t self_iter,
                     unsigned vocab_size,
                     unsigned embed_dim,
                     unsigned hidden_dim,
                     unsigned n_classes,
                     unsigned stacks = 1,
                     float dropout = .5)
        : BaseSentClf(params,
                      vocab_size,
                      embed_dim,
                      hidden_dim,
                      n_classes,
                      stacks,
                      dropout)
        , self_iter_{ self_iter }
        , dropout_{ dropout }
        , p_out_W{ p.add_parameters({ n_classes, 2 * hidden_dim } )}
        , p_out_b{ p.add_parameters({ n_classes }) }
        , p_attn_W{ p.add_parameters({ hidden_dim, hidden_dim } )}
        , p_attn_b{ p.add_parameters({ hidden_dim }) }
        , p_comb_W{ p.add_parameters({ hidden_dim, 2 * hidden_dim } )}
        , p_comb_b{ p.add_parameters({ hidden_dim }) }
        {
            if (tree_type == ListOpOpts::Tree::LTR)
                tree = std::make_unique<LtrAdjacency>();
            else if (tree_type == ListOpOpts::Tree::FLAT)
                tree = std::make_unique<FlatAdjacency>();
            else if (tree_type == ListOpOpts::Tree::GOLD)
                tree = std::make_unique<CustomAdjacency>();
            else {
                std::cerr << "Not implemented";
                std::abort();
            }

        }

    virtual vector<dy::Expression> predict_batch(dy::ComputationGraph& cg,
                                             const SentBatch& batch) override
    {
        bilstm.new_graph(cg, training_, true);
        tree->new_graph(cg);

        const auto d_COLS = 1u;

        auto out_b = dy::parameter(cg, p_out_b);
        auto out_W = dy::parameter(cg, p_out_W);
        auto attn_b = dy::parameter(cg, p_attn_b);
        auto attn_W = dy::parameter(cg, p_attn_W);
        auto comb_b = dy::parameter(cg, p_comb_b);
        auto comb_W = dy::parameter(cg, p_comb_W);

        vector<dy::Expression> out;

        for (auto&& sample : batch) {
            auto ctx = embed_ctx_sent(cg, sample.sentence);
            /* root is a vector of all zeros. We have biases.
            ctx.insert(ctx.begin(), dy::zeros(cg, {hidden_dim_}));
            */
            auto X = dy::concatenate_cols(ctx);

            // cols of G sum to 1
            auto G = tree->make_adj(ctx, sample.sentence.heads);

            /* do self attention with G */
            for (size_t i = 0; i < self_iter_; ++i) {
                auto attn = dy::affine_transform({ attn_b, attn_W, X });
                attn = attn * G;
                attn = dy::concatenate({ X, attn }); // rows
                X = dy::affine_transform({ comb_b, comb_W, attn });
                X = dy::rectify(X);
            }

            auto hid = dy::concatenate({
                    dy::mean_dim(X, { d_COLS }),
                    dy::max_dim(X, d_COLS) });

            hid = dy::affine_transform({ out_b, out_W, hid });
            out.push_back(hid);
        }
        return out;
    }

    size_t self_iter_;
    float dropout_;
    dy::Parameter p_out_W, p_out_b;
    dy::Parameter p_attn_W, p_attn_b;
    dy::Parameter p_comb_W, p_comb_b;
    std::unique_ptr<TreeAdjacency> tree;
};
