#pragma once

#include <dynet/lstm.h>
#include <dynet/rnn.h>
#include <dynet/tensor-eigen.h>
#include <dynet/tensor.h>
#include <dynet/index-tensor.h>

#include <string>
#include <tuple>

#include "args.h"
#include "basemodel.h"
#include "data.h"
#include "builders/adjmatrix.h"

namespace dy = dynet;

/* ListOps dependency model:
 * For every token we learn both a vector AND a matrix */
struct ListOps : public BaseModel
{
    explicit ListOps(dy::ParameterCollection& params,
                     unsigned iter,
                     unsigned vocab_size,
                     unsigned embed_dim,
                     unsigned n_classes)
        : BaseModel{ params.add_subcollection("listops") }
        , iter(iter)
        , p_emb_v{ p.add_lookup_parameters(vocab_size, {embed_dim}) }
        , p_emb_M{ p.add_lookup_parameters(vocab_size, {embed_dim, embed_dim}) }
        , p_out_W{ p.add_parameters({ n_classes, embed_dim }) }
        , p_out_b{ p.add_parameters({ n_classes }) }
    {
        tree = std::make_unique<CustomAdjacency>();
    }

    int
    n_correct(
        dy::ComputationGraph& cg,
        const SentBatch& batch)
    {
        //set_test_time();
        auto out_v = predict_batch(cg, batch);
        auto out_b = dy::concatenate_to_batch(out_v);

        cg.incremental_forward(out_b);
        auto out = out_b.value();
        auto pred = dy::as_vector(dy::TensorTools::argmax(out));

        int n_correct = 0;
        for (size_t i = 0; i < batch.size(); ++i)
            if (batch[i].target == pred[i])
                n_correct += 1;

        return n_correct;
    }

    dy::Expression
    batch_loss(
        dy::ComputationGraph& cg,
        const SentBatch& batch)
    {
        //set_train_time();
        auto out = predict_batch(cg, batch);

        vector<dy::Expression> losses;
        for (unsigned i = 0; i < batch.size(); ++i) {
            auto loss = dy::pickneglogsoftmax(out[i], batch[i].target);
            losses.push_back(loss);
        }

        return dy::sum(losses);
    }

    vector<dy::Expression>
    predict_batch(
        dy::ComputationGraph& cg,
        const SentBatch& batch)
    {
        tree->new_graph(cg);

        auto out_b = dy::parameter(cg, p_out_b);
        auto out_W = dy::parameter(cg, p_out_W);

        for (auto&& sent : batch) {
            // i was in the middle of this when I remembered
            // I don't have the listops data here.
        }
    }

    unsigned iter;
    dy::LookupParameter p_emb_v, p_emb_M;
    dy::Parameter p_out_W, p_out_b;
    std::unique_ptr<TreeAdjacency> tree;

};

/*
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
            ctx.insert(ctx.begin(), dy::zeros(cg, {hidden_dim_}));
            auto X = dy::concatenate_cols(ctx);

            // cols of G sum to 1
            auto G = tree->make_adj(ctx, sample.sentence.heads);

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
*/

