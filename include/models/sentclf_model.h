# pragma once

#include <string>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/tensor.h>
#include <dynet/index-tensor.h>

#include "utils.h"
#include "models/basemodel.h"
#include "builders/gcn.h"
#include "builders/adjmatrix.h"

#include <vector>

namespace dy = dynet;

using dy::Expression;
using dy::Parameter;
using dy::LookupParameter;
using dy::ComputationGraph;
using dy::ParameterCollection;
using dy::parameter;
using dy::RNNBuilder;

using std::vector;
using std::unique_ptr;


struct GCNSentClf : public BaseEmbedBiLSTMModel
{
    Parameter p_out_W;
    Parameter p_out_b;

    GCNSettings gcn_settings;
    GCNBuilder gcn;

    unsigned hidden_dim_;
    unsigned n_classes_;
    float dropout_;
    std::string strategy_;

    std::unique_ptr<TreeAdjacency> tree;

    explicit
    GCNSentClf(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned stacks,
        unsigned gcn_layers,
        float dropout,
        const std::string& strategy,
        unsigned n_classes)
        : BaseEmbedBiLSTMModel(
            params,
            vocab_size,
            embed_dim,
            hidden_dim,
            stacks,
            /*update_embed=*/true,
            dropout,
            "sentclf")
        , p_out_W(p.add_parameters({n_classes, 2 * hidden_dim}))
        , p_out_b(p.add_parameters({n_classes}))
        , gcn_settings{hidden_dim, gcn_layers, false}
        , gcn(p, gcn_settings, hidden_dim)
        , hidden_dim_(hidden_dim)
        , n_classes_(n_classes)
        , dropout_(dropout)
        , strategy_(strategy)
    {
        gcn.set_dropout(dropout_);

        // dynamically assign TreeAdjacency object
        if (strategy_ == "corenlp")
            tree.reset(new CustomAdjacency());
        else if (strategy_ == "flat")
            tree.reset(new FlatAdjacency());
        else if (strategy_ == "ltr")
            tree.reset(new LtrAdjacency());
        else {
            std::cerr << "Invalid strategy." << std::endl;
            std::abort();
        }
    }

    virtual
    int
    n_correct(
        ComputationGraph& cg,
        const SentBatch& batch)
    {
        set_test_time();
        auto out_v = predict_batch(cg, batch);
        auto out_b = dy::concatenate_to_batch(out_v);

        cg.incremental_forward(out_b);
        auto out = out_b.value();
        auto pred = dy::as_vector(dy::TensorTools::argmax(out));

        int n_correct = 0;
        for (unsigned i = 0; i < batch.size(); ++i)
            if (batch[i].target == pred[i])
                n_correct += 1;

        return n_correct;
    }

    virtual
    Expression
    batch_loss(
        ComputationGraph& cg,
        const SentBatch& batch)
    {
        set_train_time();
        auto out = predict_batch(cg, batch);

        vector<Expression> losses;
        for (unsigned i = 0; i < batch.size(); ++i)
        {
            auto loss = dy::pickneglogsoftmax(out[i], batch[i].target);
            losses.push_back(loss);
        }

        return dy::sum(losses);
    }

    virtual
    vector<Expression>
    predict_batch(
        ComputationGraph& cg,
        const SentBatch& batch)
    {

        bilstm.new_graph(cg, training_, true);
        gcn.new_graph(cg, training_, true);
        tree->new_graph(cg);
        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<Expression> out;

        for (auto && sample : batch)
        {
            auto ctx = embed_ctx_sent(cg, sample.sentence);

            // root is a vector of all zeros. We have biases.
            ctx.insert(ctx.begin(), dy::zeros(cg, {hidden_dim_}));

            //auto X = dy::concatenate_cols(ctx);
            //auto h = dy::affine_transform({bk, Wk, X});
            //h = dy::sum_dim(h, {1});
            //h = h + b_self;
            //h = dy::tanh(h);

            auto G = tree->make_adj(ctx, sample.sentence.heads);

            auto X = dy::concatenate_cols(ctx);
            auto res = gcn.apply(X, G);

            auto h =  dy::concatenate({
                dy::mean_dim(res, {1}),
                dy::max_dim(res, 1)
            });

            //auto h = dy::pick(res, (unsigned) 0, 1);
            //h = dy::pick_range(h, 0, hidden_dim_);

            h = dy::affine_transform({out_b, out_W, h});
            out.push_back(h);
        }

        return out;
    }
};
