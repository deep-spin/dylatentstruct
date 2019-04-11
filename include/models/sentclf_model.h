# pragma once

#include <string>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/tensor.h>
#include <dynet/index-tensor.h>

#include "utils.h"
#include "models/basemodel.h"
#include "builders/gcn.h"

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
        , gcn_settings{hidden_dim, gcn_layers, false}
        , gcn(p, gcn_settings, hidden_dim)
        , hidden_dim_(hidden_dim)
        , n_classes_(n_classes)
        , dropout_(dropout)
        , strategy_(strategy)
    {
        p_out_W = p.add_parameters({n_classes_, hidden_dim});
        p_out_b = p.add_parameters({n_classes_});

        gcn.set_dropout(dropout_);
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
        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<Expression> out;

        for (auto && sample : batch)
        {
            unsigned n = sample.sentence.size();
            auto ctx = embed_ctx_sent(cg, sample.sentence);
            ctx.insert(ctx.begin(), dy::zeros(cg, {hidden_dim_}));

            vector<unsigned> heads;

            if (strategy_ == "corenlp") {
                for (auto& h : sample.sentence.heads)
                    if (h >= 0)
                        heads.push_back(h);
            } else if (strategy_ == "flat") {
                heads.assign(n, 0);
            } else if (strategy_ == "ltr") {
                for (size_t k = 1; k < n; ++k)
                    heads.push_back(k + 1);
                heads.push_back(0);
            } else {
                std::abort();
            }

            auto G = dy::one_hot(cg, 1 + heads.size(),  heads);

            G = dy::reshape(G, {1 + n, n});
            G = dy::concatenate({dy::zeros(cg, {1 + n, 1}), G}, 1);

            auto X = dy::concatenate_cols(ctx);

            auto res = gcn.apply(X, G);

            auto h = dy::pick(res, (unsigned) 0, /*dim=*/1);
            h = dy::affine_transform({out_b, out_W, h});
            out.push_back(h);
        }

        return out;
    }
};
