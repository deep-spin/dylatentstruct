#pragma once

#include <vector>

#include "data.h"

#include "builders/adjmatrix.h"
//#include "builders/gatedgcn.h"
#include "builders/gcn.h"
#include "models/basemodel.h"

#include <dynet/tensor.h>
#include <dynet/index-tensor.h>

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

struct GCNTagger : public BaseEmbedModel
{
    explicit
    GCNTagger(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned n_classes,
        float dropout,
        const GCNOpts& gcn_opts,
        const dy::SparseMAPOpts& smap_opts)
        : BaseEmbedModel{
            params,
            vocab_size,
            embed_dim,
            /*update_embed=*/true,
            "sentclf"}
        , p_out_W{p.add_parameters({n_classes, 2 * hidden_dim}) }
        , p_out_b{p.add_parameters({n_classes}) }
        //, gcn{ p, gcn_opts.layers, gcn_opts.iter, hidden_dim }
        , gcn{ p, gcn_opts.layers, hidden_dim, hidden_dim, true }
        , gcn_opts_{ gcn_opts }
        , hidden_dim_{ hidden_dim }
        , n_classes_{ n_classes }
        , dropout_{ dropout }
    {
        gcn.set_dropout(gcn_opts_.dropout);

        // dynamically assign TreeAdjacency object
        auto tree_type = gcn_opts.get_tree();
        if (tree_type == GCNOpts::Tree::LTR)
            tree = std::make_unique<LtrAdjacency>();
        else if (tree_type == GCNOpts::Tree::FLAT)
            tree = std::make_unique<FlatAdjacency>();
        else if (tree_type == GCNOpts::Tree::GOLD)
            tree = std::make_unique<CustomAdjacency>();
        else if (tree_type == GCNOpts::Tree::MST)
            tree = std::make_unique<MSTAdjacency>(
              p, smap_opts, hidden_dim, false, 0);
        else if (tree_type == GCNOpts::Tree::MST_LSTM)
            tree = std::make_unique<MSTLSTMAdjacency>(
              p, smap_opts, hidden_dim, dropout_, 0);
        else {
            std::cerr << "Not implemented";
            std::abort();
        }
    }

    virtual
    int
    n_correct(
        ComputationGraph& cg,
        const TaggedBatch& batch)
    {
        set_test_time();

        auto outc = dy::concatenate_cols(predict_batch(cg, batch));
        cg.incremental_forward(outc);
        auto out = outc.value();
        auto pred = dy::as_vector(dy::TensorTools::argmax(out));

        auto i = 0u;
        int n_correct = 0;

        for (auto && sent : batch) {
            for (auto && y_true : sent.tags) {
                if (y_true == pred[i]) {
                    n_correct += 1;
                }
                i += 1;
            }
        }

        return n_correct;
    }

    virtual
    Expression
    batch_loss(
        ComputationGraph& cg,
        const TaggedBatch& batch)
    {
        set_train_time();
        auto out = predict_batch(cg, batch);
        vector<Expression> losses;

        for (auto i = 0u; i < batch.size(); ++i) {
            auto sent = batch[i];
            if (sent.size() == 1) {
                auto loss = dy::pickneglogsoftmax(out[i], sent.tags[0]);
                losses.push_back(loss);
            } else {
                for (auto j = 0u; j < sent.size(); ++j) {
                    auto scores = dy::pick(out[i], j, 1);
                    auto loss = dy::pickneglogsoftmax(scores, sent.tags[j]);
                    losses.push_back(loss);
                }
            }
        }

        return dy::sum(losses);
    }

    virtual
    vector<Expression>
    predict_batch(
        ComputationGraph& cg,
        const TaggedBatch& batch)
    {
        gcn.new_graph(cg, training_);
        tree->new_graph(cg, training_);

        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<Expression> out;

        for (auto && sample : batch)
        {
            auto ctx = embed_sent(cg, sample.sentence);

            // dropout embeddings
            if (training_)
            {
                for (auto&& h : ctx)
                    h = dy::dropout(h, dropout_);
            }

            // root is a vector of all zeros. We have biases.
            ctx.insert(ctx.begin(), dy::zeros(cg, {hidden_dim_}));
            auto G = tree->make_adj(ctx, sample.sentence);
            auto X = dy::concatenate_cols(ctx);
            auto H = gcn.apply(X, G);

            // drop the root
            H = dy::pick_range(H, 1, sample.size() + 1, 1);

            // dropout h
            if (training_)
                H = dy::dropout(H, dropout_);

            H = dy::affine_transform({out_b, out_W, H});
            out.push_back(H);
        }

        return out;
    }

    virtual
    void
    set_train_time() override
    {
        BaseEmbedModel::set_train_time();
        gcn.set_dropout(gcn_opts_.dropout);
    }

    virtual
    void
    set_test_time() override
    {
        BaseEmbedModel::set_test_time();
        gcn.set_dropout( 0.0f );
    }

    dy::Parameter p_out_W;
    dy::Parameter p_out_b;

    GCNBuilder gcn;
    // GatedGCNBuilder gcn;
    std::unique_ptr<TreeAdjacency> tree;

    GCNOpts gcn_opts_;
    unsigned hidden_dim_;
    unsigned n_classes_;
    float dropout_;
};
