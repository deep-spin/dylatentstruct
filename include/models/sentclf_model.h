#pragma once

#include <string>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/tensor.h>
#include <dynet/index-tensor.h>
#include <dynet/param-init.h>

#include "data.h"
#include "models/basemodel.h"
#include "builders/gcn.h"
//#include "builders/gatedgcn.h"
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


struct GCNSentClf : public BaseEmbedModel
{

    explicit
    GCNSentClf(
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
        , pooled_dim{ (gcn_opts.layers + 1) * hidden_dim }
        , p_out_W{ p.add_parameters({ n_classes, pooled_dim } ) }
        , p_out_b{ p.add_parameters({ n_classes }) }
        , layer_norm_g_p{ p.add_parameters({ pooled_dim },
                                           dy::ParameterInitConst(1.0f)) }
        , layer_norm_b_p{ p.add_parameters({ pooled_dim },
                                           dy::ParameterInitConst(0.0f)) }
        //, gcn{ p, gcn_opts.layers, gcn_opts.iter, hidden_dim }
        , gcn{ p, gcn_opts.layers, hidden_dim, hidden_dim, true }
        //, gcn_opts_{ gcn_opts }
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
              p, smap_opts, hidden_dim, false, gcn_opts_.budget);
        else if (tree_type == GCNOpts::Tree::MST_LSTM)
            tree = std::make_unique<MSTLSTMAdjacency>(
              p, smap_opts, hidden_dim, dropout_, gcn_opts_.budget);
        else {
            std::cerr << "Not implemented";
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
            auto res = gcn.apply(X, G);

            auto h = dy::sum_dim(res, {1});

            // apply simple layer norm
            auto layer_norm_g = parameter(cg, layer_norm_g_p);
            auto layer_norm_b = parameter(cg, layer_norm_b_p);
            h = dy::layer_norm(h, layer_norm_g, layer_norm_b);

            //if (training_)
                //h = dy::dropout(h, dropout_);

            h = dy::affine_transform({out_b, out_W, h});
            out.push_back(h);
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


    unsigned pooled_dim;

    dy::Parameter p_out_W;
    dy::Parameter p_out_b;
    dy::Parameter layer_norm_g_p;
    dy::Parameter layer_norm_b_p;

    GCNBuilder gcn;
    //GatedGCNBuilder gcn;
    std::unique_ptr<TreeAdjacency> tree;

    GCNOpts gcn_opts_;
    unsigned hidden_dim_;
    unsigned n_classes_;
    float dropout_;
};
