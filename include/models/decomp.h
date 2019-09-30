#pragma once

#include <vector>
#include <dynet/index-tensor.h>
#include <dynet/tensor.h>

#include "args.h"
#include "builders/biattn.h"
#include "models/basemodel.h"

/*
 * Decomposable attention NLI classifiers
 */

namespace dy = dynet;

struct DecompAttnParams
{
    DecompAttnParams(dy::ParameterCollection& pc,
                     unsigned dim,
                     unsigned embed_dim,
                     unsigned n_classes)
      : W_enc1{ pc.add_parameters({ dim, embed_dim }, 0, "W-enc-1") }
      , b_enc1{ pc.add_parameters({ dim }, 0, "b-enc-1") }
      , W_enc2{ pc.add_parameters({ dim, dim }, 0, "W-enc-2") }
      , b_enc2{ pc.add_parameters({ dim }, 0, "b-enc-2") }
      , W_comp1a{ pc.add_parameters({ dim, embed_dim }, 0, "W-comp-1a") }
      , W_comp1b{ pc.add_parameters({ dim, embed_dim }, 0, "W-comp-1b") }
      , b_comp1{ pc.add_parameters({ dim }, 0, "b-comp-1") }
      , W_comp2{ pc.add_parameters({ dim, dim }, 0, "W-comp-2") }
      , b_comp2{ pc.add_parameters({ dim }, 0, "b-comp-2") }
      , W_agg1a{ pc.add_parameters({ dim, dim }, 0, "W-agg-1a") }
      , W_agg1b{ pc.add_parameters({ dim, dim }, 0, "W-agg-1b") }
      , b_agg1{ pc.add_parameters({ dim }, 0, "b-agg-1") }
      , W_out{ pc.add_parameters({ n_classes, dim }, 0, "W-out") }
      , b_out{ pc.add_parameters({ n_classes }, 0, "b-out") }
    {}

    dy::Parameter W_enc1;
    dy::Parameter b_enc1;
    dy::Parameter W_enc2;
    dy::Parameter b_enc2;
    dy::Parameter W_comp1a;
    dy::Parameter W_comp1b;
    dy::Parameter b_comp1;
    dy::Parameter W_comp2;
    dy::Parameter b_comp2;
    dy::Parameter W_agg1a;
    dy::Parameter W_agg1b;
    dy::Parameter b_agg1;
    dy::Parameter W_out;
    dy::Parameter b_out;
};

struct DecompAttnExprs
{
    DecompAttnExprs() = default;
    DecompAttnExprs(dy::ComputationGraph& cg, DecompAttnParams params)
    {
        W_enc1 = dy::parameter(cg, params.W_enc1);
        b_enc1 = dy::parameter(cg, params.b_enc1);
        W_enc2 = dy::parameter(cg, params.W_enc2);
        b_enc2 = dy::parameter(cg, params.b_enc2);
        W_comp1a = dy::parameter(cg, params.W_comp1a);
        W_comp1b = dy::parameter(cg, params.W_comp1b);
        b_comp1 = dy::parameter(cg, params.b_comp1);
        W_comp2 = dy::parameter(cg, params.W_comp2);
        b_comp2 = dy::parameter(cg, params.b_comp2);
        W_agg1a = dy::parameter(cg, params.W_agg1a);
        W_agg1b = dy::parameter(cg, params.W_agg1b);
        b_agg1 = dy::parameter(cg, params.b_agg1);
        W_out = dy::parameter(cg, params.W_out);
        b_out = dy::parameter(cg, params.b_out);
    }

    dy::Expression W_enc1;
    dy::Expression b_enc1;
    dy::Expression W_enc2;
    dy::Expression b_enc2;
    dy::Expression W_comp1a;
    dy::Expression W_comp1b;
    dy::Expression b_comp1;
    dy::Expression W_comp2;
    dy::Expression b_comp2;
    dy::Expression W_agg1a;
    dy::Expression W_agg1b;
    dy::Expression b_agg1;
    dy::Expression W_out;
    dy::Expression b_out;
};

struct DecompAttn : public BaseEmbedModel
{

    explicit DecompAttn(dy::ParameterCollection& pc,
                        unsigned vocab_size,
                        unsigned embed_dim,
                        unsigned hidden_dim,
                        unsigned n_classes,
                        AttnOpts::Attn attn_type,
                        const dy::SparseMAPOpts& smap_opts,
                        float dropout_p,
                        bool update_embed)
      : BaseEmbedModel(pc, vocab_size, embed_dim, update_embed)
      , decomp_params(p, hidden_dim, embed_dim, n_classes)
      , dropout_p{ dropout_p }
      , hidden_dim{ hidden_dim }
    {
        if (attn_type == AttnOpts::Attn::SOFTMAX)
            attn = std::make_unique<BiSoftmaxBuilder>();
        else if (attn_type == AttnOpts::Attn::SPARSEMAX)
            attn = std::make_unique<BiSparsemaxBuilder>();
        else if (attn_type == AttnOpts::Attn::MATCH)
            attn = std::make_unique<MatchingBuilder>(smap_opts);
        else if (attn_type == AttnOpts::Attn::XOR_MATCH)
            attn = std::make_unique<XORMatchingBuilder>(smap_opts);
        else if (attn_type == AttnOpts::Attn::NEIGHBOR_MATCH)
            attn = std::make_unique<NeighborMatchingBuilder>(p, smap_opts);
        else {
            std::cerr << "Unimplemented attention mechanism." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void new_graph(dy::ComputationGraph& cg)
    {
        attn->new_graph(cg, training_);
        ex = DecompAttnExprs(cg, decomp_params);
    }

    /*
    virtual void set_train_time() override
    {
        BaseEmbedBiLSTMModel::set_train_time();
    }

    virtual void set_test_time() override
    {
        BaseEmbedBiLSTMModel::set_test_time();
    }
    */

    // 3.1, Equation 1, this is the function F.
    dy::Expression attend_input(dy::Expression X)
    {
        auto d = dy::Dim{ hidden_dim, X.dim()[1] };

        X = dy::affine_transform({ ex.b_enc1, ex.W_enc1, X });
        X = dy::reshape(X, d);
        if (training_)
            X = dy::dropout(X, dropout_p);
        X = dy::rectify(X);

        X = dy::affine_transform({ ex.b_enc2, ex.W_enc2, X });
        X = dy::reshape(X, d);
        if (training_)
            X = dy::dropout(X, dropout_p);
        X = dy::rectify(X);

        return X;
    }

    // 3.2, Equation 3
    dy::Expression compare(dy::Expression X, dy::Expression X_ctx)
    {
        using dy::affine_transform;
        using dy::rectify;

        auto d = dy::Dim{ hidden_dim, X.dim()[1] };

        dy::Expression Y;
        Y =
          affine_transform({ ex.b_comp1, ex.W_comp1a, X, ex.W_comp1b, X_ctx });
        Y = dy::reshape(Y, d);
        if (training_)
            Y = dy::dropout(Y, dropout_p);
        Y = rectify(Y);

         Y = affine_transform({ ex.b_comp2, ex.W_comp2, Y });
         Y = dy::reshape(Y, d);
         if (training_)
             Y = dy::dropout(Y, dropout_p);
         Y = rectify(Y);

        return Y;
    }

    // 3.3. Aggregate (eqn 4-5)
    dy::Expression aggregate(dy::Expression VP, dy::Expression VH)
    {
        auto vp_sum = dy::sum_dim(VP, { 1u });
        auto vh_sum = dy::sum_dim(VH, { 1u });

        auto Y = dy::affine_transform(
          { ex.b_agg1, ex.W_agg1a, vp_sum, ex.W_agg1b, vh_sum });

        if (training_)
            Y = dy::dropout(Y, dropout_p);

        Y = dy::rectify(Y);
        Y = dy::affine_transform({ ex.b_out, ex.W_out, Y });
        return Y;
    }

    std::vector<dy::Expression> predict_batch(dy::ComputationGraph& cg,
                                              const NLIBatch& batch)
    {
        new_graph(cg);
        std::vector<dy::Expression> out;

        for (auto& sample : batch) {
            auto enc_prem = embed_sent(cg, sample.prem),
                 enc_hypo = embed_sent(cg, sample.hypo);

            auto P = dy::concatenate_cols(enc_prem);
            auto H = dy::concatenate_cols(enc_hypo);

            auto WP = attend_input(P);
            auto WH = attend_input(H);

            auto scores = dy::transpose(WP) * WH;

            dy::Expression U_prem, U_hypo;
            std::tie(U_prem, U_hypo) = attn->apply(scores, sample);

            auto P_ctx = H * U_hypo;
            auto H_ctx = P * U_prem;

            // compare: eqn 3
            auto VP = compare(P, P_ctx);
            auto VH = compare(H, H_ctx);

            auto Y_hat = aggregate(VP, VH);
            out.push_back(Y_hat);
        }

        return out;
    }

    dy::Expression batch_loss(dy::ComputationGraph& cg,
                              const NLIBatch& batch)
    {
        set_train_time();
        auto out = predict_batch(cg, batch);

        std::vector<dy::Expression> losses;
        for (size_t i = 0; i < batch.size(); ++i) {
            auto loss = dy::pickneglogsoftmax(out[i], batch[i].target);
            losses.push_back(loss);
        }

        return dy::average(losses);
    }

    int n_correct(dy::ComputationGraph& cg, const NLIBatch& batch)
    {
        set_test_time();
        auto out_v = predict_batch(cg, batch);
        auto out_b = dy::concatenate_to_batch(out_v);

        cg.incremental_forward(out_b);
        auto out = out_b.value();
        auto pred = dy::as_vector(dy::TensorTools::argmax(out));

        int n_correct = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            if (batch[i].target == pred[i])
                n_correct += 1;
        }
        return n_correct;
    }

    DecompAttnParams decomp_params;
    DecompAttnExprs ex;
    AttnOpts::Attn attn_type;
    std::unique_ptr<BiAttentionBuilder> attn;
    float dropout_p;
    unsigned hidden_dim;
};

