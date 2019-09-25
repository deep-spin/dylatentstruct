#pragma once

#include <vector>

#include "args.h"
#include "builders/adjmatrix.h"
#include "builders/biattn.h"
#include "builders/gatedgcn.h"
#include "models/basenli.h"

/*
 * Decomposable attention NLI classifiers
 */

namespace dy = dynet;

struct Decomp : public BaseNLI
{

    explicit Decomp(dy::ParameterCollection& params,
                    unsigned vocab_size,
                    unsigned embed_dim,
                    unsigned hidden_dim,
                    unsigned n_classes,
                    AttnOpts::Attn attn_type,
                    const dy::SparseMAPOpts& smap_opts,
                    float dropout_p,
                    bool update_embed)
      : BaseNLI(params,
                vocab_size,
                embed_dim,
                hidden_dim,
                update_embed)
      , p_out_b1{ p.add_parameters({ hidden_dim }) }
      , p_out_W1a{ p.add_parameters({ hidden_dim, hidden_dim }) }
      , p_out_W1b{ p.add_parameters({ hidden_dim, hidden_dim }) }
      , p_out_b2{ p.add_parameters({ n_classes, hidden_dim }) }
      , p_out_W2{ p.add_parameters({ n_classes, hidden_dim }) }
      , p_enc_b1{ p.add_parameters({ hidden_dim }) }
      , p_enc_W1{ p.add_parameters({ hidden_dim, embed_dim }) }
      , p_enc_b2{ p.add_parameters({ hidden_dim }) }
      , p_enc_W2{ p.add_parameters({ hidden_dim, hidden_dim }) }
      , p_comp_b1{ p.add_parameters({ hidden_dim }) }
      , p_comp_W1a{ p.add_parameters({ hidden_dim, embed_dim }) }
      , p_comp_W1b{ p.add_parameters({ hidden_dim, embed_dim }) }
      , p_comp_b2{ p.add_parameters({ hidden_dim }) }
      , p_comp_W2{ p.add_parameters({ hidden_dim, hidden_dim }) }
      //, p_sink{ p.add_parameters({ embed_dim }) }
      , dropout_p{ dropout_p }
    {

        if (attn_type == AttnOpts::Attn::SOFTMAX)
            attn = std::make_unique<BiSoftmaxBuilder>();
        else if (attn_type == AttnOpts::Attn::SPARSEMAX)
            attn = std::make_unique<BiSparsemaxBuilder>();
        else if (attn_type == AttnOpts::Attn::MATCH)
            attn = std::make_unique<MatchingBuilder>(smap_opts);
        else if (attn_type == AttnOpts::Attn::HEAD)
            attn = std::make_unique<HeadPreservingBuilder>(p, smap_opts);
        else if (attn_type == AttnOpts::Attn::HEADMATCH)
            attn =
              std::make_unique<HeadPreservingMatchingBuilder>(p, smap_opts);
        else if (attn_type == AttnOpts::Attn::HEADHO)
            attn = std::make_unique<HeadHOBuilder>(p, smap_opts);
        else if (attn_type == AttnOpts::Attn::HEADMATCHHO)
            attn = std::make_unique<HeadHOMatchingBuilder>(p, smap_opts);
        else {
            std::cerr << "Unimplemented attention mechanism." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void new_graph(dy::ComputationGraph& cg)
    {
        attn->new_graph(cg, training_);

        e_out_b1 = dy::parameter(cg, p_out_b1);
        e_out_W1a = dy::parameter(cg, p_out_W1a);
        e_out_W1b = dy::parameter(cg, p_out_W1b);

        e_out_b2 = dy::parameter(cg, p_out_b2);
        e_out_W2 = dy::parameter(cg, p_out_W2);

        e_enc_b1 = dy::parameter(cg, p_enc_b1);
        e_enc_W1 = dy::parameter(cg, p_enc_W1);

        e_enc_b2 = dy::parameter(cg, p_enc_b2);
        e_enc_W2 = dy::parameter(cg, p_enc_W2);

        e_comp_b1 = dy::parameter(cg, p_comp_b1);
        e_comp_W1a = dy::parameter(cg, p_comp_W1a);
        e_comp_W1b = dy::parameter(cg, p_comp_W1b);

        e_comp_W2 = dy::parameter(cg, p_comp_W2);
        e_comp_b2 = dy::parameter(cg, p_comp_b2);

        //e_sink = dy::parameter(cg, p_sink);
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
        auto d = dy::Dim{ hidden_dim_, X.dim()[1] };

        X = dy::affine_transform({ e_enc_b1, e_enc_W1, X });
        X = dy::reshape(X, d);

        if (training_)
            X = dy::dropout(X, dropout_p);
        X = dy::rectify(X);

        X = dy::affine_transform({ e_enc_b2, e_enc_W2, X });
        X = dy::reshape(X, d);

        return X;
    }

    // 3.2, Equation 3
    dy::Expression compare(dy::Expression X, dy::Expression X_ctx)
    {
        using dy::affine_transform;
        using dy::rectify;

        //auto d = X.dim();
        auto d = dy::Dim{ hidden_dim_, X.dim()[1] };

        dy::Expression Y;
        Y = affine_transform({ e_comp_b1, e_comp_W1a, X, e_comp_W1b, X_ctx });
        Y = dy::reshape(Y, d);
        if (training_)
            Y = dy::dropout(Y, dropout_p);
        Y = rectify(Y);

        Y = affine_transform({ e_comp_b2, e_comp_W2, Y });
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
          { e_out_b1, e_out_W1a, vp_sum, e_out_W1b, vh_sum });
        if (training_)
            Y = dy::dropout(Y, dropout_p);
        Y = dy::rectify(Y);
        Y = dy::affine_transform({ e_out_b2, e_out_W2, Y });
        return Y;
    }

    virtual std::vector<dy::Expression> predict_batch(
      dy::ComputationGraph& cg,
      const NLIBatch& batch) override
    {
        new_graph(cg);
        std::vector<dy::Expression> out;

        for (auto&& sample : batch) {
            auto enc_prem = embed_sent(cg, sample.prem),
                 enc_hypo = embed_sent(cg, sample.hypo);

            //enc_prem.insert(enc_prem.begin(), e_sink);
            //enc_hypo.insert(enc_hypo.begin(), e_sink);

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

    AttnOpts::Attn attn_type;
    std::unique_ptr<BiAttentionBuilder> attn;

    dy::Parameter p_out_b1, p_out_W1a, p_out_W1b;
    dy::Parameter p_out_b2, p_out_W2;
    dy::Parameter p_enc_b1, p_enc_W1;
    dy::Parameter p_enc_b2, p_enc_W2;
    dy::Parameter p_comp_b1, p_comp_W1a, p_comp_W1b;
    dy::Parameter p_comp_b2, p_comp_W2;
    //dy::Parameter p_sink;

    dy::Expression e_out_b1, e_out_W1a, e_out_W1b;
    dy::Expression e_out_b2, e_out_W2;
    dy::Expression e_enc_b1, e_enc_W1;
    dy::Expression e_enc_b2, e_enc_W2;
    dy::Expression e_comp_b1, e_comp_W1a, e_comp_W1b;
    dy::Expression e_comp_b2, e_comp_W2;
    //dy::Expression e_sink;

    float dropout_p;
};

