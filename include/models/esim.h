#pragma once

/*
 * ESIM-style NLI classifiers.
 * Align premise to hypo and viceversa
 */

#include <dynet/lstm.h>
#include <dynet/rnn.h>
#include <dynet/tensor-eigen.h>
#include <dynet/tensor.h>

#include <string>
#include <tuple>

#include "args.h"
#include "basenli.h"
#include "builders/biattn.h"

#include <iostream>

typedef std::tuple<dy::Expression, dy::Expression> Expr2;

namespace dy = dynet;

struct ESIM : public BaseNLI
{
    float dropout_p;

    Parameter p_inf_W, p_inf_b;
    Parameter p_hid_W, p_hid_b;
    Parameter p_out_W, p_out_b;

    BiLSTMBuilder bilstm_inf;

    ESIMArgs::Attn attn_type;
    std::unique_ptr<BiAttentionBuilder> attn;

    explicit ESIM(ParameterCollection& params,
                  unsigned vocab_size,
                  unsigned embed_dim,
                  unsigned hidden_dim,
                  unsigned n_classes,
                  ESIMArgs::Attn attn_type,
                  const dy::SparseMAPOpts& smap_opts,
                  float dropout_p = .5,
                  unsigned stacks = 1,
                  bool update_embed = true)
      : BaseNLI(params,
                vocab_size,
                embed_dim,
                hidden_dim,
                stacks,
                update_embed,
                dropout_p)
      , dropout_p(dropout_p)
      , p_inf_W(p.add_parameters({ hidden_dim, 4 * hidden_dim }))
      , p_inf_b(p.add_parameters({ hidden_dim }))
      , p_hid_W(p.add_parameters({ hidden_dim, 4 * hidden_dim }))
      , p_hid_b(p.add_parameters({ hidden_dim }))
      , p_out_W(p.add_parameters({ n_classes, hidden_dim }))
      , p_out_b(p.add_parameters({ n_classes }))
      , bilstm_inf(p, bilstm_settings, hidden_dim)
      , attn_type(attn_type)
    {

        if (attn_type == ESIMArgs::Attn::SOFTMAX)
            attn = std::make_unique<BiSoftmaxBuilder>();
        else if (attn_type == ESIMArgs::Attn::SPARSEMAX)
            attn = std::make_unique<BiSparsemaxBuilder>();
        else if (attn_type == ESIMArgs::Attn::HEAD)
            attn = std::make_unique<HeadPreservingBuilder>(p, smap_opts);
        else if (attn_type == ESIMArgs::Attn::HEADMATCH)
            attn =
              std::make_unique<HeadPreservingMatchingBuilder>(p, smap_opts);
        else if (attn_type == ESIMArgs::Attn::HEADHO)
            attn = std::make_unique<HeadHOBuilder>(p, smap_opts);
        else if (attn_type == ESIMArgs::Attn::HEADMATCHHO)
            attn = std::make_unique<HeadHOMatchingBuilder>(p, smap_opts);
        else {
            std::cerr << "Unimplemented attention mechanism." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    Expr2 infer_rnn(Expression X)
    {
        unsigned sz;

        // ensure 2d
        auto d = X.dim();
        if (d.ndims() == 1) {
            sz = 1;
            X = dy::reshape(X, { hidden_dim_, sz });
        } else
            sz = d.cols();

        vector<Expression> cols(sz);

        for (size_t i = 0; i < sz; ++i)
            cols.at(i) = dy::pick(X, i, 1u);

        auto enc = dy::concatenate_cols(bilstm_inf(cols));

        auto mean_enc = dy::mean_dim(enc, { 1u });
        auto max_enc = dy::max_dim(enc, 1u);

        return std::tie(mean_enc, max_enc);
    }

    virtual void new_graph(ComputationGraph& cg)
    {
        bilstm.new_graph(cg, training_, true);
        bilstm_inf.new_graph(cg, training_, true);
        attn->new_graph(cg, training_);
    }

    virtual Expr2 syntactic_encode(const NLIPair&,
                                   const std::vector<dy::Expression>& prem,
                                   const std::vector<dy::Expression>& hypo)
    {
        auto P = dy::concatenate_cols(prem);
        auto H = dy::concatenate_cols(hypo);
        return std::forward_as_tuple(P, H);
    }

    virtual vector<Expression> predict_batch(ComputationGraph& cg,
                                             const NLIBatch& batch) override
    {
        using dy::affine_transform;
        using dy::cmult;
        using dy::concatenate;
        using dy::concatenate_cols;
        using dy::parameter;
        using dy::rectify;
        using dy::transpose;

        new_graph(cg);

        auto inf_b = parameter(cg, p_inf_b);
        auto inf_W = parameter(cg, p_inf_W);
        auto hid_b = parameter(cg, p_hid_b);
        auto hid_W = parameter(cg, p_hid_W);
        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<Expression> out;

        for (auto&& sample : batch) {
            auto enc_prem = embed_ctx_sent(cg, sample.prem),
                 enc_hypo = embed_ctx_sent(cg, sample.hypo);

            Expression P, H;
            std::tie(P, H) = syntactic_encode(sample, enc_prem, enc_hypo);

            // M is prem.size * hypo.size
            auto scores = transpose(P) * H;

            Expression U_prem, U_hypo;
            std::tie(U_prem, U_hypo) = attn->apply(scores, sample);

            // get attention-weighted context representations of other side
            auto P_ctx = H * U_hypo;
            auto H_ctx = P * U_prem;

            // augment P and H with contexts
            P = concatenate({ P, P_ctx, P - P_ctx, cmult(P, P_ctx) }, 0);
            P = rectify(affine_transform({ inf_b, inf_W, P }));

            H = concatenate({ H, H_ctx, H - H_ctx, cmult(H, H_ctx) }, 0);
            H = rectify(affine_transform({ inf_b, inf_W, H }));

            // bi-lstm, pool, concat
            Expression mean_enc_p, max_enc_p, mean_enc_h, max_enc_h;
            std::tie(mean_enc_p, max_enc_p) = infer_rnn(P);
            std::tie(mean_enc_h, max_enc_h) = infer_rnn(H);

            auto hid =
              concatenate({ mean_enc_p, max_enc_p, mean_enc_h, max_enc_h });

            hid = rectify(affine_transform({ hid_b, hid_W, hid }));

            if (training_)
                hid = dy::dropout(hid, dropout_p);

            hid = affine_transform({ out_b, out_W, hid });
            out.push_back(hid);
        }

        return out;
    }

    virtual void set_train_time() override
    {
        BaseEmbedBiLSTMModel::set_train_time();
        bilstm_inf.set_dropout(dropout_p);
    }

    virtual void set_test_time() override
    {
        BaseEmbedBiLSTMModel::set_test_time();
        bilstm_inf.disable_dropout();
    }
};
