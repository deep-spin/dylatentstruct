#pragma once

#include <dynet/lstm.h>
#include <dynet/rnn.h>
#include <dynet/tensor-eigen.h>
#include <dynet/tensor.h>

#include <string>
#include <tuple>

#include "args.h"
#include "basesent.h"

namespace dy = dynet;

struct ListOps : public BaseSentClf
{
    explicit ListOps(dy::ParameterCollection& params,
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
        , dropout_{ dropout }
        , p_out_W{ p.add_parameters({ n_classes, 2 * hidden_dim } )}
        , p_out_b{ p.add_parameters({ n_classes }) }
        { }

    virtual vector<dy::Expression> predict_batch(dy::ComputationGraph& cg,
                                             const SentBatch& batch) override
    {
        bilstm.new_graph(cg, training_, true);
        auto out_b = dy::parameter(cg, p_out_b);
        auto out_W = dy::parameter(cg, p_out_W);

        vector<dy::Expression> out;

        for (auto&& sample : batch) {
            auto enc = embed_ctx_sent(cg, sample.sentence);

            auto X = dy::concatenate_cols(enc);
            auto hid = dy::concatenate({
                    dy::mean_dim(X, {1}),
                    dy::max_dim(X, 1) });

            hid = dy::affine_transform({ out_b, out_W, hid });
            out.push_back(hid);
        }
        return out;
    }

    float dropout_;
    dy::Parameter p_out_W, p_out_b;
};
