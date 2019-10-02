#pragma once

#include <dynet/index-tensor.h>
#include <dynet/tensor.h>
#include <vector>

#include <ad3/FactorGraph.h>

#include "models/basemodel.h"
#include "sparsemap.h"

/*
 * Multilabel classifier
 */

namespace dy = dynet;

struct MultiLabelParams
{
    MultiLabelParams(dy::ParameterCollection& pc,
                     unsigned dim,
                     unsigned n_labels)
      : W_hid{ pc.add_parameters({ dim, dim }, 0, "W-hid") }
      , b_hid{ pc.add_parameters({ dim }, 0, "b-hid") }
      , W_out{ pc.add_parameters({ n_labels, dim }, 0, "W-out") }
      , b_out{ pc.add_parameters({ n_labels }, 0, "b-out") }
    {}

    dy::Parameter W_hid;
    dy::Parameter b_hid;
    dy::Parameter W_out;
    dy::Parameter b_out;
};

struct MultiLabelExprs
{
    MultiLabelExprs() = default;
    MultiLabelExprs(dy::ComputationGraph& cg, MultiLabelParams params)
    {
        W_hid = dy::parameter(cg, params.W_hid);
        b_hid = dy::parameter(cg, params.b_hid);
        W_out = dy::parameter(cg, params.W_out);
        b_out = dy::parameter(cg, params.b_out);
    }

    dy::Expression W_hid;
    dy::Expression b_hid;
    dy::Expression W_out;
    dy::Expression b_out;
};

struct MultiLabel : public BaseEmbedModel
{

    explicit MultiLabel(dy::ParameterCollection& pc,
                        unsigned vocab_size,
                        unsigned hidden_dim,
                        unsigned n_labels,
                        float dropout_p)
      : BaseEmbedModel(pc, vocab_size, hidden_dim, true)
      , ml_params(p, hidden_dim, n_labels)
      , dropout_p{ dropout_p }
      , hidden_dim{ hidden_dim }
      , n_labels{ n_labels }
    {}

    void new_graph(dy::ComputationGraph& cg)
    {
        ex = MultiLabelExprs(cg, ml_params);
    }

    dy::Expression sum_embed_sent(dy::ComputationGraph& cg,
                                  const std::vector<int>& feats)
    {
        auto sent_sz = feats.size();
        std::vector<dy::Expression> embeds(sent_sz);
        for (size_t i = 0; i < sent_sz; ++i) {
            embeds[i] = dy::lookup(cg, p_emb, feats[i]);
        }
        return dy::sum(embeds);
    }

    virtual std::vector<std::vector<float>> predict(dy::ComputationGraph& cg,
                                                    const MLBatch& batch)
    {
        auto pred = std::vector<std::vector<float>>{};
        auto out = dy::concatenate_to_batch(forward(cg, batch));
        out = dy::logistic(out);
        cg.incremental_forward(out);
        auto all_scores = out.value();
        for (auto i = 0u; i < batch.size(); ++i)
            pred.push_back(dy::as_vector(all_scores.batch_elem(i)));
        return pred;
    }

    std::vector<dy::Expression> forward(dy::ComputationGraph& cg,
                                        const MLBatch& batch)
    {
        new_graph(cg);
        std::vector<dy::Expression> out;

        for (auto& sample : batch) {
            auto x = sum_embed_sent(cg, sample.features);
            if (training_)
                x = dy::dropout(x, dropout_p);
            x = dy::rectify(x);
            x = dy::affine_transform({ ex.b_hid, ex.W_hid, x });
            if (training_)
                x = dy::dropout(x, dropout_p);
            x = dy::rectify(x);
            x = dy::affine_transform({ ex.b_out, ex.W_out, x });
            out.push_back(x);
        }

        return out;
    }

    virtual dy::Expression batch_loss(dy::ComputationGraph& cg,
                                      const MLBatch& batch)
    {
        set_train_time();
        auto out = forward(cg, batch);

        std::vector<dy::Expression> losses;
        for (size_t i = 0; i < batch.size(); ++i) {
            auto y = batch[i].labels;
            const auto data = std::vector<float>(y.size(), 1.0f);
            const auto idx = std::vector<unsigned>(y.begin(), y.end());
            auto y_vec = dy::input(cg, { n_labels }, idx, data, .0f);
            auto loss = dy::binary_log_loss(dy::logistic(out[i]), y_vec);
            losses.push_back(loss);
        }

        return dy::average(losses);
    }

    MultiLabelParams ml_params;
    MultiLabelExprs ex;
    float dropout_p;
    unsigned hidden_dim;
    unsigned n_labels;
};

struct StructuredMultiLabel : public MultiLabel
{
    explicit StructuredMultiLabel(dy::ParameterCollection& pc,
                                  unsigned vocab_size,
                                  unsigned hidden_dim,
                                  unsigned n_labels,
                                  float dropout_p,
                                  bool sparsemap,
                                  const dy::SparseMAPOpts& sm_opts)
      : MultiLabel{ pc, vocab_size, hidden_dim, n_labels, dropout_p }
      , p_corr{ p.add_parameters({ (n_labels * (n_labels - 1)) / 2 },
                                 0,
                                 "label-corr") }
      , sparsemap{ sparsemap }
      , sm_opts{ sm_opts }
    {}

    // std::tuple<dy::Expression, dy::Expression>

    std::tuple<std::vector<float>, std::vector<float>> decode(
      std::vector<float> eta_uf,
      std::vector<float> eta_vf,
      const std::vector<int> y_true,
      bool margin,
      bool qp)
    {
        auto fg = std::make_unique<AD3::FactorGraph>();
        fg->SetMaxIterationsAD3(sm_opts.max_iter);
        fg->SetEtaAD3(sm_opts.eta);
        fg->AdaptEtaAD3(sm_opts.adapt_eta);
        fg->SetResidualThresholdAD3(sm_opts.residual_thr);

        auto vars = std::vector<AD3::BinaryVariable*>{ n_labels };
        auto eta_u = std::vector<double>(eta_uf.begin(), eta_uf.end());
        auto eta_v = std::vector<double>(eta_vf.begin(), eta_vf.end());

        if (margin) {
            /* cost-augment */
            for (auto&& x : eta_u)
                x += 1.0f;
            for (auto i : y_true)
                eta_u.at(i) -= 1.0f;
        }

        std::vector<double> mu_u(eta_u.size());
        std::vector<double> mu_v(eta_v.size());

        for (auto i = 0u; i < n_labels; ++i) {
            vars.at(i) = fg->CreateBinaryVariable();
            vars.at(i)->SetLogPotential(eta_u.at(i));
        }

        auto ix = 0;
        for (auto i = 0u; i < n_labels; ++i)
            for (auto j = i + 1; j < n_labels; ++j)
                fg->CreateFactorPAIR({ vars.at(i), vars.at(j) },
                                     eta_v.at(ix++));

        double val;
        // fg->SetVerbosity(100);
        if (qp) {
            fg->SolveQP(&mu_u, &mu_v, &val);
        } else {
            fg->SolveLPMAPWithAD3(&mu_u, &mu_v, &val);
        }

        auto mu_uf = std::vector<float>(mu_u.begin(), mu_u.end());
        auto mu_vf = std::vector<float>(mu_v.begin(), mu_v.end());

        /*
        for (auto x : mu_uf)
            std::cout << x << " ";
        std::cout << std::endl;
        for (auto x : mu_vf)
            std::cout << x << " ";
        std::cout << std::endl;
        */

        return std::tie(mu_uf, mu_vf);
    }

    virtual std::vector<std::vector<float>> predict(dy::ComputationGraph& cg,
                                                    const MLBatch& batch)
    {
        auto pred = std::vector<std::vector<float>>{};
        auto eta_v = dy::parameter(cg, p_corr);
        auto out = dy::concatenate_to_batch(forward(cg, batch));
        cg.incremental_forward(out);
        auto eta_vf = dy::as_vector(eta_v.value());

        auto all_scores = out.value();
        for (auto i = 0u; i < batch.size(); ++i) {
            auto y = batch[i].labels;
            std::vector<float> mu_u, mu_v;
            auto eta_uf = dy::as_vector(all_scores.batch_elem(i));
            std::tie(mu_u, mu_v) = decode(eta_uf, eta_vf, y, false, false);
            pred.push_back(mu_u);
        }
        return pred;
    }

    virtual dy::Expression batch_loss(dy::ComputationGraph& cg,
                                      const MLBatch& batch) override
    {
        set_train_time();

        auto eta_v = dy::parameter(cg, p_corr);
        auto out = forward(cg, batch);

        cg.incremental_forward(out.at(out.size() - 1));

        auto eta_vf = dy::as_vector(eta_v.value());

        std::vector<dy::Expression> losses;
        for (size_t i = 0; i < batch.size(); ++i) {
            auto y = batch[i].labels;
            auto eta_u = out.at(i);
            auto eta_uf = dy::as_vector(eta_u.value());

            // decode
            std::vector<float> mu_u_data, mu_v_data;
            std::tie(mu_u_data, mu_v_data) =
              decode(eta_uf, eta_vf, y, !sparsemap, sparsemap);

            // put into expressions
            auto mu_u = dy::input(cg, { eta_uf.size() }, mu_u_data);
            auto mu_v = dy::input(cg, { eta_vf.size() }, mu_v_data);

            // build ground truth vector
            std::vector<float> y_u_data(eta_uf.size(), 0);
            std::vector<float> y_v_data(eta_vf.size(), 0);

            for (auto& ix : y)
                y_u_data.at(ix) = 1;

            auto k = 0u;
            for (auto i = 0u; i < n_labels; ++i) {
                for (auto j = i + 1; j < n_labels; ++j) {
                    if ((y_u_data.at(i) > 0.5) && (y_u_data.at(j) > 0.5)) {
                        y_v_data.at(k) = 1;
                    }
                    k += 1;
                }
            }

            auto y_u = dy::input(cg, { eta_uf.size() }, y_u_data);
            auto y_v = dy::input(cg, { eta_vf.size() }, y_v_data);

            auto loss = dy::dot_product(mu_u - y_u, eta_u) +
                        dy::dot_product(mu_v - y_v, eta_v);

            if (sparsemap) {
                loss =
                  loss + 0.5 * (dy::squared_norm(y_u) - dy::squared_norm(mu_u));
            } else {
                auto margin_score = .0f;

                for (auto& val : mu_u_data)
                    margin_score += val;
                for (auto& ix : y)
                    margin_score -= mu_u_data.at(ix);

                loss = loss + margin_score;
            }

            losses.push_back(loss);
        }

        return dy::average(losses);
    }

    dy::Parameter p_corr;
    bool sparsemap;
    dy::SparseMAPOpts sm_opts;
};

