#pragma once

#include <dynet/index-tensor.h>
#include <dynet/lstm.h>
#include <dynet/rnn.h>
#include <dynet/tensor-eigen.h>
#include <dynet/tensor.h>

#include <string>
#include <tuple>

#include "args.h"
#include "basemodel.h"
#include "builders/adjmatrix.h"
#include "data.h"

namespace dy = dynet;

/* ListOps dependency model:
 * For every token we learn both a vector AND a matrix */
struct ListOps : public BaseModel
{
    explicit ListOps(dy::ParameterCollection& params,
                     ListOpOpts::Tree tree_type,
                     unsigned iter,
                     unsigned vocab_size,
                     unsigned embed_dim,
                     unsigned n_classes,
                     const dy::SparseMAPOpts& smap_opts)
      : BaseModel{ params.add_subcollection("listops") }
      , iter{ iter }
      , embed_dim{ embed_dim }
      , p_emb_v{ p.add_lookup_parameters(vocab_size, { embed_dim }) }
      , p_emb_M{ p.add_lookup_parameters(vocab_size, { embed_dim, embed_dim }) }
      , p_gV{ p.add_parameters({ embed_dim, embed_dim }) }
      , p_gW{ p.add_parameters({ embed_dim, embed_dim }) }
      , p_gb{ p.add_parameters({ embed_dim }) }
      , p_out_W{ p.add_parameters({ n_classes, embed_dim }) }
      , p_out_b{ p.add_parameters({ n_classes }) }
    {
        if (tree_type == ListOpOpts::Tree::GOLD)
            tree = std::make_unique<CustomAdjacency>();
        else if (tree_type == ListOpOpts::Tree::MST)
            tree = std::make_unique<MSTAdjacency>(p, smap_opts, embed_dim);
        else if (tree_type == ListOpOpts::Tree::MST_LSTM)
            tree = std::make_unique<MSTLSTMAdjacency>(p, smap_opts, embed_dim);
    }

    int n_correct(dy::ComputationGraph& cg, const SentBatch& batch)
    {
        // set_test_time();
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

    dy::Expression batch_loss(dy::ComputationGraph& cg, const SentBatch& batch)
    {
        // set_train_time();
        auto out = predict_batch(cg, batch);

        vector<dy::Expression> losses;
        for (unsigned i = 0; i < batch.size(); ++i) {
            auto loss = dy::pickneglogsoftmax(out[i], batch[i].target);
            losses.push_back(loss);
        }

        return dy::average(losses);
    }

    dy::Expression predict_sentence(dy::ComputationGraph& cg,
                                    const Sentence& sent)
    {
        size_t sz = sent.size();

        std::vector<dy::Expression> emb_M(sz), emb_v(sz);

        /* fetch embedding vectors and matrices */
        for (size_t i = 0; i < sz; ++i) {
            emb_M.at(i) = dy::lookup(cg, p_emb_M, sent.word_ixs.at(i));
            emb_v.at(i) = dy::lookup(cg, p_emb_v, sent.word_ixs.at(i));
        }

        /* get adjacency matrix */
        auto G = tree->make_adj(emb_v, sent);
        auto Gt = dy::transpose(G);

        auto M = dy::concatenate_to_batch(emb_M);
        auto status = dy::concatenate_cols(emb_v);

        for (size_t it = 0; it < iter; ++it) {
            auto input = status * Gt;

            // match matvec
            input = dy::reshape(input, dy::Dim({ embed_dim }, sz));
            auto out = dy::reshape(dy::tanh(M * input), status.dim());

            // compute a gate
            auto gate = dy::logistic(
              dy::affine_transform({ e_gb, e_gW, status, e_gV, out }));

            status = dy::cmult(gate, status) + dy::cmult(1 - gate,  out);
        }
        auto root = dy::pick(status, /* first elem */ 0u, /* along cols */ 1u);

        return dy::affine_transform({ e_out_b, e_out_W, root });
    }

    vector<dy::Expression> predict_batch(dy::ComputationGraph& cg,
                                         const SentBatch& batch)
    {
        tree->new_graph(cg);
        e_out_b = dy::parameter(cg, p_out_b);
        e_out_W = dy::parameter(cg, p_out_W);
        // gate params
        e_gb = dy::parameter(cg, p_gb);
        e_gV = dy::parameter(cg, p_gV);
        e_gW = dy::parameter(cg, p_gW);

        std::vector<dy::Expression> out;

        for (auto&& s : batch) {
            out.push_back(predict_sentence(cg, s.sentence));
        }

        return out;
    }

    unsigned iter;
    unsigned embed_dim;
    dy::LookupParameter p_emb_v, p_emb_M;
    dy::Parameter p_gV, p_gW, p_gb;
    dy::Parameter p_out_W, p_out_b;
    std::unique_ptr<TreeAdjacency> tree;

  private:
    dy::Expression e_out_W, e_out_b;
    dy::Expression e_gV, e_gW, e_gb;
};

