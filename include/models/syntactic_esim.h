#pragma once

#include <string>
#include <tuple>

#include "args.h"
#include "builders/adjmatrix.h"
#include "builders/gcn.h"
#include "models/esim.h"

#include <iostream>

struct SyntacticESIM : public ESIM
{
    explicit SyntacticESIM(ParameterCollection& params,
                           unsigned vocab_size,
                           unsigned embed_dim,
                           unsigned hidden_dim,
                           unsigned n_classes,
                           unsigned gcn_layers,
                           GCNOpts::Tree tree_type,
                           ESIMArgs::Attn attn_type,
                           const dy::SparseMAPOpts& smap_opts,
                           float dropout_p = .5,
                           float gcn_dropout_p = .1,
                           unsigned stacks = 1,
                           bool update_embed = true)
      : ESIM{ params,    vocab_size, embed_dim, hidden_dim, n_classes,
              attn_type, smap_opts,  dropout_p, stacks,     update_embed }
      , gcn_settings{ hidden_dim, gcn_layers, /*dense=*/true }
      , gcn{ p, gcn_settings, hidden_dim }
      , p_Ws(p.add_parameters({ hidden_dim, 2 * hidden_dim }))
      , p_bs(p.add_parameters({ hidden_dim }))
      {
        if (tree_type == GCNOpts::Tree::LTR)
            tree = std::make_unique<LtrAdjacency>();
        else if (tree_type == GCNOpts::Tree::FLAT)
            tree = std::make_unique<FlatAdjacency>();
        else if (tree_type == GCNOpts::Tree::GOLD)
            tree = std::make_unique<CustomAdjacency>();
        else if (tree_type == GCNOpts::Tree::MST)
            tree = std::make_unique<MSTAdjacency>(p, smap_opts, hidden_dim);
        else {
            std::cerr << "Not implemented";
            std::abort();
        }

        std::cout << gcn_dropout_p << dropout_p << std::endl;

        gcn.set_dropout(gcn_dropout_p);
    }

    virtual void new_graph(dy::ComputationGraph& cg) override
    {
        ESIM::new_graph(cg);
        gcn.new_graph(cg, training_, true);
        tree->new_graph(cg);

        Ws = dy::parameter(cg, p_Ws);
        bs = dy::parameter(cg, p_bs);
    }

    virtual Expr2 syntactic_encode(
      const NLIPair& sample,
      const std::vector<dy::Expression>& prem,
      const std::vector<dy::Expression>& hypo) override
    {

        auto pg = prem[0].pg;

        // make a copy, cheap;
        auto prem_ = prem;
        auto hypo_ = hypo;

        prem_.insert(prem_.begin(), dy::zeros(*pg, { hidden_dim_ }));
        hypo_.insert(hypo_.begin(), dy::zeros(*pg, { hidden_dim_ }));

        auto P = dy::concatenate_cols(prem_);
        auto H = dy::concatenate_cols(hypo_);

        dy::Expression Gp, Gh;
        std::tie(Gp, Gh) =
          tree->make_adj_pair(prem_, hypo_, sample.prem, sample.hypo);

        auto P_enc = gcn.apply(P, Gp);
        auto H_enc = gcn.apply(H, Gh);

        P_enc = dy::rectify(dy::affine_transform({bs, Ws, P_enc}));
        H_enc = dy::rectify(dy::affine_transform({bs, Ws, H_enc}));

        //return std::forward_as_tuple(P_enc, H_enc);

        return std::tie(P_enc, H_enc);
    }


    GCNSettings gcn_settings;
    GCNBuilder gcn;
    std::unique_ptr<TreeAdjacency> tree;
    dy::Parameter p_Ws, p_bs;
    dy::Expression Ws, bs;
};

