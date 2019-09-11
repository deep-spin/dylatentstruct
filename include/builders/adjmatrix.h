#pragma once

/* build an adj matrix using different strategies */

#include <vector>

#include <dynet/expr.h>
#include <dynet/model.h>

#include "data.h"

#include "builders/arcscorers.h"
#include "builders/bilstm.h"
#include "builders/distance-bias.h"
#include "sparsemap.h"

namespace dy = dynet;

struct TreeAdjacency
{
    virtual void new_graph(dy::ComputationGraph& cg, bool training) = 0;
    virtual dy::Expression make_adj(const std::vector<dy::Expression>& input,
                                    const Sentence& sent) = 0;

    /* This is so that we can jointly learn two trees with cross-constraints */
    virtual std::tuple<dy::Expression, dy::Expression> make_adj_pair(
      const std::vector<dy::Expression>& enc_prem,
      const std::vector<dy::Expression>& enc_hypo,
      const Sentence& prem,
      const Sentence& hypo);

    virtual void set_print(const std::string&) {};
    virtual void clear_print() {};
};

struct FixedAdjacency : TreeAdjacency
{
    dy::ComputationGraph* cg_;
    virtual void new_graph(dy::ComputationGraph& cg, bool training) override;
    virtual dy::Expression make_fixed_adj(const std::vector<unsigned>& heads);
};

struct FlatAdjacency : FixedAdjacency
{
    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
};

struct LtrAdjacency : FixedAdjacency
{
    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
};

struct CustomAdjacency : FixedAdjacency
{
    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
};

struct MSTAdjacency : TreeAdjacency
{
    explicit MSTAdjacency(dy::ParameterCollection& params,
                          const dy::SparseMAPOpts& opts,
                          unsigned hidden_dim,
                          bool use_distance=true,
                          int budget=0);

    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
    virtual void new_graph(dy::ComputationGraph& cg, bool training) override;

    virtual void set_print(const std::string& fn) override {
        opts.log_stream = std::make_shared<std::ofstream>(fn);
    }

    virtual void clear_print() override {
        opts.log_stream.reset();
    }

    dy::ComputationGraph* cg_;
    dy::SparseMAPOpts opts;
    BilinearScoreBuilder scorer;
    DistanceBiasBuilder distance_bias;
    int budget;
};


struct MSTLSTMAdjacency : MSTAdjacency
{

    explicit MSTLSTMAdjacency(dy::ParameterCollection& params,
                              const dy::SparseMAPOpts& opts,
                              unsigned hidden_dim,
                              float dropout_p=.0f,
                              int budget=0);

    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
    virtual void new_graph(dy::ComputationGraph& cg, bool training) override;

    BiLSTMSettings bilstm_settings;
    BiLSTMBuilder bilstm;
    float dropout_p;
};

