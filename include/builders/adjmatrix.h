#pragma once

/* build an adj matrix using different strategies */

#include <vector>

#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/lstm.h>

#include "data.h"

#include "builders/arcscorers.h"
#include "builders/distance-bias.h"
#include "sparsemap.h"

namespace dy = dynet;

struct TreeAdjacency
{
    virtual void new_graph(dy::ComputationGraph& cg) = 0;
    virtual dy::Expression make_adj(const std::vector<dy::Expression>& input,
                                    const Sentence& sent) = 0;

    /* This is so that we can jointly learn two trees with cross-constraints */
    virtual std::tuple<dy::Expression, dy::Expression> make_adj_pair(
      const std::vector<dy::Expression>& enc_prem,
      const std::vector<dy::Expression>& enc_hypo,
      const Sentence& prem,
      const Sentence& hypo);
};

struct FixedAdjacency : TreeAdjacency
{
    dy::ComputationGraph* cg_;
    virtual void new_graph(dy::ComputationGraph& cg) override;
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
    dy::ComputationGraph* cg_;
    dy::SparseMAPOpts opts;
    //MLPScoreBuilder scorer;
    BilinearScoreBuilder scorer;
    DistanceBiasBuilder distance_bias;
    int budget;

    explicit MSTAdjacency(dy::ParameterCollection& params,
                          const dy::SparseMAPOpts& opts,
                          unsigned hidden_dim,
                          bool use_distance=true,
                          int budget=0);

    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
    virtual void new_graph(dy::ComputationGraph& cg) override;

};


struct MSTLSTMAdjacency : MSTAdjacency
{

    dy::VanillaLSTMBuilder lstm;

    explicit MSTLSTMAdjacency(dy::ParameterCollection& params,
                              const dy::SparseMAPOpts& opts,
                              unsigned hidden_dim);

    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
    virtual void new_graph(dy::ComputationGraph& cg) override;
};

struct MSTLSTMConstrAdjacency : MSTLSTMAdjacency
{

    explicit MSTLSTMConstrAdjacency(dy::ParameterCollection& params,
                                    const dy::SparseMAPOpts& opts,
                                    unsigned hidden_dim,
                                    int budget);

    virtual dy::Expression make_adj(const std::vector<dy::Expression>&,
                                    const Sentence& sent) override;
    void new_graph(dy::ComputationGraph& cg) override;

    int budget;
};
