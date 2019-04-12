# pragma once

/* build an adj matrix using different strategies */

#include <vector>

#include <dynet/model.h>
#include <dynet/expr.h>

namespace dy = dynet;

struct TreeAdjacency
{
    virtual void new_graph(dy::ComputationGraph& cg) = 0;
    virtual dy::Expression make_adj(
        const std::vector<dy::Expression>& input,
        const std::vector<int>& heads
    ) = 0;
};


struct FixedAdjacency : TreeAdjacency
{
    dy::ComputationGraph* cg_;
    virtual void new_graph(dy::ComputationGraph& cg) override
    {
        cg_ = &cg;
    };

    virtual dy::Expression make_fixed_adj(const std::vector<unsigned>& heads)
    {
        unsigned int n = heads.size();

        /* dense
        std::vector<float> data((1 + n) * (1 + n), 0.0f);
        for (size_t i = 0; i < n; ++i)
            data[(1 + n) * (1 + i) + heads[i]] = 1;
        auto U = dy::input(*cg_, {1 + n, 1 + n}, data);
        */

        // sparse
        std::vector<float> data(n, 1.0f);
        std::vector<unsigned int> ixs(n);

        for (size_t i = 0; i < n; ++i)
            ixs[i] = (1 + n) * (1 + i) + heads[i];

        // sparse input
        auto U = dy::input(*cg_, {1 + n, 1 + n}, ixs, data);

        return U;
    }
};

struct FlatAdjacency : FixedAdjacency
{
    virtual dy::Expression make_adj(
        const std::vector<dy::Expression>&,
        const std::vector<int>& heads)
    override
    {
        size_t n = heads.size() - 1;
        std::vector<unsigned> nonneg_heads(n, 0);
        return make_fixed_adj(nonneg_heads);
    }
};

struct LtrAdjacency : FixedAdjacency
{
    virtual dy::Expression make_adj(
        const std::vector<dy::Expression>&,
        const std::vector<int>& heads)
    override
    {
        size_t n = heads.size() - 1;
        std::vector<unsigned> nonneg_heads;
        for (size_t k = 1; k < n; ++k)
            nonneg_heads.push_back(k + 1);
        nonneg_heads.push_back(0);
        return make_fixed_adj(nonneg_heads);
    }
};

struct CustomAdjacency : FixedAdjacency
{
    virtual dy::Expression make_adj(
        const std::vector<dy::Expression>&,
        const std::vector<int>& heads)
    override
    {
        std::vector<unsigned> nonneg_heads(heads.begin() + 1, heads.end());
        return make_fixed_adj(nonneg_heads);
    }
};


struct LatentAdjacency : TreeAdjacency
{
};
