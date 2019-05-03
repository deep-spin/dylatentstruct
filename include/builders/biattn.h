#pragma once

/* Bi-Attention between two sentences (as in ESIM) */

#include <vector>

#include <dynet/expr.h>
#include <dynet/model.h>

#include "data.h"
#include "sparsemap.h"

struct BiAttentionBuilder
{
    virtual void new_graph(dynet::ComputationGraph& cg, bool training);
    virtual std::tuple<dynet::Expression, dynet::Expression> apply(
      const dynet::Expression& scores,
      const NLIPair& sample) = 0;
};

struct BiSoftmaxBuilder : BiAttentionBuilder
{

    virtual std::tuple<dynet::Expression, dynet::Expression> apply(
      const dynet::Expression& scores,
      const NLIPair& sample);
};

struct BiSparsemaxBuilder : BiAttentionBuilder
{

    virtual std::tuple<dynet::Expression, dynet::Expression> apply(
      const dynet::Expression& scores,
      const NLIPair& sample);
};

struct SymmBiAttnBuilder : BiAttentionBuilder
{
    virtual dynet::Expression attend(const dynet::Expression scores,
                                     const std::vector<int>& prem_heads,
                                     const std::vector<int>& hypo_heads) = 0;

    virtual std::tuple<dynet::Expression, dynet::Expression> apply(
      const dynet::Expression& scores,
      const NLIPair& sample);
};

struct IndepBiAttnBuilder : BiAttentionBuilder
{
    virtual dynet::Expression attend(const dynet::Expression scores,
                                     const std::vector<int>& prem_heads,
                                     const std::vector<int>& hypo_heads) = 0;

    virtual std::tuple<dynet::Expression, dynet::Expression> apply(
      const dynet::Expression& scores,
      const NLIPair& sample);
};

struct HeadPreservingBuilder : IndepBiAttnBuilder
{

    dynet::ParameterCollection p;

    dynet::Parameter p_affinity;
    dynet::Expression e_affinity;
    dynet::SparseMAPOpts opts;

    explicit HeadPreservingBuilder(dynet::ParameterCollection& params,
                                   const dynet::SparseMAPOpts& opts);

    virtual void new_graph(dynet::ComputationGraph& cg, bool training);

    virtual dynet::Expression attend(const dynet::Expression scores,
                                     const std::vector<int>& prem_heads,
                                     const std::vector<int>& hypo_heads);
};

struct HeadPreservingMatchingBuilder : SymmBiAttnBuilder
{

    dynet::ParameterCollection p;

    dynet::Parameter p_affinity;
    dynet::Expression e_affinity;
    dynet::SparseMAPOpts opts;

    explicit HeadPreservingMatchingBuilder(dynet::ParameterCollection& params,
                                           const dynet::SparseMAPOpts& opts);

    virtual void new_graph(dynet::ComputationGraph& cg, bool training);

    virtual dynet::Expression attend(const dynet::Expression scores,
                                     const std::vector<int>& prem_heads,
                                     const std::vector<int>& hypo_heads);
};
