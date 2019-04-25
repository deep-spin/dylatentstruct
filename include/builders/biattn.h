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

struct HeadPreservingBuilder : BiAttentionBuilder
{

    dynet::ParameterCollection p;

    dynet::Parameter p_affinity;
    dynet::Expression e_affinity;
    dynet::SparseMAPOpts opts;

    explicit HeadPreservingBuilder(dynet::ParameterCollection& params,
                                   const dynet::SparseMAPOpts& opts);

    virtual void new_graph(dynet::ComputationGraph& cg, bool training);

    dynet::Expression attend(const dynet::Expression scores,
                             const std::vector<int>& prem_heads,
                             const std::vector<int>& hypo_heads);

    virtual std::tuple<dynet::Expression, dynet::Expression> apply(
      const dynet::Expression& scores,
      const NLIPair& sample);
};
