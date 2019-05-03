#include "builders/biattn.h"
#include "factors/FactorMatching.h"
#include <dynet/devices.h>
#include <dynet/param-init.h>
#include <sparsemap.h>

namespace dy = dynet;

//
// Utility functions
//

// cannot use dy::ones since it is on gpu
dy::Expression
ravel(dy::Expression x)
{
    return dy::reshape(x, { x.dim().size() });
}

dy::Expression
ones_cpu(dy::ComputationGraph& cg, dy::Dim d)
{
    std::vector<float> data(d.size(), 1.0f);
    auto* cpu = dy::get_device_manager()->get_global_device("CPU");
    auto ones = dy::input(cg, d, data, cpu);
    return ones;
}

unsigned
add_head_pairs(AD3::FactorGraph* fg,
               size_t prem_sz,
               size_t hypo_sz,
               const std::vector<int>& prem_heads,
               const std::vector<int>& hypo_heads)
{
    unsigned n_pairs = 0;
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {

            int hi = prem_heads.at(1 + i) - 1;
            int hj = hypo_heads.at(1 + j) - 1;

            if (hi >= 0 && hj >= 0) {
                auto ij = prem_sz * j + i;
                auto hihj = prem_sz * hj + hi;
                auto var_ij = fg->GetBinaryVariable(ij);
                auto var_hihj = fg->GetBinaryVariable(hihj);
                fg->CreateFactorPAIR({ var_ij, var_hihj }, .0f);
                ++n_pairs;
            }
        }
    }
    return n_pairs;
}

unsigned
add_cross_pairs(AD3::FactorGraph* fg,
                size_t prem_sz,
                size_t hypo_sz,
                const std::vector<int>& prem_heads,
                const std::vector<int>& hypo_heads)
{
    unsigned n_pairs = 0;
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {

            int hi = prem_heads.at(1 + i) - 1;
            int hj = hypo_heads.at(1 + j) - 1;

            if (hi >= 0 && hj >= 0) {
                auto i_hj = prem_sz * hj + i;
                auto hi_j = prem_sz * j + hi;
                auto var_i_hj = fg->GetBinaryVariable(i_hj);
                auto var_hi_j = fg->GetBinaryVariable(hi_j);
                fg->CreateFactorPAIR({ var_i_hj, var_hi_j }, .0f);
                ++n_pairs;
            }
        }
    }
    return n_pairs;
}

unsigned
add_grandpa_pairs(AD3::FactorGraph* fg,
                  size_t prem_sz,
                  size_t hypo_sz,
                  const std::vector<int>& prem_heads,
                  const std::vector<int>& hypo_heads)
{
    //std::cout << prem_sz << " " << hypo_sz << std::endl;
    unsigned n_pairs = 0;
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {

            int hi = prem_heads.at(1 + i) - 1;
            int gi = prem_heads.at(1 + hi) - 1;

            int hj = hypo_heads.at(1 + j) - 1;
            int gj = hypo_heads.at(1 + hj) - 1;

            //std::cout << hi <<' '<< hj <<' ' << gj << std::endl;
            //std::cout << hi << ' ' << gi <<' '<< gj << std::endl;

            if (hi >= 0 && hj >= 0 && gj >= 0) {
                auto ij = prem_sz * j + i;
                auto hi_gj = prem_sz * gj + hi;
                auto var_ij = fg->GetBinaryVariable(ij);
                auto var_higj = fg->GetBinaryVariable(hi_gj);
                fg->CreateFactorPAIR({ var_ij, var_higj }, .0f);
                ++n_pairs;
            }

            if (hi >= 0 && gi >= 0 && hj >= 0) {
                auto ij = prem_sz * j + i;
                auto gi_hj = prem_sz * hj + gi;
                auto var_ij = fg->GetBinaryVariable(ij);
                auto var_gihj = fg->GetBinaryVariable(gi_hj);
                fg->CreateFactorPAIR({ var_ij, var_gihj }, .0f);
                ++n_pairs;
            }
        }
    }
    return n_pairs;
}

// ************
// Constructors
// ************

HeadPreservingBuilder::HeadPreservingBuilder(dy::ParameterCollection& params,
                                             const dy::SparseMAPOpts& opts)
  : p(params.add_subcollection("headattn"))
  , p_affinity(
      p.add_parameters({ 1 },
                       dy::ParameterInitConst(1.0f),
                       "affinity",
                       dy::get_device_manager()->get_global_device("CPU")))
  , opts(opts)
{}

HeadPreservingMatchingBuilder::HeadPreservingMatchingBuilder(
  dy::ParameterCollection& params,
  const dy::SparseMAPOpts& opts)
  : p(params.add_subcollection("headattn"))
  , p_affinity(
      p.add_parameters({ 1 },
                       dy::ParameterInitConst(1.0f),
                       "affinity",
                       dy::get_device_manager()->get_global_device("CPU")))
  , opts(opts)
{}

HeadHOBuilder::HeadHOBuilder(dy::ParameterCollection& params,
                             const dy::SparseMAPOpts& opts)
  : HeadPreservingBuilder(params, opts)
  , p_cross(
      p.add_parameters({ 1 },
                       dy::ParameterInitConst(1.0f),
                       "cross-affinity",
                       dy::get_device_manager()->get_global_device("CPU")))
  , p_grandpa(
      p.add_parameters({ 1 },
                       dy::ParameterInitConst(1.0f),
                       "grand-affinity",
                       dy::get_device_manager()->get_global_device("CPU")))
{}

HeadHOMatchingBuilder::HeadHOMatchingBuilder(dy::ParameterCollection& params,
                                             const dy::SparseMAPOpts& opts)
  : HeadPreservingMatchingBuilder(params, opts)
  , p_cross(
      p.add_parameters({ 1 },
                       dy::ParameterInitConst(1.0f),
                       "cross-affinity",
                       dy::get_device_manager()->get_global_device("CPU")))
  , p_grandpa(
      p.add_parameters({ 1 },
                       dy::ParameterInitConst(1.0f),
                       "grand-affinity",
                       dy::get_device_manager()->get_global_device("CPU")))
{}

// ***
// new graph
// ***

void
HeadPreservingBuilder::new_graph(dy::ComputationGraph& cg, bool)
{
    e_affinity = dy::parameter(cg, p_affinity);
}

void
HeadPreservingMatchingBuilder::new_graph(dy::ComputationGraph& cg, bool)
{
    e_affinity = dy::parameter(cg, p_affinity);
}

void
HeadHOBuilder::new_graph(dy::ComputationGraph& cg, bool)
{
    e_affinity = dy::parameter(cg, p_affinity);
    e_cross = dy::parameter(cg, p_cross);
    e_grandpa = dy::parameter(cg, p_grandpa);
}

void
HeadHOMatchingBuilder::new_graph(dy::ComputationGraph& cg, bool)
{
    e_affinity = dy::parameter(cg, p_affinity);
    e_cross = dy::parameter(cg, p_cross);
    e_grandpa = dy::parameter(cg, p_grandpa);
}

dynet::Expression
HeadPreservingBuilder::attend(const dynet::Expression scores,
                              const std::vector<int>& prem_heads,
                              const std::vector<int>& hypo_heads)
{

    auto d = scores.dim();
    unsigned prem_sz = d[0], hypo_sz = d[1];

    auto fg = std::make_unique<AD3::FactorGraph>();

    std::vector<AD3::BinaryVariable*> vars;

    std::vector<AD3::BinaryVariable*> vars_col(prem_sz);
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {
            auto var = fg->CreateBinaryVariable();
            vars.push_back(var);
            vars_col.at(i) = var;
        }
        fg->CreateFactorXOR(vars_col);
    }

    // PairFactor between (i, j) and (head(i), head(j))
    unsigned n_pairs =
      add_head_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);

    auto ones = ones_cpu(*scores.pg, { n_pairs, 1 });
    auto eta_u = dy::reshape(scores, { prem_sz * hypo_sz });
    auto eta_v = dy::reshape(ones * e_affinity, { n_pairs });
    auto u = dy::sparsemap(eta_u, eta_v, std::move(fg), opts);

    u = dy::reshape(u, d);
    return u;
}

dynet::Expression
HeadPreservingMatchingBuilder::attend(const dynet::Expression scores,
                                      const std::vector<int>& prem_heads,
                                      const std::vector<int>& hypo_heads)
{
    auto d = scores.dim();
    // heads are 1 greater
    unsigned prem_sz = d[0], hypo_sz = d[1];

    auto fg = std::make_unique<AD3::FactorGraph>();

    std::vector<AD3::BinaryVariable*> vars;
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {
            auto var = fg->CreateBinaryVariable();
            vars.push_back(var);
        }
    }

    // MatchingFactor over all of them
    auto* matching = new sparsemap::FactorMatching;
    fg->DeclareFactor(matching, vars, /*owned_by_graph=*/true);
    matching->Initialize(hypo_sz, prem_sz);

    // PairFactor between (i, j) and (head(i), head(j))
    unsigned n_pairs =
      add_head_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);

    auto ones = ones_cpu(*scores.pg, { n_pairs, 1 });
    auto eta_u = dy::reshape(scores, { prem_sz * hypo_sz });
    auto eta_v = dy::reshape(ones * e_affinity, { n_pairs });
    auto u = dy::sparsemap(eta_u, eta_v, std::move(fg), opts);

    u = dy::reshape(u, d);

    // std::cout << u.value() << std::endl;
    // std::cout << dy::sum_dim(u, {0u}).value() << std::endl;
    // std::cout << dy::sum_dim(u, {1u}).value() << std::endl;
    // std::abort();
    return u;
}

dynet::Expression
HeadHOBuilder::attend(const dynet::Expression scores,
                      const std::vector<int>& prem_heads,
                      const std::vector<int>& hypo_heads)
{
    auto d = scores.dim();
    unsigned prem_sz = d[0], hypo_sz = d[1];

    auto fg = std::make_unique<AD3::FactorGraph>();

    std::vector<AD3::BinaryVariable*> vars;

    std::vector<AD3::BinaryVariable*> vars_col(prem_sz);
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {
            auto var = fg->CreateBinaryVariable();
            vars.push_back(var);
            vars_col.at(i) = var;
        }
        fg->CreateFactorXOR(vars_col);
    }

    // PairFactor between (i, j) and (head(i), head(j))
    unsigned n_hd =
      add_head_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);
    unsigned n_cr =
      add_cross_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);
    unsigned n_gp =
      add_grandpa_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);

    //std::cerr << n_hd << " " << n_cr << " " << n_gp << std::endl;

    //fg->Print(std::cout);
    //fg->SetVerbosity(2);

    auto eta_u = dy::reshape(scores, { prem_sz * hypo_sz });

    auto* cg = scores.pg;
    auto eta_v_hd = ravel(ones_cpu(*cg, { n_hd, 1 }) * e_affinity);
    auto eta_v_cr = ravel(ones_cpu(*cg, { n_cr, 1 }) * e_cross);
    auto eta_v_gp = ravel(ones_cpu(*cg, { n_gp, 1 }) * e_grandpa);

    std::vector<dy::Expression> v_expr;
    if (n_hd > 0)
        v_expr.push_back(eta_v_hd);

    if (n_cr > 0)
        v_expr.push_back(eta_v_cr);

    if (n_gp > 0)
        v_expr.push_back(eta_v_gp);

    dy::Expression u;
    if (v_expr.size() > 0) {
        auto eta_v = dy::concatenate(v_expr);
        u = dy::sparsemap(eta_u, eta_v, std::move(fg), opts);
    }
    else
        u = dy::sparsemap(eta_u, std::move(fg), opts);

    u = dy::reshape(u, d);

    //std::cout << u.value() << std::endl;
    //std::abort();
    return u;
}

dynet::Expression
HeadHOMatchingBuilder::attend(const dynet::Expression scores,
                              const std::vector<int>& prem_heads,
                              const std::vector<int>& hypo_heads)
{

    auto d = scores.dim();
    unsigned prem_sz = d[0], hypo_sz = d[1];

    auto fg = std::make_unique<AD3::FactorGraph>();

    std::vector<AD3::BinaryVariable*> vars;
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {
            auto var = fg->CreateBinaryVariable();
            vars.push_back(var);
        }
    }

    // MatchingFactor over all of them
    auto* matching = new sparsemap::FactorMatching;
    fg->DeclareFactor(matching, vars, /*owned_by_graph=*/true);
    matching->Initialize(hypo_sz, prem_sz);

    unsigned n_hd =
      add_head_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);
    unsigned n_cr =
      add_cross_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);
    unsigned n_gp =
      add_grandpa_pairs(fg.get(), prem_sz, hypo_sz, prem_heads, hypo_heads);

    auto eta_u = dy::reshape(scores, { prem_sz * hypo_sz });

    auto* cg = scores.pg;
    auto eta_v_hd = ravel(ones_cpu(*cg, { n_hd, 1 }) * e_affinity);
    auto eta_v_cr = ravel(ones_cpu(*cg, { n_cr, 1 }) * e_cross);
    auto eta_v_gp = ravel(ones_cpu(*cg, { n_gp, 1 }) * e_grandpa);

    std::vector<dy::Expression> v_expr;
    if (n_hd > 0)
        v_expr.push_back(eta_v_hd);

    if (n_cr > 0)
        v_expr.push_back(eta_v_cr);

    if (n_gp > 0)
        v_expr.push_back(eta_v_gp);

    dy::Expression u;
    if (v_expr.size() > 0) {
        auto eta_v = dy::concatenate(v_expr);
        u = dy::sparsemap(eta_u, eta_v, std::move(fg), opts);
    }
    else
        u = dy::sparsemap(eta_u, std::move(fg), opts);

    u = dy::reshape(u, d);
    return u;
}
