#include "builders/adjmatrix.h"
#include "factors/FactorTree.h"
#include <dynet/devices.h>

std::tuple<dy::Expression, dy::Expression>
TreeAdjacency::make_adj_pair(const std::vector<dy::Expression>& enc_prem,
                             const std::vector<dy::Expression>& enc_hypo,
                             const Sentence& prem,
                             const Sentence& hypo)
{
    auto&& Gprem = make_adj(enc_prem, prem);
    auto&& Ghypo = make_adj(enc_hypo, hypo);
    return std::forward_as_tuple(Gprem, Ghypo);
}

void
FixedAdjacency::new_graph(dy::ComputationGraph& cg)
{
    cg_ = &cg;
}

dy::Expression
FixedAdjacency::make_fixed_adj(const std::vector<unsigned>& heads)
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
    auto U = dy::input(*cg_, { 1 + n, 1 + n }, ixs, data);

    return U;
}

dy::Expression
FlatAdjacency::make_adj(const std::vector<dy::Expression>&,
                        const Sentence& sent)
{
    size_t n = sent.heads.size() - 1;
    std::vector<unsigned> nonneg_heads(n, 0);
    return make_fixed_adj(nonneg_heads);
}

dy::Expression
LtrAdjacency::make_adj(const std::vector<dy::Expression>&, const Sentence& sent)
{
    size_t n = sent.heads.size() - 1;
    std::vector<unsigned> nonneg_heads;
    for (size_t k = 1; k < n; ++k)
        nonneg_heads.push_back(k + 1);
    nonneg_heads.push_back(0);
    return make_fixed_adj(nonneg_heads);
}

dy::Expression
CustomAdjacency::make_adj(const std::vector<dy::Expression>&,
                          const Sentence& sent)
{
    std::vector<unsigned> nonneg_heads(sent.heads.begin() + 1,
                                       sent.heads.end());
    return make_fixed_adj(nonneg_heads);
}

MSTAdjacency::MSTAdjacency(dy::ParameterCollection& params,
                           const dy::SparseMAPOpts& opts,
                           unsigned hidden_dim)
  : opts{ opts }
  , scorer{ params, hidden_dim, hidden_dim }
{}

void
MSTAdjacency::new_graph(dy::ComputationGraph& cg)
{
    scorer.new_graph(cg);
}

dy::Expression
MSTAdjacency::make_adj(const std::vector<dy::Expression>& enc, const Sentence&)
{

    auto fg = std::make_unique<AD3::FactorGraph>();
    std::vector<AD3::BinaryVariable*> vars;
    std::vector<std::tuple<int, int>> arcs;
    size_t sz = enc.size();

    std::cout << "sent size plus root " <<  sz << std::endl;

    for (size_t m = 1; m < sz; ++m) {
        for (size_t h = 0; h < sz; ++h) {
            if (h != m) {
                arcs.push_back(std::make_tuple(h, m));
                vars.push_back(fg->CreateBinaryVariable());
            }
        }
    }

    // ugly transfer of ownership, as in AD3. How to be safer?
    AD3::Factor* tree_factor = new AD3::FactorTree;
    fg->DeclareFactor(tree_factor, vars, /*pass_ownership=*/true);
    static_cast<AD3::FactorTree*>(tree_factor)->Initialize(sz, arcs);

    std::cout << "made it here" << std::endl;
    auto scores = scorer.make_potentials(enc);
    std::cout << "made it here" << std::endl;
    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);
    auto scores_cpu =
      dy::to_device(scores, dy::get_device_manager()->get_global_device("CPU"));

    fg->SetVerbosity(10);
    auto u_cpu = dy::sparsemap(scores_cpu, std::move(fg), opts);

    std::cout << u_cpu.value() << std::endl;

    // turn this into an adjacency matrix
    //

    auto u = dy::to_device(u_cpu, device);
    return u;
}
