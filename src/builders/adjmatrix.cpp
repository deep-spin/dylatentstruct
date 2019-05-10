#include "builders/adjmatrix.h"
#include "factors/FactorTree.h"
#include <dynet/devices.h>

#include "layers/arcs-to-adj.h"

dy::Expression
make_fixed_adj(dy::ComputationGraph& cg, const std::vector<unsigned>& heads)
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
    auto U = dy::input(cg, { 1 + n, 1 + n }, ixs, data);

    return U;
}

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
    return ::make_fixed_adj(*cg_, heads);
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
    cg_ = &cg;
    scorer.new_graph(cg);
}

dy::Expression
MSTAdjacency::make_adj(const std::vector<dy::Expression>& enc, const Sentence&)
{

    auto fg = std::make_unique<AD3::FactorGraph>();
    std::vector<AD3::BinaryVariable*> vars;
    std::vector<std::tuple<int, int>> arcs;
    unsigned sz = enc.size();

    if (sz > 385) { // 99th percentile: use flat
        std::vector<unsigned> nonneg_heads(sz - 1, 0);
        return ::make_fixed_adj(*cg_, nonneg_heads);
    }

    unsigned k = 1;

    std::vector<unsigned> reverse_ixs(sz); // start with a column of zeros

    for (size_t m = 1; m < sz; ++m) {
        for (size_t h = 0; h < sz; ++h) {
            if (h != m) {
                arcs.push_back(std::make_tuple(h, m));
                vars.push_back(fg->CreateBinaryVariable());
                reverse_ixs.push_back(k);
                ++k;
            } else {
                reverse_ixs.push_back(0);
            }
        }
    }

    // ugly transfer of ownership, as in AD3. How to be safer?
    AD3::Factor* tree_factor = new AD3::FactorTree;
    fg->DeclareFactor(tree_factor, vars, /*pass_ownership=*/true);
    static_cast<AD3::FactorTree*>(tree_factor)->Initialize(sz, arcs);

    auto scores = scorer.make_potentials(enc);
    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);
    auto* cpu = dy::get_device_manager()->get_global_device("CPU");
    auto scores_cpu_matrix = dy::to_device(scores, cpu);
    auto scores_cpu = dy::adj_to_arcs(scores_cpu_matrix);

    //fg->SetVerbosity(10);
    auto u_cpu = dy::sparsemap(scores_cpu, std::move(fg), opts);
    u_cpu = dy::arcs_to_adj(u_cpu, sz);

    /*
    //u_cpu = dy::reshape(u_cpu, { u_cpu.dim()[1] });
    auto zero = dy::input(*cg_, 0.0, cpu);
    auto u_cpu_ = dy::concatenate({ zero, u_cpu });
    u_cpu_ = dy::select_rows(u_cpu_, reverse_ixs);
    u_cpu_ = dy::reshape(u_cpu_, {sz, sz});

    std::cout << u_cpu_.dim() << std::endl;
    std::cout << u_cpu_.value() << std::endl;
    std::cout << sz << std::endl << std::endl;;

    auto f = dy::arcs_to_adj(u_cpu, sz);
    std::cout << f.dim() << std::endl;
    std::cout << f.value() << std::endl;

    std::cout << u_cpu_.value() << std::endl;
    std::abort();
    */

    auto u = dy::to_device(u_cpu, device);
    return u;
}

MSTLSTMAdjacency::MSTLSTMAdjacency(dy::ParameterCollection& params,
                                   const dy::SparseMAPOpts& opts,
                                   unsigned hidden_dim)
  : MSTAdjacency{ params, opts, hidden_dim }
  , lstm{ /*layers=*/1, hidden_dim, hidden_dim, params }
{}

void
MSTLSTMAdjacency::new_graph(dy::ComputationGraph& cg)
{
    MSTAdjacency::new_graph(cg);
    lstm.new_graph(cg, /*update=*/true);
}

dy::Expression
MSTLSTMAdjacency::make_adj(const std::vector<dy::Expression>& enc,
                           const Sentence& sentence)
{
    lstm.start_new_sequence();
    auto sz = enc.size();
    std::vector<dy::Expression> lstm_out(sz);
    for (auto i = 0u; i < sz; ++i)
        lstm_out.at(i) = lstm.add_input(enc.at(i));

    return MSTAdjacency::make_adj(lstm_out, sentence);
}


MSTLSTMConstrAdjacency::MSTLSTMConstrAdjacency(dy::ParameterCollection& params,
                                               const dy::SparseMAPOpts& opts,
                                               unsigned hidden_dim,
                                               int budget)
  : MSTLSTMAdjacency{ params, opts, hidden_dim }
  , budget(budget)
{}

void
MSTLSTMConstrAdjacency::new_graph(dy::ComputationGraph& cg)
{
    MSTLSTMAdjacency::new_graph(cg);
}


dy::Expression
MSTLSTMConstrAdjacency::make_adj(const std::vector<dy::Expression>& enc,
                                 const Sentence&)
{
    lstm.start_new_sequence();
    auto sz = enc.size();
    std::vector<dy::Expression> lstm_out(sz);
    for (auto i = 0u; i < sz; ++i)
        lstm_out.at(i) = lstm.add_input(enc.at(i));

    auto fg = std::make_unique<AD3::FactorGraph>();
    std::vector<AD3::BinaryVariable*> vars;
    std::vector<std::tuple<int, int>> arcs;

    if (sz > 385) { // 99th percentile: use flat
        std::vector<unsigned> nonneg_heads(sz - 1, 0);
        return ::make_fixed_adj(*cg_, nonneg_heads);
    }

    unsigned k = 1;

    std::vector<std::vector<AD3::BinaryVariable*>> kids(sz);

    for (size_t m = 1; m < sz; ++m) {
        for (size_t h = 0; h < sz; ++h) {
            if (h != m) {
                arcs.push_back(std::make_tuple(h, m));
                auto var = fg->CreateBinaryVariable();
                vars.push_back(var);
                kids.at(h).push_back(var);
                ++k;
            }
        }
    }

    // ugly transfer of ownership, as in AD3. How to be safer?
    AD3::Factor* tree_factor = new AD3::FactorTree;
    fg->DeclareFactor(tree_factor, vars, /*pass_ownership=*/true);
    static_cast<AD3::FactorTree*>(tree_factor)->Initialize(sz, arcs);

    for (size_t h = 0; h < sz; ++h)
        fg->CreateFactorBUDGET(kids.at(h), budget, /*own=*/true);

    auto scores = scorer.make_potentials(enc);
    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);
    auto* cpu = dy::get_device_manager()->get_global_device("CPU");
    auto scores_cpu_matrix = dy::to_device(scores, cpu);
    auto scores_cpu = dy::adj_to_arcs(scores_cpu_matrix);

    //fg->SetVerbosity(10);
    auto u_cpu = dy::sparsemap(scores_cpu, std::move(fg), opts);
    u_cpu = dy::arcs_to_adj(u_cpu, sz);

    auto u = dy::to_device(u_cpu, device);
    return u;
}
