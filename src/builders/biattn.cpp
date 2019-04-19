#include <dynet/param-init.h>
#include <dynet/devices.h>

#include "builders/biattn.h"

#include <ad3/FactorGraph.h>
#include <sparsemap.h>


#include <iostream>

namespace dy = dynet;


void
BiAttentionBuilder::new_graph(dy::ComputationGraph&, bool)
{ }

std::tuple<dynet::Expression, dynet::Expression>
BiSoftmaxBuilder::apply(const dynet::Expression& scores, const NLIPair&)
{
    auto U_hypo = dy::transpose(dy::softmax(scores, 1));
    auto U_prem = dy::transpose(dy::softmax(dy::transpose(scores), 1));

    return std::tie(U_prem, U_hypo);
}

std::tuple<dynet::Expression, dynet::Expression>
BiSparsemaxBuilder::apply(const dynet::Expression& scores, const NLIPair&)
{
    auto d = scores.dim();
    unsigned prem_sz = d[0], hypo_sz = d[1];

    dy::Expression scores_cpu;

    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);

    scores_cpu = dy::to_device(scores,
        dy::get_device_manager()->get_global_device("CPU"));

    std::vector<dy::Expression> hypo_w, prem_w;
    for (size_t i = 0; i < prem_sz; ++i)
        hypo_w.push_back(dy::sparsemax(dy::pick(scores_cpu, i, 0)));
    for (size_t i = 0; i < hypo_sz; ++i)
        prem_w.push_back(dy::sparsemax(dy::pick(scores_cpu, i, 1)));

    auto U_prem = dy::concatenate_cols(prem_w);
    auto U_hypo = dy::concatenate_cols(hypo_w);

    U_prem = dy::to_device(U_prem, device);
    U_hypo = dy::to_device(U_hypo, device);

    //std::cout << U_prem.value() << std::endl;
    //std::cout << U_hypo.value() << std::endl;
    //std::abort();

    return std::tie(U_prem, U_hypo);
}

HeadPreservingBuilder::HeadPreservingBuilder(dy::ParameterCollection& params)
    : p(params.add_subcollection("headattn"))
    //, p_affinity(p.add_parameters({1}, dy::ParameterInitConst(1)))
    , p_affinity(p.add_parameters({1},
                 dy::ParameterInitConst(1.0f),
                 "affinity",
                 dy::get_device_manager()->get_global_device("CPU")))
{}

void
HeadPreservingBuilder::new_graph(
    dy::ComputationGraph& cg,
    bool)
{
    e_affinity = dy::parameter(cg, p_affinity);
}

dynet::Expression
HeadPreservingBuilder::attend(
    const dynet::Expression scores,
    const std::vector<int>& prem_heads,
    const std::vector<int>& hypo_heads)
{

    auto d = scores.dim();
    unsigned prem_sz = d[0], hypo_sz = d[1];

    // heads are 1 greater
    //std::cout << prem_heads.size() << " "
              //<< hypo_heads.size() << "\n"
              //<< prem_sz << " " << hypo_sz << std::endl;

    auto fg = std::make_unique<AD3::FactorGraph>();
    //fg->SetVerbosity(100);

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

    unsigned n_pairs = 0;
    for (size_t j = 0; j < hypo_sz; ++j) {
        for (size_t i = 0; i < prem_sz; ++i) {

            int hi = prem_heads.at(1 + i) - 1;
            int hj = hypo_heads.at(1 + j) - 1;

            if (hi >= 0 && hj >= 0) {
                auto ij = prem_sz * j + i;
                auto hihj = prem_sz * hj + hi;
                //auto ij = hypo_sz * i + j;
                //auto hihj = hypo_sz * hi + hj;
                auto var_ij = vars.at(ij);
                auto var_hihj = vars.at(hihj);
                fg->CreateFactorPAIR({ var_ij, var_hihj }, .0f);
                ++n_pairs;
            }
        }
    }

    //fg->Print(cout);

    auto* cg = scores.pg;
    std::vector<float> ones_v(n_pairs, 1.0f);

    // cannot use dy::ones since it is on gpu
    auto ones = dy::input(*cg, { n_pairs, 1 }, ones_v,
        dy::get_device_manager()->get_global_device("CPU"));

    auto eta_u = dy::reshape(scores, { prem_sz * hypo_sz });
    auto eta_v = dy::reshape(ones * e_affinity, { n_pairs });

    //std::cout << ones.get_device_name() << std::endl;
    //std::cout << e_affinity.get_device_name() << std::endl;
    //std::cout << eta_u.get_device_name() << std::endl;
    //std::cout << eta_v.get_device_name() << std::endl;

    //std::cout << prem_sz * hypo_sz << " " << n_pairs << std::endl;
    //std::cout << eta_u.value() << std::endl << std::endl;
    //std::cout << eta_v.value() << std::endl;
    auto u = dy::sparsemap(eta_u, eta_v, std::move(fg));
    //std::cout << u.value() << std::endl;
    //std::cout << std::endl;
    u = dy::reshape(u, d);
    //return dy::reshape(u, d);
    //cg->forward(u);
    //std::cout << u.value() << std::endl;
    //cg->backward(e_affinity);
    //std::cout << "my grad" << e_affinity.gradient() << std::endl;
    //std::abort();

    return u;
    //return scores;

}

std::tuple<dynet::Expression, dynet::Expression>
HeadPreservingBuilder::apply(
    const dynet::Expression& scores,
    const NLIPair& sample)
{
    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);

    auto S = dy::to_device(scores,
        dy::get_device_manager()->get_global_device("CPU"));

    auto U_prem = attend(S, sample.prem.heads, sample.hypo.heads);
    auto U_hypo = attend(dy::transpose(S), sample.hypo.heads, sample.prem.heads);

    U_prem = dy::to_device(U_prem, device);
    U_hypo = dy::to_device(U_hypo, device);

    //auto U_prem = dy::transpose(U_hypo);
    //std::cout << U_prem.value() << std::endl;
    //std::cout << U_hypo.value() << std::endl;
    //std::abort();


    return std::tie(U_prem, U_hypo);
}
