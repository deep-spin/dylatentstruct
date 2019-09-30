#include "builders/biattn.h"
#include <dynet/devices.h>
#include <dynet/param-init.h>


#include <ad3/FactorGraph.h>
#include <sparsemap.h>

#include <iostream>

namespace dy = dynet;

void
BiAttentionBuilder::new_graph(dy::ComputationGraph&, bool)
{}

std::tuple<dynet::Expression, dynet::Expression>
BiSoftmaxBuilder::apply(const dynet::Expression scores, const NLIPair&)
{
    auto U_hypo = dy::transpose(dy::softmax(scores, 1));
    auto U_prem = dy::transpose(dy::softmax(dy::transpose(scores), 1));

    return std::tie(U_prem, U_hypo);
}

std::tuple<dynet::Expression, dynet::Expression>
BiSparsemaxBuilder::apply(const dynet::Expression scores, const NLIPair&)
{
    auto d = scores.dim();
    unsigned prem_sz = d[0], hypo_sz = d[1];

    dy::Expression scores_cpu;

    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);

    scores_cpu =
      dy::to_device(scores, dy::get_device_manager()->get_global_device("CPU"));

    std::vector<dy::Expression> hypo_w, prem_w;
    for (size_t i = 0; i < prem_sz; ++i)
        hypo_w.push_back(dy::sparsemax(dy::pick(scores_cpu, i, 0)));
    for (size_t i = 0; i < hypo_sz; ++i)
        prem_w.push_back(dy::sparsemax(dy::pick(scores_cpu, i, 1)));

    auto U_prem = dy::concatenate_cols(prem_w);
    auto U_hypo = dy::concatenate_cols(hypo_w);

    U_prem = dy::to_device(U_prem, device);
    U_hypo = dy::to_device(U_hypo, device);

    return std::tie(U_prem, U_hypo);
}

std::tuple<dynet::Expression, dynet::Expression>
SymmBiAttnBuilder::apply(const dynet::Expression scores,
                         const NLIPair& sample)
{
    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);
    auto* cpu = dy::get_device_manager()->get_global_device("CPU");

    auto d = scores.dim();
    dy::Expression U_p, U_h, S;

    // for simplicity, always ensure more rows than cols
    if (d[0] < d[1]) {
        auto scores_t = dy::transpose(scores);
        S = dy::to_device(scores_t, cpu);
        U_h = attend(S, sample.hypo.heads, sample.prem.heads);
        U_h = dy::to_device(U_h, device);
        U_p = dy::transpose(U_h);
    } else {
        S = dy::to_device(scores, cpu);
        U_p = attend(S, sample.prem.heads, sample.hypo.heads);
        U_p = dy::to_device(U_p, device);
        U_h = dy::transpose(U_p);
    }
    return std::tie(U_p, U_h);

}

std::tuple<dynet::Expression, dynet::Expression>
IndepBiAttnBuilder::apply(const dynet::Expression scores,
                          const NLIPair& sample)
{
    const auto device_name = scores.get_device_name();
    auto* device = dy::get_device_manager()->get_global_device(device_name);

    auto S =
      dy::to_device(scores, dy::get_device_manager()->get_global_device("CPU"));

    auto U_p = attend(S, sample.prem.heads, sample.hypo.heads);
    auto U_h = attend(dy::transpose(S), sample.hypo.heads, sample.prem.heads);

    U_p = dy::to_device(U_p, device);
    U_h = dy::to_device(U_h, device);

    return std::tie(U_p, U_h);
}
