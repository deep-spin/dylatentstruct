#include "layers/arcs-to-adj.h"
#include <dynet/nodes-impl-macros.h>
#include <dynet/tensor-eigen.h>

#include <iostream>

namespace dynet {

Expression
arcs_to_adj(const Expression& u,
            unsigned size)
{
    return Expression(u.pg, u.pg->add_function<ArcsToAdj>({ u.i }, size));
}

Expression
adj_to_arcs(const Expression& G)
{
    return Expression(G.pg, G.pg->add_function<AdjToArcs>({ G.i }));
}

ArcsToAdj::ArcsToAdj(const std::initializer_list<VariableIndex>& a,
                     unsigned size)
    : Node(a)
    , size(size)
{ }

AdjToArcs::AdjToArcs(const std::initializer_list<VariableIndex>& a)
    : Node(a)
{ }

std::string
ArcsToAdj::as_string(const std::vector<std::string>& arg_names) const
{
    std::ostringstream s;
    s << "arcs-to-adj(";
    for (auto&& arg_name : arg_names)
        s << arg_name << ", ";
    s << ")";
    return s.str();
}

std::string
AdjToArcs::as_string(const std::vector<std::string>& arg_names) const
{
    std::ostringstream s;
    s << "adj-to-arcs(";
    for (auto&& arg_name : arg_names)
        s << arg_name << ", ";
    s << ")";
    return s.str();
}

Dim
ArcsToAdj::dim_forward(const std::vector<Dim>&) const
{
    return {size, size};
}

Dim
AdjToArcs::dim_forward(const std::vector<Dim>& d) const
{
    int d0 = d[0][0];
    return { (unsigned) ((d0 - 1) * (d0 - 1)) };
}

template<class MyDevice>
void
ArcsToAdj::forward_dev_impl(const MyDevice&,
                            const std::vector<const Tensor*>& xs,
                            Tensor& fx) const
{
    auto u = vec(*xs[0]);
    auto adj = mat(fx);
    adj.setZero();

    size_t k = 0;
    for (size_t m = 1; m < size; ++m)
        for (size_t h = 0; h < size; ++h)
            if (h != m)
                adj(h, m) = u(k++);
}

template<class MyDevice>
void
AdjToArcs::forward_dev_impl(const MyDevice&,
                            const std::vector<const Tensor*>& xs,
                            Tensor& fx) const
{
    auto G = mat(*xs[0]);
    auto eta_u = vec(fx);
    eta_u.setZero();

    size_t size = xs[0]->d[0];

    size_t k = 0;
    for (size_t m = 1; m < size; ++m)
        for (size_t h = 0; h < size; ++h)
            if (h != m)
                eta_u(k++) = G(h, m);
}


template<class MyDevice>
void
ArcsToAdj::backward_dev_impl(const MyDevice&,
                             const std::vector<const Tensor*>&,
                             const Tensor&,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const
{
    assert(i == 0);
    auto dE_dadj = mat(dEdf);
    auto dE_du = vec(dEdxi);
    size_t k = 0;
    for (size_t m = 1; m < size; ++m)
        for (size_t h = 0; h < size; ++h)
            if (h != m)
                dE_du(k++) += dE_dadj(h, m);
}

template<class MyDevice>
void
AdjToArcs::backward_dev_impl(const MyDevice&,
                             const std::vector<const Tensor*>&,
                             const Tensor&,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const
{
    assert(i == 0);
    auto dE_dG = mat(dEdxi);
    auto dE_du = vec(dEdf);
    size_t size = dEdxi.d[0];
    size_t k = 0;
    for (size_t m = 1; m < size; ++m)
        for (size_t h = 0; h < size; ++h)
            if (h != m)
                dE_dG(h, m) += dE_du(k++);
}

DYNET_NODE_INST_DEV_IMPL(ArcsToAdj)
DYNET_NODE_INST_DEV_IMPL(AdjToArcs)

}


