#include <vector>

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>

#include "builders/adjmatrix.h"
#include "sparsemap.h"
#include "data.h"

namespace dy = dynet;

int main(int argc, char** argv)
{
    dy::initialize(argc, argv);
    const unsigned int vocab_size{ 5 };
    const unsigned int embed_dim{ 10 };

    dy::ParameterCollection params;
    auto emb = params.add_lookup_parameters(vocab_size, { embed_dim });

    dy::SparseMAPOpts opts;
    //opts.eta = 0;
    //opts.max_iter = 1;
    //opts.max_active_set_iter = 50;

    auto sent = Sentence{};
    sent.word_ixs.push_back(1);
    sent.word_ixs.push_back(3);
    sent.word_ixs.push_back(2);
    sent.word_ixs.push_back(0);

    sent.heads.push_back(-1);
    sent.heads.push_back(2);
    sent.heads.push_back(0);
    sent.heads.push_back(2);
    sent.heads.push_back(2);

    for (auto && h : sent.word_ixs)
        std::cout << h << " ";
    std::cout << std::endl;
    for (auto && h : sent.heads)
        std::cout << h << " ";
    std::cout << std::endl;

    //auto tree = MSTLSTMAdjacency{ params, opts, embed_dim };
    auto tree = MSTAdjacency{ params, opts, embed_dim, false, 0 };

    dy::AdamTrainer trainer(params, 0.01);

    const auto n_iter = 20;

    for (auto i = 0u; i < n_iter; ++i) {

        dy::ComputationGraph cg;
        tree.new_graph(cg, true);

        auto enc = std::vector<dy::Expression>{};
        enc.push_back(dy::zeros(cg, { embed_dim }));
        for (auto && i : sent.word_ixs)
            enc.push_back(dy::lookup(cg, emb, i));

        auto out = tree.make_adj(enc, sent);

        if (i == n_iter - 1)
            std::cout << out.value() << std::endl;

        auto y_true_cols = std::vector<dy::Expression>{};
        y_true_cols.push_back(dy::zeros(cg, { 5u }));
        y_true_cols.push_back(dy::one_hot(cg, 5u, 2u));
        y_true_cols.push_back(dy::one_hot(cg, 5u, 0u));
        y_true_cols.push_back(dy::one_hot(cg, 5u, 2u));
        y_true_cols.push_back(dy::one_hot(cg, 5u, 2u));
        auto y_true = dy::concatenate_cols(y_true_cols);

        auto loss = dy::squared_distance(y_true, out);
        cg.forward(loss);
        std::cout << loss.value() << std::endl;
        cg.backward(loss);
        trainer.update();
    }

        //auto diff = dy::squared_distance(y_true, out);
        ////auto diff = y_true - out;
        ////diff = dy::reshape(diff, { 25 });

        //cg.incremental_forward(diff);
        //std::cout << diff.value() << std::endl;
}


