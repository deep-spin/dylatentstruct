#include <vector>

#include <dynet/dynet.h>
#include <dynet/expr.h>

#include "builders/adjmatrix.h"
#include "data.h"

#include <iostream>

namespace dy = dynet;


Sentence make_sentence(int length)
{
    auto sent = Sentence{};

    for (int i = 0; i < length; ++i) {
        sent.word_ixs.push_back(1);
        sent.heads.push_back(0);
    }

    sent.heads.at(0) = -1;
    return sent;
}

std::vector<dy::Expression> make_ctx(dy::ComputationGraph& cg,
                                     int sentence_size) {
    const size_t d = 3;
    return std::vector<dy::Expression>(sentence_size, dy::zeros(cg, { d }));
}


int main(int argc, char** argv)
{
    dy::initialize(argc, argv);

    for (int sentence_size = 1; sentence_size < 4; ++sentence_size)
    {

        {
            dy::ComputationGraph cg;
            auto sent = make_sentence(sentence_size);
            auto ctx = make_ctx(cg, sentence_size);

            auto tree = FlatAdjacency{};
            tree.new_graph(cg, true);
            auto out = tree.make_adj(ctx, sent);
            std::cout << out.value() << "\n\n";
        }

        {
            dy::ComputationGraph cg;
            auto sent = make_sentence(sentence_size);
            auto ctx = make_ctx(cg, sentence_size);

            auto tree = LtrAdjacency{};
            tree.new_graph(cg, true);
            auto out = tree.make_adj(ctx, sent);
            std::cout << out.value() << "\n\n";
        }
    }
}


