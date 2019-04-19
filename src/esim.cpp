#include <iostream>
#include <algorithm>

#include "utils.h"
#include "models/esim.h"

namespace dy = dynet;

using std::vector;
using std::cout;
using std::endl;


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);

    // the only thing fixed in this model
    unsigned EMBED_DIM = 300;

    TrainOpts opts;
    opts.parse(argc, argv);
    std::cout << opts << std::endl;

    ESIMArgs args;
    args.parse(argc, argv);
    std::cout << args << std::endl;

    if (opts.override_dy)
    {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    std::stringstream vocab_fn, train_fn, valid_fn, test_fn, embed_fn, class_fn;
    vocab_fn << "data/nli/" << args.dataset << ".vocab";
    class_fn << "data/nli/" << args.dataset << ".classes";
    //train_fn << "data/nli/" << args.dataset << ".train.txt";
    train_fn << "data/nli/" << args.dataset << ".train-small.txt";
    //valid_fn << "data/nli/" << args.dataset << ".valid.txt";
     valid_fn << "data/nli/" << args.dataset << ".train-small.txt";
    test_fn  << "data/nli/" << args.dataset << ".test.txt";
    // test_fn  << "data/nli/" << args.dataset << ".train-tiny.txt";
    embed_fn << "data/nli/" << args.dataset << ".embed";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    size_t n_classes = 3;

    dy::ParameterCollection params;

    std::unique_ptr<BaseNLI> clf = nullptr;

    clf.reset(new ESIM(params,
                       vocab_size,
                       EMBED_DIM,
                       300, // hidden dim
                       n_classes,
                       args.get_attn(),
                       args.dropout,
                       true, // update emb
                       args.max_decode_iter));

    clf->load_embeddings(embed_fn.str());

    if (opts.test)
        test(clf, opts, valid_fn.str(), test_fn.str());
    else
        train(clf, opts, train_fn.str(), valid_fn.str());

    return 0;
}
