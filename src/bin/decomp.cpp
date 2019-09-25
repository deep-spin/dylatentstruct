#include <algorithm>
#include <iostream>

#include "mlflow.h"
#include "models/decomp.h"
#include "utils.h"

namespace dy = dynet;

using std::cout;
using std::endl;
using std::vector;

int
main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);

    // the only thing fixed in this model
    unsigned EMBED_DIM = 300;

    TrainOpts opts;
    opts.parse(argc, argv);

    DecompOpts decomp_opts;
    decomp_opts.parse(argc, argv);

    AttnOpts attn_opts;
    attn_opts.parse(argc, argv);

    SparseMAPOpts smap_opts;
    smap_opts.parse(argc, argv);

    bool is_sparsemap = attn_opts.is_sparsemap();

    if (opts.override_dy) {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    std::cout << opts << std::endl;
    std::cout << decomp_opts << std::endl;

    if (is_sparsemap)
        std::cout << smap_opts << std::endl;

    std::stringstream vocab_fn, train_fn, valid_fn, test_fn, embed_fn, class_fn;
    vocab_fn << "data/nli/" << decomp_opts.dataset << ".vocab";
    class_fn << "data/nli/" << decomp_opts.dataset << ".classes";
    train_fn << "data/nli/" << decomp_opts.dataset << ".train.txt";
    //train_fn << "data/nli/" << decomp_opts.dataset << ".train-tiny.txt";
    valid_fn << "data/nli/" << decomp_opts.dataset << ".valid.txt";
    //valid_fn << "data/nli/" << decomp_opts.dataset << ".valid-tiny.txt";
    test_fn << "data/nli/" << decomp_opts.dataset << ".test.txt";
    embed_fn << "data/nli/" << decomp_opts.dataset << ".embed";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    unsigned n_classes = line_count(class_fn.str());
    cout << "n_classes: " << n_classes << endl;

    dy::ParameterCollection params;

    // clf = std::make_unique<Decomp>(params,
    std::unique_ptr<BaseNLI> clf =
      std::make_unique<Decomp>(params,
                               vocab_size,
                               EMBED_DIM,
                               opts.dim,
                               n_classes,
                               attn_opts.get_attn(),
                               smap_opts.sm_opts,
                               opts.dropout,
                               decomp_opts.update_embed);

    clf->load_embeddings(embed_fn.str(), decomp_opts.normalize_embed);

    /* log mlflow run */
    MLFlowRun mlflow(opts.mlflow_exp, opts.mlflow_host);
    mlflow.set_tag("mlflow.runName", opts.mlflow_name);

    mlflow.log_parameter("dataset", decomp_opts.dataset);

    mlflow.log_parameter("update_embed", decomp_opts.update_embed ? "true" : "false");
    mlflow.log_parameter("normalize_embed", decomp_opts.normalize_embed ? "true" : "false");

    mlflow.log_parameter("mode", opts.test ? "test" : "train");
    mlflow.log_parameter("lr", std::to_string(opts.lr));
    mlflow.log_parameter("decay", std::to_string(opts.decay));
    mlflow.log_parameter("patience", std::to_string(opts.patience));
    mlflow.log_parameter("max_iter", std::to_string(opts.max_iter));
    mlflow.log_parameter("saved_model", opts.saved_model);
    mlflow.log_parameter("batch_size", std::to_string(opts.batch_size));

    mlflow.log_parameter("dropout", std::to_string(opts.dropout));
    mlflow.log_parameter("fn_prefix", opts.save_prefix);

    mlflow.log_parameter("attention", attn_opts.attn_str);

    if (is_sparsemap) {
        mlflow.log_parameter("SM_maxit",
                             std::to_string(smap_opts.sm_opts.max_iter));
        mlflow.log_parameter("SM_thr",
                             std::to_string(smap_opts.sm_opts.residual_thr));
        mlflow.log_parameter("SM_eta", std::to_string(smap_opts.sm_opts.eta));
        mlflow.log_parameter("SM_adapt", std::to_string(smap_opts.sm_opts.adapt_eta));
        mlflow.log_parameter(
          "SM_BW_maxit", std::to_string(smap_opts.sm_opts.max_iter_backward));
        mlflow.log_parameter(
          "SM_BW_thr", std::to_string(smap_opts.sm_opts.atol_thr_backward));
        mlflow.log_parameter(
          "SM_ASET_maxit",
          std::to_string(smap_opts.sm_opts.max_active_set_iter));
    }

    // tweak filename
    std::ostringstream fn;
    fn << opts.save_prefix << "_" << opts.get_filename() << "_"
       << decomp_opts.get_filename() << "_";

    if (is_sparsemap)
        fn << smap_opts.get_filename();

    if (opts.test)
        test(clf, opts, valid_fn.str(), test_fn.str());
    else
        train(clf, opts, fn.str(), train_fn.str(), valid_fn.str(), mlflow);

    return 0;
}
