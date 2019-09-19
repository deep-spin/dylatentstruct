#include <algorithm>
#include <iostream>

#include "mlflow.h"
#include "models/esim.h"
#include "models/syntactic_esim.h"
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

    ESIMOpts esim_opts;
    esim_opts.parse(argc, argv);

    AttnOpts attn_opts;
    attn_opts.parse(argc, argv);

    GCNOpts gcn_opts;
    gcn_opts.parse(argc, argv);

    SparseMAPOpts smap_opts;
    smap_opts.parse(argc, argv);

    bool is_gcn = gcn_opts.layers > 0;
    bool is_sparsemap = (attn_opts.is_sparsemap() ||
                         gcn_opts.is_sparsemap());

    if (opts.override_dy) {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    std::cout << opts << std::endl;
    std::cout << esim_opts << std::endl;
    std::cout << attn_opts << std::endl;

    if (is_sparsemap)
        std::cout << smap_opts << std::endl;

    if (is_gcn)
        std::cout << gcn_opts << std::endl;

    std::stringstream vocab_fn, train_fn, valid_fn, test_fn, embed_fn, class_fn;
    vocab_fn << "data/nli/" << esim_opts.dataset << ".vocab";
    class_fn << "data/nli/" << esim_opts.dataset << ".classes";
    train_fn << "data/nli/" << esim_opts.dataset << ".train.txt";
    // train_fn << "data/nli/" << esim_opts.dataset << ".train-tiny.txt";
    valid_fn << "data/nli/" << esim_opts.dataset << ".valid.txt";
    // valid_fn << "data/nli/" << esim_opts.dataset << ".valid-tiny.txt";
    test_fn << "data/nli/" << esim_opts.dataset << ".test.txt";
    embed_fn << "data/nli/" << esim_opts.dataset << ".embed";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    unsigned n_classes = line_count(class_fn.str());
    cout << "n_classes: " << n_classes << endl;

    dy::ParameterCollection params;

    std::unique_ptr<BaseNLI> clf;

    if (is_gcn)
        clf = std::make_unique<SyntacticESIM>(params,
                                              vocab_size,
                                              EMBED_DIM,
                                              /* hidden_dim = */ 300,
                                              n_classes,
                                              gcn_opts.layers,
                                              gcn_opts.get_tree(),
                                              attn_opts.get_attn(),
                                              smap_opts.sm_opts,
                                              esim_opts.dropout,
                                              gcn_opts.dropout,
                                              /* lstm_stacks = */ 1,
                                              /* update_embed = */ true);

    else
        clf = std::make_unique<ESIM>(params,
                                     vocab_size,
                                     EMBED_DIM,
                                     /* hidden_dim = */ 300,
                                     n_classes,
                                     attn_opts.get_attn(),
                                     smap_opts.sm_opts,
                                     esim_opts.dropout,
                                     /* lstm_stacks = */ 1,
                                     /* update_embed = */ true);

    clf->load_embeddings(embed_fn.str());

    /* log mlflow run */
    MLFlowRun mlflow(opts.mlflow_exp, opts.mlflow_host);
    mlflow.set_tag("mlflow.runName", opts.mlflow_name);

    mlflow.log_parameter("dataset", esim_opts.dataset);

    mlflow.log_parameter("mode", opts.test ? "test" : "train");
    mlflow.log_parameter("lr", std::to_string(opts.lr));
    mlflow.log_parameter("decay", std::to_string(opts.decay));
    mlflow.log_parameter("patience", std::to_string(opts.patience));
    mlflow.log_parameter("max_iter", std::to_string(opts.max_iter));
    mlflow.log_parameter("saved_model", opts.saved_model);
    mlflow.log_parameter("batch_size", std::to_string(opts.batch_size));
    mlflow.log_parameter("attn", attn_opts.attn_str);

    mlflow.log_parameter("dropout", std::to_string(esim_opts.dropout));
    mlflow.log_parameter("fn_prefix", opts.save_prefix);

    if (is_gcn) {
        mlflow.log_parameter("GCN_layers", std::to_string(gcn_opts.layers));
        mlflow.log_parameter("GCN_tree_type", gcn_opts.tree_str);
        mlflow.log_parameter("GCN_dropout", std::to_string(gcn_opts.dropout));
    }

    if (is_sparsemap) {
        mlflow.log_parameter("SM_maxit",
                             std::to_string(smap_opts.sm_opts.max_iter));
        mlflow.log_parameter("SM_thr",
                             std::to_string(smap_opts.sm_opts.residual_thr));
        mlflow.log_parameter("SM_eta", std::to_string(smap_opts.sm_opts.eta));
        mlflow.log_parameter("SM_adapt",
                             std::to_string(smap_opts.sm_opts.adapt_eta));
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
    fn << opts.save_prefix << "_esim_"
       << "_" << opts.get_filename()
       << "_" << esim_opts.get_filename()
       << "_" << attn_opts.get_filename();

    if (is_sparsemap)
        fn << "_" << smap_opts.get_filename();

    if (is_gcn)
        fn << "_" << gcn_opts.get_filename();

    if (opts.test)
        test(clf, opts, valid_fn.str(), test_fn.str());
    else
        train(clf, opts, fn.str(), train_fn.str(), valid_fn.str(), mlflow);

    return 0;
}
