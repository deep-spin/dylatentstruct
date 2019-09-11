#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/globals.h>
#include <dynet/io.h>
#include <dynet/timing.h>
#include <dynet/training.h>

#include <iostream>
#include <algorithm>

#include "utils.h"
#include "data.h"
#include "args.h"
#include "mlflow.h"
#include "crayon.h"

#include "models/tagger.h"

namespace dy = dynet;

using std::vector;
using std::cout;
using std::endl;


float
validate(
    std::unique_ptr<GCNTagger>& clf,
    const vector<TaggedBatch>& data)
{
    int n_correct = 0;
    int n_total = 0;
    for (auto&& valid_batch : data)
    {
        dy::ComputationGraph cg;
        n_correct += clf->n_correct(cg, valid_batch);
        for (auto&& sent : valid_batch)
            n_total += sent.size();
    }

    return float(n_correct) / n_total;
}


void
test(
    std::unique_ptr<GCNTagger>& clf,
    const TrainOpts& opts,
    const std::string& valid_fn,
    const std::string& test_fn)
{
    clf->load(opts.saved_model);

    clf->tree->set_print(opts.saved_model + "valid-trees.txt");
    auto valid_data = read_batches<TaggedSentence>(valid_fn, opts.batch_size);
    float valid_acc = validate(clf, valid_data);
    cout << "Valid accuracy: " << valid_acc << endl;

    clf->tree->set_print(opts.saved_model + "test-trees.txt");
    auto test_data = read_batches<TaggedSentence>(test_fn, opts.batch_size);
    float test_acc = validate(clf, test_data);
    cout << "Test accuracy: " << test_acc << endl;
}


void
train(
    std::unique_ptr<GCNTagger>& clf,
    const TrainOpts& opts,
    const std::string& out_fn,
    const std::string& train_fn,
    const std::string& valid_fn,
    MLFlowRun& mlflow)
{
    auto train_data = read_batches<TaggedSentence>(train_fn, opts.batch_size);
    auto valid_data = read_batches<TaggedSentence>(valid_fn, opts.batch_size);
    auto n_batches = train_data.size();

    unsigned n_train_toks = 0;
    for (auto&& batch : train_data)
        for (auto&& sent : batch)
            n_train_toks += sent.size();

    std::cout << "Training on " << n_train_toks << " tokens." << std::endl;

    // make an identity permutation vector of pointers into the batches
    vector<vector<TaggedBatch>::iterator> train_iter(n_batches);
    std::iota(train_iter.begin(), train_iter.end(), train_data.begin());

    //dy::SimpleSGDTrainer trainer(clf->p, opts.lr);
    dy::AdamTrainer trainer(clf->p, opts.lr);
    trainer.clip_threshold = 100;
    trainer.sparse_updates_enabled = false;

    float best_valid_acc = 0;
    float last_valid_acc = 0;
    unsigned impatience = 0;

    Crayon crayon(out_fn);

    for (unsigned it = 0; it < opts.max_iter; ++it)
    {
        // shuffle the permutation vector
        std::shuffle(train_iter.begin(), train_iter.end(), *dy::rndeng);

        float total_loss = 0;

        {
            std::unique_ptr<dy::Timer> timer(new dy::Timer("train took"));
            for (auto&& batch : train_iter)
            {
                dy::ComputationGraph cg;
                auto loss = clf->batch_loss(cg, *batch);
                auto lossval = dy::as_scalar(cg.incremental_forward(loss));
                total_loss += lossval;
                cg.backward(loss);
                trainer.update();
            }
        }

        float valid_acc;
        {
            std::unique_ptr<dy::Timer> timer(new dy::Timer("valid took"));
            valid_acc = validate(clf, valid_data);
        }

        auto training_loss = total_loss / n_train_toks;

        mlflow.log_metric("train_loss",   training_loss);
        mlflow.log_metric("valid_acc",    valid_acc);
        mlflow.log_metric("effective_lr", trainer.learning_rate);

        crayon.log_metric("train_loss",   training_loss, 1 + it);
        crayon.log_metric("valid_acc",    valid_acc, 1 + it);
        crayon.log_metric("effective_lr", trainer.learning_rate, 1 + it);

        std::cout << "training loss " << training_loss
                  << " valid accuracy " << valid_acc << std::endl;

        if ((valid_acc + 0.0001) > last_valid_acc)
        {
            impatience = 0;
        }
        else
        {
            trainer.learning_rate *= opts.decay;
            cout << "Decaying LR to " << trainer.learning_rate << endl;
            impatience += 1;
        }

        if (valid_acc > best_valid_acc)
        {
            best_valid_acc = valid_acc;

            std::ostringstream fn;
            fn << out_fn
               << "_acc_"
               << std::internal << std::setfill('0')
               << std::fixed << std::setprecision(2) << std::setw(5)
               << valid_acc * 100.0
               << "_iter_" << std::setw(3) << it
               << ".dy";
            clf->save(fn.str());
        }

        if (impatience > opts.patience)
        {
            cout << opts.patience << " epochs without improvement." << endl;
            break;
        }
        last_valid_acc = valid_acc;
    }
    mlflow.log_metric("best_valid_acc", best_valid_acc);
}


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);

    TrainOpts opts;
    opts.parse(argc, argv);
    std::cout << opts << std::endl;

    ClfOpts clf_opts;
    clf_opts.parse(argc, argv);
    std::cout << clf_opts << std::endl;

    GCNOpts gcn_opts;
    gcn_opts.parse(argc, argv);
    std::cout << gcn_opts << std::endl;

    SparseMAPOpts smap_opts;
    smap_opts.parse(argc, argv);

    bool is_sparsemap = gcn_opts.get_tree() == GCNOpts::Tree::MST;

    if (opts.override_dy)
    {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    unsigned EMBED_DIM = 300;
    unsigned HIDDEN_DIM = 300;

    std::stringstream vocab_fn, train_fn, valid_fn, test_fn, embed_fn, class_fn;

    train_fn << "data/tag/" << clf_opts.dataset << ".train";
    valid_fn << "data/tag/" << clf_opts.dataset << ".valid";
    test_fn  << "data/tag/" << clf_opts.dataset << ".test";
    vocab_fn << "data/tag/" << clf_opts.dataset << ".vocab";
    class_fn << "data/tag/" << clf_opts.dataset << ".classes";
    embed_fn << "data/tag/" << clf_opts.dataset << ".embed";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    unsigned n_classes = line_count(class_fn.str());
    cout << "number of classes: " << n_classes << endl;

    dy::ParameterCollection params;
    auto clf = std::make_unique<GCNTagger>(
        params,
        vocab_size,
        EMBED_DIM,
        HIDDEN_DIM,
        n_classes,
        opts.dropout,
        gcn_opts,
        smap_opts.sm_opts);

    //clf->load_embeddings(embed_fn.str());
    //
    if (opts.test) {
        test(clf, opts, valid_fn.str(), test_fn.str());
        return 0;
    }

    /* log mlflow run options */
    MLFlowRun mlflow(opts.mlflow_exp);

    mlflow.set_tag("mlflow.runName", opts.mlflow_name);

    mlflow.log_parameter("dataset",     clf_opts.dataset);

    mlflow.log_parameter("mode",        opts.test ? "test" : "train");
    mlflow.log_parameter("lr",          std::to_string(opts.lr));
    mlflow.log_parameter("decay",       std::to_string(opts.decay));
    mlflow.log_parameter("patience",    std::to_string(opts.patience));
    mlflow.log_parameter("max_iter",    std::to_string(opts.max_iter));
    mlflow.log_parameter("saved_model", opts.saved_model);
    mlflow.log_parameter("batch_size",  std::to_string(opts.batch_size));
    mlflow.log_parameter("dropout",     std::to_string(opts.dropout));

    mlflow.log_parameter("strategy",    gcn_opts.tree_str);
    mlflow.log_parameter("gcn_layers",  std::to_string(gcn_opts.layers));
    mlflow.log_parameter("gcn_iter",    std::to_string(gcn_opts.iter));
    mlflow.log_parameter("gcn_dropout", std::to_string(gcn_opts.dropout));

    mlflow.log_parameter("fn_prefix",   opts.save_prefix);

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
    fn << opts.save_prefix
       << mlflow.run_uuid
       << "_"
       << clf_opts.get_filename()
       << "_" << opts.get_filename()
       << "_" << gcn_opts.get_filename();

    if (is_sparsemap)
        fn << smap_opts.get_filename();

    train(clf, opts, fn.str(), train_fn.str(), valid_fn.str(), mlflow);
    return 0;
}
