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

#include "models/sentclf_model.h"

namespace dy = dynet;

using std::vector;
using std::cout;
using std::endl;


float
validate(
    std::unique_ptr<GCNSentClf>& clf,
    const vector<SentBatch>& data)
{
    int n_correct = 0;
    int n_total = 0;
    for (auto&& valid_batch : data)
    {
        dy::ComputationGraph cg;
        n_correct += clf->n_correct(cg, valid_batch);
        n_total += valid_batch.size();
    }

    return float(n_correct) / n_total;
}


void
test(
    std::unique_ptr<GCNSentClf>& clf,
    const TrainOpts& opts,
    const std::string& test_fn)
{
    clf->load(opts.saved_model);
    auto test_data = read_batches<LabeledSentence>(test_fn, opts.batch_size);
    float acc = validate(clf, test_data);
    cout << "Test accuracy: " << acc << endl;
}


void
train(
    std::unique_ptr<GCNSentClf>& clf,
    const TrainOpts& opts,
    const std::string& out_fn,
    const std::string& train_fn,
    const std::string& valid_fn,
    MLFlowRun& mlflow)
{
    auto train_data = read_batches<LabeledSentence>(train_fn, opts.batch_size);
    auto valid_data = read_batches<LabeledSentence>(valid_fn, opts.batch_size);
    auto n_batches = train_data.size();

    unsigned n_train_sents = 0;
    for (auto&& batch : train_data)
        n_train_sents += batch.size();

    std::cout << "Training on " << n_train_sents << " sentences." << std::endl;

    // make an identity permutation vector of pointers into the batches
    vector<vector<SentBatch>::iterator> train_iter(n_batches);
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

        auto training_loss = total_loss / n_train_sents;

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

    if (opts.override_dy)
    {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    unsigned EMBED_DIM = 300;

    std::stringstream vocab_fn, train_fn, valid_fn, test_fn, embed_fn, class_fn;

    train_fn << "data/sentclf/" << clf_opts.dataset << ".train.txt";
    valid_fn << "data/sentclf/" << clf_opts.dataset << ".valid.txt";
    test_fn  << "data/sentclf/" << clf_opts.dataset << ".test.txt";
    vocab_fn << "data/sentclf/" << clf_opts.dataset << ".vocab";
    class_fn << "data/sentclf/" << clf_opts.dataset << ".classes";
    embed_fn << "data/sentclf/" << clf_opts.dataset << ".embed";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    unsigned n_classes = line_count(class_fn.str());
    cout << "number of classes: " << n_classes << endl;


    dy::ParameterCollection params;
    std::unique_ptr<GCNSentClf> clf(new GCNSentClf(
            params,
            vocab_size,
            EMBED_DIM,
            /* hidden dim*/ 300,
            gcn_opts.lstm_layers,
            gcn_opts.gcn_layers,
            gcn_opts.dropout,
            gcn_opts.strat,
            n_classes));
    clf->load_embeddings(embed_fn.str());

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

    mlflow.log_parameter("strategy",    gcn_opts.strat);
    mlflow.log_parameter("lstm_layers", std::to_string(gcn_opts.lstm_layers));
    mlflow.log_parameter("gcn_layers",  std::to_string(gcn_opts.gcn_layers));
    mlflow.log_parameter("dropout",     std::to_string(gcn_opts.dropout));

    mlflow.log_parameter("fn_prefix",   opts.save_prefix);

    // tweak filename
    std::ostringstream fn;
    fn << opts.save_prefix
       << mlflow.run_uuid
       << "_sentclf_"
       << clf_opts.get_filename()
       << "_" << opts.get_filename()
       << "_" << gcn_opts.get_filename();


    if (opts.test)
        test(clf, opts, test_fn.str());
    else
        train(clf, opts, fn.str(), train_fn.str(), valid_fn.str(), mlflow);

    return 0;
}
