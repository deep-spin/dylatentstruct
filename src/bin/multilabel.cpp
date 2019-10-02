#include <algorithm>
#include <iostream>

#include "mlflow.h"
#include "utils.h"
#include <dynet/timing.h>
#include <dynet/training.h>

#include "args.h"
#include "data.h"
#include "evaluation.h"
#include "models/multilabel.h"

namespace dy = dynet;

using std::cout;
using std::endl;
using std::vector;

MultiLabelPRF
validate(std::unique_ptr<MultiLabel>& clf, const std::vector<MLBatch>& data)
{
    clf->set_test_time();
    auto prf = MultiLabelPRF{};

    for (auto&& valid_batch : data) {
        dy::ComputationGraph cg;
        auto pred = clf->predict(cg, valid_batch);

        for (auto i = 0u; i < valid_batch.size(); ++i) {
            prf.insert(pred.at(i), valid_batch.at(i).labels);
        }
    }
    return prf;
}

void
train(std::unique_ptr<MultiLabel>& clf,
      const TrainOpts& args,
      const std::string& train_fn,
      const std::string& test_fn,
      const MLFlowRun& mlflow)
{
    auto train_data =
      read_batches<MultiLabelInstance>(train_fn, args.batch_size);
    auto test_data = read_batches<MultiLabelInstance>(test_fn, args.batch_size);
    auto n_batches = train_data.size();

    unsigned n_train_sents = 0;
    for (auto&& batch : train_data)
        n_train_sents += batch.size();

    std::cout << "Training on " << n_train_sents << " instances." << std::endl;

    // make an identity permutation vector of pointers into the batches
    std::vector<std::vector<MLBatch>::iterator> train_iter(n_batches);
    std::iota(train_iter.begin(), train_iter.end(), train_data.begin());

    dy::AdamTrainer trainer(clf->p, args.lr);

    for (unsigned it = 0; it < args.max_iter; ++it) {
        // shuffle the permutation vector
        std::shuffle(train_iter.begin(), train_iter.end(), *dy::rndeng);

        float total_loss = 0;
        {
            auto timer = std::make_unique<dy::Timer>("train took");

            for (auto&& batch : train_iter) {
                dy::ComputationGraph cg;
                // cg.set_immediate_compute(true);
                // cg.set_check_validity(true);
                auto loss = clf->batch_loss(cg, *batch);
                auto loss_val = dy::as_scalar(cg.incremental_forward(loss));
                total_loss += batch->size() * loss_val;
                cg.backward(loss);
                trainer.update();
            }
        }

        float train_p, train_r, train_f;
        {
            auto timer = std::make_unique<dy::Timer>("eval took");
            auto prf = validate(clf, train_data);
            std::tie(train_p, train_r, train_f) = prf.get_prf();
        }

        float test_p, test_r, test_f;
        {
            auto timer = std::make_unique<dy::Timer>("test took");
            auto prf = validate(clf, test_data);
            std::tie(test_p, test_r, test_f) = prf.get_prf();
        }

        auto training_loss = total_loss / n_train_sents;

        mlflow.log_metric("train_loss", training_loss);
        mlflow.log_metric("effective_lr", trainer.learning_rate);
        mlflow.log_metric("train_p", train_p);
        mlflow.log_metric("train_r", train_r);
        mlflow.log_metric("train_f", train_f);
        mlflow.log_metric("test_p", test_p);
        mlflow.log_metric("test_r", test_r);
        mlflow.log_metric("test_f", test_f);

        std::cout << "Completed epoch " << it << " training loss "
                  << training_loss << "\ntraining F " << std::fixed
                  << std::setprecision(2) << train_f * 100 << "\n testing F "
                  << test_f * 100 << std::endl;
    }
}

std::pair<int, int>
count_dimensions(const std::string& filename)
{
    int max_feature = 0, max_label = 0;
    auto train_data = read_batches<MultiLabelInstance>(filename, 1);
    for (auto& batch : train_data) {
        auto instance = batch[0];
        for (auto& label : instance.labels)
            if (label > max_label)
                max_label = label;
        for (auto& feature : instance.features)
            if (feature > max_feature)
                max_feature = feature;
    }

    return std::make_pair(max_feature + 1, max_label + 1);
}

int
main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);

    TrainOpts opts;
    opts.parse(argc, argv);

    MultiLabelOpts ml_opts;
    ml_opts.parse(argc, argv);

    SparseMAPOpts smap_opts;
    smap_opts.parse(argc, argv);

    if (opts.override_dy) {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    std::cout << opts << ml_opts << std::endl;

    auto is_sparsemap = ml_opts.method != "simple";

    if (is_sparsemap)
        std::cout << smap_opts << std::endl;

    std::stringstream train_fn, test_fn;
    train_fn << "data/multilabel/" << ml_opts.dataset << ".train.txt";
    test_fn << "data/multilabel/" << ml_opts.dataset << ".test.txt";

    unsigned vocab_size, n_labels;
    std::tie(vocab_size, n_labels) = count_dimensions(train_fn.str());

    cout << "vocab_size: " << vocab_size << endl;
    cout << "n_labels: " << n_labels << endl;

    dy::ParameterCollection params;
    auto clf = std::unique_ptr<MultiLabel>{};

    if (ml_opts.method == "simple") {
        clf = std::make_unique<MultiLabel>(
          params, vocab_size, opts.dim, n_labels, opts.dropout);
    } else if (ml_opts.method == "ssvm") {
        clf = std::make_unique<StructuredMultiLabel>(params,
                                                     vocab_size,
                                                     opts.dim,
                                                     n_labels,
                                                     opts.dropout,
                                                     false,
                                                     smap_opts.sm_opts);
    } else if (ml_opts.method == "sparsemap") {
        clf = std::make_unique<StructuredMultiLabel>(params,
                                                     vocab_size,
                                                     opts.dim,
                                                     n_labels,
                                                     opts.dropout,
                                                     true,
                                                     smap_opts.sm_opts);
    }

    /* log mlflow run */
    MLFlowRun mlflow(opts.mlflow_exp, opts.mlflow_host);
    mlflow.set_tag("mlflow.runName", opts.mlflow_name);

    mlflow.log_parameter("dataset", ml_opts.dataset);
    mlflow.log_parameter("method", ml_opts.method);

    mlflow.log_parameter("lr", std::to_string(opts.lr));
    mlflow.log_parameter("decay", std::to_string(opts.decay));
    mlflow.log_parameter("max_iter", std::to_string(opts.max_iter));
    mlflow.log_parameter("batch_size", std::to_string(opts.batch_size));
    mlflow.log_parameter("dropout", std::to_string(opts.dropout));

    train(clf, opts, train_fn.str(), test_fn.str(), mlflow);
}
