#include "utils.h"
#include "data.h"
#include "args.h"
#include "mlflow.h"
#include "models/listops_model.h"

#include <dynet/timing.h>
#include <dynet/training.h>

#include <memory>

namespace dy = dynet;

float
validate(ListOps& clf, const vector<SentBatch>& data)
{
    int n_correct = 0;
    int n_total = 0;
    for (auto&& valid_batch : data) {
        dy::ComputationGraph cg;
        n_correct += clf.n_correct(cg, valid_batch);
        n_total += valid_batch.size();
    }

    return float(n_correct) / n_total;
}

void
train(ListOps& clf,
      const TrainOpts& opts,
      const std::string& out_fn,
      const std::string& train_fn,
      const std::string& valid_fn,
      const MLFlowRun& mlflow
      )
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

    dy::AdamTrainer trainer(clf.p, opts.lr);

    size_t patience = 0;
    float best_valid_acc = 0;
    float last_valid_acc = 0;

    //Crayon crayon(out_fn, mlflow.hostname);

    for (unsigned it = 0; it < opts.max_iter; ++it) {
        // shuffle the permutation vector
        std::shuffle(train_iter.begin(), train_iter.end(), *dy::rndeng);

        float total_loss = 0;
        {
            auto timer = std::make_unique<dy::Timer>("train took");

            for (auto&& batch : train_iter) {
                dy::ComputationGraph cg;
                auto loss = clf.batch_loss(cg, *batch);
                auto loss_val = dy::as_scalar(cg.incremental_forward(loss));
                total_loss += batch->size() * loss_val;
                cg.backward(loss);
                trainer.update();
                //clf.save("test.dy");
                //abort();
            }

        }

        float valid_acc;
        {
            auto timer = std::make_unique<dy::Timer>("valid took");
            valid_acc = validate(clf, valid_data);
        }

        auto training_loss = total_loss / n_train_sents;

        mlflow.log_metric("train_loss",   training_loss);
        mlflow.log_metric("valid_acc",    valid_acc);
        mlflow.log_metric("effective_lr", trainer.learning_rate);

        //crayon.log_metric("train_loss",   training_loss, 1 + it);
        //crayon.log_metric("valid_acc",    valid_acc, 1 + it);
        //crayon.log_metric("effective_lr", trainer.learning_rate, 1 + it);

        std::cout << "Completed epoch " << it << " training loss "
                  << training_loss << " valid accuracy "
                  << valid_acc << std::endl;

        if ((valid_acc + 0.0001) > last_valid_acc) {
            patience = 0;
        } else {
            trainer.learning_rate *= opts.decay;
            std::cout << "Decay to " << trainer.learning_rate << std::endl;
            patience += 1;
        }
        if (valid_acc > best_valid_acc) {
            best_valid_acc = valid_acc;
            mlflow.log_metric("best_valid_acc", best_valid_acc);

            std::ostringstream fn;
            fn << out_fn
               << std::internal << std::setfill('0') << std::fixed
               << std::setprecision(2) << std::setw(5) << valid_acc * 100.0
               << "_iter_" << std::setw(3) << it
               << "_" << mlflow.run_uuid
               << ".dy";
            clf.save(fn.str());
        }

        if (patience > opts.patience) {
            std::cout << opts.patience
            << " epochs without improvement, stopping." << std::endl;
            return;
        }
        last_valid_acc = valid_acc;
    }
}

int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);

    TrainOpts opts;
    opts.parse(argc, argv);
    std::cout << opts << std::endl;

    ListOpOpts list_opts;
    list_opts.parse(argc, argv);
    std::cout << list_opts << std::endl;

    SparseMAPOpts smap_opts;
    smap_opts.parse(argc, argv);

    bool is_sparsemap = (list_opts.get_tree() == ListOpOpts::Tree::MST
                         || list_opts.get_tree() == ListOpOpts::Tree::MST_LSTM
                         || list_opts.get_tree() == ListOpOpts::Tree::MST_CONSTR);

    if (is_sparsemap)
        std::cout << smap_opts << std::endl;

    if (opts.override_dy) {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    const auto train_fn = "data/listops/listops.train.txt";
    const auto valid_fn = "data/listops/listops.valid.txt";
    //const auto test_fn = "data/listops/listops.test.txt";

    const unsigned vocab_size = 19;
    const unsigned n_classes = 10;

    dy::ParameterCollection params;
    auto clf = ListOps(params,
                       list_opts.get_tree(),
                       list_opts.self_iter,
                       vocab_size,
                       list_opts.hidden_dim,
                       n_classes,
                       smap_opts.sm_opts);

    /* log mlflow run */

    MLFlowRun mlflow(opts.mlflow_exp);
    mlflow.set_tag("mlflow.runName", opts.mlflow_name);

    mlflow.log_parameter("lr", std::to_string(opts.lr));
    mlflow.log_parameter("decay", std::to_string(opts.decay));
    mlflow.log_parameter("patience", std::to_string(opts.patience));
    mlflow.log_parameter("max_iter", std::to_string(opts.max_iter));
    mlflow.log_parameter("saved_model", opts.saved_model);
    mlflow.log_parameter("batch_size", std::to_string(opts.batch_size));

    mlflow.log_parameter("hidden_dim", std::to_string(list_opts.hidden_dim));
    mlflow.log_parameter("dropout", std::to_string(list_opts.dropout));
    mlflow.log_parameter("self_iter", std::to_string(list_opts.self_iter));
    mlflow.log_parameter("tree", list_opts.tree_str);
    mlflow.log_parameter("fn_prefix", opts.save_prefix);

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
    fn << opts.save_prefix
       << opts.get_filename()
       << list_opts.get_filename();

    if (is_sparsemap)
        fn << smap_opts.get_filename();

    train(clf, opts, fn.str(), train_fn, valid_fn, mlflow);

    /*
    if (opts.test)
        test(clf, opts, valid_fn, test_fn);
    else
        train(clf, opts, fn.str(), train_fn, valid_fn, mlflow);
        */
}
