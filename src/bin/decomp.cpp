#include <algorithm>
#include <iostream>

#include "args.h"
#include "data.h"
#include "mlflow.h"
#include "models/basemodel.h"
#include "models/decomp.h"
#include "utils.h"
#include <dynet/timing.h>
#include <dynet/training.h>

namespace dy = dynet;

using std::cout;
using std::endl;
using std::vector;

float
validate(std::unique_ptr<DecompAttn>& clf, const std::vector<NLIBatch>& data)
{
    int n_correct = 0;
    int n_total = 0;
    for (auto&& valid_batch : data) {
        dy::ComputationGraph cg;
        n_correct += clf->n_correct(cg, valid_batch);
        n_total += valid_batch.size();
    }

    return float(n_correct) / n_total;
}

void
test(
    std::unique_ptr<DecompAttn>& clf,
    const TrainOpts& opts,
    const std::string& valid_fn,
    const std::string& test_fn)
{
    clf->load(opts.saved_model);

    clf->attn->set_print(opts.saved_model + ".valid-attn.txt");
    auto valid_data = read_batches<NLIPair>(valid_fn, opts.batch_size);
    auto valid_acc = validate(clf, valid_data);
    cout << "Valid accuracy: " << valid_acc << endl;

    clf->attn->set_print(opts.saved_model + ".test-attn.txt");
    auto test_data = read_batches<NLIPair>(test_fn, opts.batch_size);
    auto test_acc = validate(clf, test_data);
    cout << " Test accuracy: " << test_acc << endl;
}

void
train(std::unique_ptr<DecompAttn>& clf,
      const TrainOpts& args,
      const std::string& out_fn,
      const std::string& train_fn,
      const std::string& valid_fn,
      const MLFlowRun& mlflow)
{
    auto train_data = read_batches<NLIPair>(train_fn, args.batch_size);
    auto valid_data = read_batches<NLIPair>(valid_fn, args.batch_size);
    auto n_batches = train_data.size();

    unsigned n_train_sents = 0;
    for (auto&& batch : train_data)
        n_train_sents += batch.size();

    std::cout << "Training on " << n_train_sents << " sentences." << std::endl;

    // make an identity permutation vector of pointers into the batches
    std::vector<std::vector<NLIBatch>::iterator> train_iter(n_batches);
    std::iota(train_iter.begin(), train_iter.end(), train_data.begin());

    dy::AdamTrainer trainer(clf->p, args.lr);

    size_t patience = 0;
    float best_valid_acc = 0;
    float last_valid_acc = 0;

    for (unsigned it = 0; it < args.max_iter; ++it) {
        // shuffle the permutation vector
        std::shuffle(train_iter.begin(), train_iter.end(), *dy::rndeng);

        float total_loss = 0;
        {
            auto timer = std::make_unique<dy::Timer>("train took");

            for (auto&& batch : train_iter) {
                dy::ComputationGraph cg;
                //cg.set_immediate_compute(true);
                //cg.set_check_validity(true);
                auto loss = clf->batch_loss(cg, *batch);
                auto loss_val = dy::as_scalar(cg.incremental_forward(loss));
                total_loss += batch->size() * loss_val;
                cg.backward(loss);
                trainer.update();
            }
        }

        float valid_acc;
        {
            auto timer = std::make_unique<dy::Timer>("valid took");
            valid_acc = validate(clf, valid_data);
        }

        auto training_loss = total_loss / n_train_sents;

        mlflow.log_metric("train_loss", training_loss);
        mlflow.log_metric("valid_acc", valid_acc);
        mlflow.log_metric("effective_lr", trainer.learning_rate);

        std::cout << "Completed epoch " << it << " training loss "
                  << training_loss << " valid accuracy " << valid_acc
                  << std::endl;

        if (valid_acc > last_valid_acc) {
            patience = 0;
        } else {
            trainer.learning_rate *= args.decay;
            std::cout << "Decay to " << trainer.learning_rate << std::endl;
            patience += 1;
        }
        if (valid_acc > best_valid_acc) {
            best_valid_acc = valid_acc;
            mlflow.log_metric("best_valid_acc", best_valid_acc);

            std::ostringstream fn;
            fn << out_fn << std::internal << std::setfill('0') << std::fixed
               << std::setprecision(2) << std::setw(5) << valid_acc * 100.0
               << "_iter_" << std::setw(3) << it << "_" << mlflow.run_uuid
               << ".dy";
            clf->save(fn.str());
        }

        if (patience > args.patience) {
            std::cout << args.patience
                      << " epochs without improvement, stopping." << std::endl;
            return;
        }
        last_valid_acc = valid_acc;
    }
}

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
    // train_fn << "data/nli/" << decomp_opts.dataset << ".train-tiny.txt";
    valid_fn << "data/nli/" << decomp_opts.dataset << ".valid.txt";
    // valid_fn << "data/nli/" << decomp_opts.dataset << ".valid-tiny.txt";
    test_fn << "data/nli/" << decomp_opts.dataset << ".test.txt";
    embed_fn << "data/nli/" << decomp_opts.dataset << ".embed";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    unsigned n_classes = line_count(class_fn.str());
    cout << "n_classes: " << n_classes << endl;

    dy::ParameterCollection params;
    /* log mlflow run */
    MLFlowRun mlflow(opts.mlflow_exp, opts.mlflow_host);
    mlflow.set_tag("mlflow.runName", opts.mlflow_name);

    mlflow.log_parameter("dataset", decomp_opts.dataset);

    mlflow.log_parameter("update_embed",
                         decomp_opts.update_embed ? "true" : "false");
    mlflow.log_parameter("normalize_embed",
                         decomp_opts.normalize_embed ? "true" : "false");

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
    fn << opts.save_prefix << "_" << opts.get_filename() << "_"
       << decomp_opts.get_filename() << "_";

    if (is_sparsemap)
        fn << smap_opts.get_filename();

    auto clf = std::make_unique<DecompAttn>(params,
                                            vocab_size,
                                            EMBED_DIM,
                                            opts.dim,
                                            n_classes,
                                            attn_opts.get_attn(),
                                            smap_opts.sm_opts,
                                            opts.dropout,
                                            decomp_opts.update_embed);

    clf->load_embeddings(embed_fn.str(), decomp_opts.normalize_embed);

    if (opts.test)
        test(clf, opts, valid_fn.str(), test_fn.str());
    else
        train(clf, opts, fn.str(), train_fn.str(), valid_fn.str(), mlflow);
    return 0;
}
