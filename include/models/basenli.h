#pragma once

#include <dynet/devices.h>
#include <dynet/index-tensor.h>
#include <dynet/tensor.h>
#include <dynet/timing.h>
#include <dynet/training.h>

#include <string>

#include "args.h"
#include "basemodel.h"
#include "crayon.h"
#include "data.h"
#include "mlflow.h"

namespace dy = dynet;

struct BaseNLI : public BaseEmbedModel
{
    unsigned hidden_dim_;

    explicit BaseNLI(dy::ParameterCollection& params,
                     unsigned vocab_size,
                     unsigned embed_dim,
                     unsigned hidden_dim,
                     bool update_embed)
      : BaseEmbedModel(params,
                       vocab_size,
                       embed_dim,
                       update_embed,
                       "nliclf")
      , hidden_dim_{ hidden_dim }
    {}

    virtual int n_correct(dy::ComputationGraph& cg, const NLIBatch& batch)
    {
        set_test_time();
        auto out_v = predict_batch(cg, batch);
        auto out_b = dy::concatenate_to_batch(out_v);

        cg.incremental_forward(out_b);
        auto out = out_b.value();
        auto pred = dy::as_vector(dy::TensorTools::argmax(out));

        int n_correct = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            if (batch[i].target == pred[i])
                n_correct += 1;
        }
        return n_correct;
    }

    virtual dy::Expression batch_loss(dy::ComputationGraph& cg,
                                      const NLIBatch& batch)
    {
        set_train_time();
        auto out = predict_batch(cg, batch);

        std::vector<dy::Expression> losses;
        for (size_t i = 0; i < batch.size(); ++i) {
            auto loss = dy::pickneglogsoftmax(out[i], batch[i].target);
            losses.push_back(loss);
        }

        return dy::average(losses); // was: sum; this is better
    }

    virtual std::vector<dy::Expression> predict_batch(
      dy::ComputationGraph& cg,
      const NLIBatch& batch) = 0;
};

// shared training code
float
validate(std::unique_ptr<BaseNLI>& clf, const std::vector<NLIBatch>& data)
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
test(std::unique_ptr<BaseNLI>& clf,
     const TrainOpts& args,
     const std::string& valid_fn,
     const std::string& test_fn)
{
    clf->load(args.saved_model);

    std::ostringstream valid_print_fn(args.save_prefix);
    auto valid_data = read_batches<NLIPair>(valid_fn, args.batch_size);
    float acc = validate(clf, valid_data);
    std::cout << "Validation accuracy: " << acc << std::endl;
    auto test_data = read_batches<NLIPair>(test_fn, args.batch_size);
    acc = validate(clf, test_data);
    std::cout << "Test accuracy: " << acc << std::endl;
}

void
train(std::unique_ptr<BaseNLI>& clf,
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

    Crayon crayon(out_fn);

    for (unsigned it = 0; it < args.max_iter; ++it) {
        // shuffle the permutation vector
        std::shuffle(train_iter.begin(), train_iter.end(), *dy::rndeng);

        float total_loss = 0;
        {
            auto timer = std::make_unique<dy::Timer>("train took");

            for (auto&& batch : train_iter) {
                dy::ComputationGraph cg;
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

        crayon.log_metric("train_loss", training_loss, 1 + it);
        crayon.log_metric("valid_acc", valid_acc, 1 + it);
        crayon.log_metric("effective_lr", trainer.learning_rate, 1 + it);

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
