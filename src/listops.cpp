#include "mlflow.h"
#include "utils.h"
#include "data.h"
#include "args.h"

#include "models/listops_model.h"

namespace dy = dynet;

int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);

    TrainOpts opts;
    opts.parse(argc, argv);
    std::cout << opts << std::endl;

    ListOpOpts list_opts;
    list_opts.parse(argc, argv);
    std::cout << list_opts << std::endl;

    if (opts.override_dy) {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }

    dy::initialize(dyparams);

    const auto train_fn = "data/listops/listops.train.txt";
    const auto valid_fn = "data/listops/listops.valid.txt";
    const auto test_fn = "data/listops/listops.test.txt";

    const unsigned vocab_size = 19;
    const unsigned n_classes = 10;

    dy::ParameterCollection params;
    std::unique_ptr<BaseSentClf> clf =
        std::make_unique<ListOps>(params,
                                  vocab_size,
                                  list_opts.hidden_dim,
                                  list_opts.hidden_dim,
                                  n_classes,
                                  /*lstm_stacks = */1,
                                  list_opts.dropout);

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
    mlflow.log_parameter("fn_prefix", opts.save_prefix);

    // tweak filename
    std::ostringstream fn;
    fn << opts.save_prefix
       << opts.get_filename()
       << list_opts.get_filename();

    if (opts.test)
        test(clf, opts, valid_fn, test_fn);
    else
        train(clf, opts, fn.str(), train_fn, valid_fn, mlflow);


}
