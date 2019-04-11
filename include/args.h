#pragma once
#include <iostream>
#include <cassert>
#include <sstream>

struct BaseOpts
{
    virtual std::ostream& print(std::ostream& o) const = 0;
    virtual std::string get_filename() const = 0;
};

struct TrainOpts : BaseOpts
{
    unsigned max_iter = 20;
    unsigned batch_size = 16;
    unsigned patience = 5;
    float lr = 1;
    float decay = 0.9;

    std::string saved_model;
    std::string save_prefix = "./";
    bool test = false;
    bool override_dy = true;

    int mlflow_exp = -1;
    std::string mlflow_name = "";

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc)
        {
            std::string arg = argv[i];
            if (arg == "--test")
            {
                test = true;
                i += 1;
            }
            else if (arg == "--no-override-dy")
            {
                override_dy = false;
                i += 1;
            }
            else if (arg == "--lr")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> lr;
                i += 2;
            }
            else if (arg == "--decay")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> decay;
                i += 2;
            }
            else if (arg == "--patience")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> patience;
                i += 2;
            }
            else if (arg == "--save-prefix")
            {
                assert(i + 1 < argc);
                save_prefix = argv[i + 1];
                i += 2;
            }
            else if (arg == "--max-iter")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> max_iter;
                i += 2;
            }
            else if (arg == "--saved-model")
            {
                assert(i + 1 < argc);
                saved_model = argv[i + 1];
                i += 2;
            }
            else if (arg == "--batch-size")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> batch_size;
                i += 2;
            }
            else if (arg == "--mlflow-experiment")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> mlflow_exp;
                i += 2;
            }
            else if (arg == "--mlflow-name")
            {
                assert(i + 1 < argc);
                mlflow_name = argv[i + 1];
                i += 2;
            }
            else
            {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o)
    const
    override
    {
        o <<  (test ? "Test" : "Train") << " mode. "
          << "Arguments:\n"
          << " Save prefix: " << save_prefix     << '\n'
          << "   Max. iter: " << max_iter        << '\n'
          << "  Batch size: " << batch_size      << '\n'
          << "          LR: " << lr              << '\n'
          << "       Decay: " << decay           << '\n'
          << "  MLFlow exp: " << mlflow_exp      << '\n'
          << "  MLFlow run: " << mlflow_name     << '\n'
          << "  Model file: " << saved_model     << std::endl;

        return o;
    }

    virtual std::string get_filename()
    const
    override
    {
        std::ostringstream fn;
        fn << "bs_" << batch_size
           << "_lr_" << lr
           << "_decay_" << decay
           << "_";
        return fn.str();
    }

};


struct GCNOpts : BaseOpts
{
    unsigned lstm_layers = 1;
    unsigned gcn_layers = 1;
    double dropout = .1;

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc)
        {
            std::string arg = argv[i];
            if (arg == "--lstm-layers")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> lstm_layers;
                i += 2;
            }
            else if (arg == "--gcn-layers")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> gcn_layers;
                i += 2;
            }
            if (arg == "--drop")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> dropout;
                i += 2;
            }
            else
            {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o) const
    {
        o << "GCN settinsk\n";
        o << " LSTM layers: " << lstm_layers << '\n';
        o << "  GCN layers: " << gcn_layers << '\n';
        o << "     dropout: " << dropout << '\n';
        return o;
    }

    virtual std::string get_filename()
    const
    {
        std::ostringstream fn;
        fn << "_lstm_" << lstm_layers << "_gcn_" << gcn_layers;
        return fn.str();
    }
};

struct ClfOpts : BaseOpts
{
    std::string dataset;

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc)
        {
            std::string arg = argv[i];
            if (arg == "--dataset")
            {
                assert(i + 1 < argc);
                dataset = argv[i + 1];
                i += 2;
            }
            else
            {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o) const
    {
        o << "Classfication task\n";
        o << "     Dataset: " << dataset << '\n';
        return o;
    }

    virtual std::string get_filename()
    const
    {
        std::ostringstream fn;
        fn << dataset;
        return fn.str();
    }
};


std::ostream& operator << (std::ostream &o, const BaseOpts &opts)
{
    return opts.print(o);
}

