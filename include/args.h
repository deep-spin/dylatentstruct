#pragma once

#include <cassert>
#include <iostream>
#include <sstream>

#include "sparsemap.h"

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
    std::string mlflow_host = "localhost";

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--test") {
                test = true;
                i += 1;
            } else if (arg == "--no-override-dy") {
                override_dy = false;
                i += 1;
            } else if (arg == "--lr") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> lr;
                i += 2;
            } else if (arg == "--decay") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> decay;
                i += 2;
            } else if (arg == "--patience") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> patience;
                i += 2;
            } else if (arg == "--save-prefix") {
                assert(i + 1 < argc);
                save_prefix = argv[i + 1];
                i += 2;
            } else if (arg == "--max-iter") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> max_iter;
                i += 2;
            } else if (arg == "--saved-model") {
                assert(i + 1 < argc);
                saved_model = argv[i + 1];
                i += 2;
            } else if (arg == "--batch-size") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> batch_size;
                i += 2;
            } else if (arg == "--mlflow-experiment") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> mlflow_exp;
                i += 2;
            } else if (arg == "--mlflow-name") {
                assert(i + 1 < argc);
                mlflow_name = argv[i + 1];
                i += 2;
            } else if (arg == "--mlflow-host") {
                assert(i + 1 < argc);
                mlflow_host = argv[i + 1];
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o) const override
    {
        o << (test ? "Test" : "Train") << " mode. "
          << "Arguments:\n"
          << " Save prefix: " << save_prefix << '\n'
          << "   Max. iter: " << max_iter << '\n'
          << "  Batch size: " << batch_size << '\n'
          << "          LR: " << lr << '\n'
          << "       Decay: " << decay << '\n'
          << " MLFlow host: " << mlflow_host << '\n'
          << "  MLFlow exp: " << mlflow_exp << '\n'
          << "  MLFlow run: " << mlflow_name << '\n'
          << "  Model file: " << saved_model << std::endl;

        return o;
    }

    virtual std::string get_filename() const override
    {
        std::ostringstream fn;
        fn << "bs_" << batch_size << "_lr_" << lr << "_decay_" << decay << "_";
        return fn.str();
    }
};

struct GCNOpts : BaseOpts
{
    unsigned layers = 0;
    unsigned iter = 1;
    unsigned use_distance = false;

    float dropout = .1f;
    std::string tree_str = "gold";

    enum class Tree
    {
        FLAT,
        LTR,
        GOLD,
        MST,
        MST_CONSTR
    };

    Tree get_tree() const
    {
        if (tree_str == "flat")
            return Tree::FLAT;
        else if (tree_str == "ltr")
            return Tree::LTR;
        else if (tree_str == "gold")
            return Tree::GOLD;
        else if (tree_str == "mst")
            return Tree::MST;
        else if (tree_str == "mst-constr")
            return Tree::MST_CONSTR;
        else {
            std::cerr << "Invalid tree type." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    bool is_sparsemap()
    {
        auto tree = get_tree();
        return (tree == Tree::MST ||
                tree == Tree::MST_CONSTR);
    }

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--gcn-layers") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> layers;
                i += 2;
            } else if (arg == "--gcn-iter") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> iter;
                i += 2;
            } else if (arg == "--gcn-drop") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> dropout;
                i += 2;
            } else if (arg == "--tree") {
                assert(i + 1 < argc);
                tree_str = argv[i + 1];
                i += 2;
            } else if (arg == "--use-distance") {
                use_distance = true;
                i += 1;
            } else {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o) const
    {
        o << "GCN settins\n";
        o << "  GCN layers: " << layers << '\n';
        o << "    GCN iter: " << iter << '\n';
        o << " GCN dropout: " << dropout << '\n';
        o << "        tree: " << tree_str << '\n';
        o << "    use dist: " << use_distance << '\n';
        return o;
    }

    virtual std::string get_filename() const
    {
        std::ostringstream fn;
        if (layers > 0)
            fn << "_gcn_" << layers
               << "_iter_" << iter
               << "_gcndrop_" << dropout
               << "_strat_" << tree_str
               << "_usedist_" << use_distance;
        return fn.str();
    }
};

struct ClfOpts : BaseOpts
{
    std::string dataset;

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--dataset") {
                assert(i + 1 < argc);
                dataset = argv[i + 1];
                i += 2;
            } else {
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

    virtual std::string get_filename() const
    {
        std::ostringstream fn;
        fn << dataset;
        return fn.str();
    }
};

struct ListOpOpts : public BaseOpts
{
    size_t hidden_dim = 100;
    float dropout = 0.1;
    std::string tree_str = "gold";
    size_t self_iter = 5;

    enum class Tree
    {
        FLAT,
        LTR,
        GOLD,
        MST,
        MST_LSTM,
        MST_CONSTR
    };

    Tree get_tree() const
    {
        if (tree_str == "flat")
            return Tree::FLAT;
        else if (tree_str == "ltr")
            return Tree::LTR;
        else if (tree_str == "gold")
            return Tree::GOLD;
        else if (tree_str == "mst")
            return Tree::MST;
        else if (tree_str == "mst-lstm")
            return Tree::MST_LSTM;
        else if (tree_str == "mst-constr")
            return Tree::MST_CONSTR;
        else {
            std::cerr << "Invalid tree type." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--hidden-dim") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> hidden_dim;
                i += 2;
            } else if (arg == "--self-iter") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> self_iter;
                i += 2;
            } else if (arg == "--drop") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> dropout;
                i += 2;
            } else if (arg == "--tree") {
                assert(i + 1 < argc);
                tree_str = argv[i + 1];
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o) const
    {
        o << "ListOps task\n"
          << " hidden dim: " << hidden_dim << '\n'
          << "       tree: " << tree_str << '\n'
          << "   selfiter: " << self_iter << '\n'
          << "    dropout: " << dropout << '\n';
        return o;
    }

    virtual std::string get_filename() const
    {
        std::ostringstream fn;
        fn << "_dim" << hidden_dim
           << "_drop" << dropout
           << "_tree_" << tree_str
           << "_selfit" << self_iter
           << "_";
        return fn.str();
    }
};


struct AttnOpts : public BaseOpts
{
    enum class Attn
    {
        SOFTMAX,
        SPARSEMAX,
        HEAD,
        HEADMATCH,
        HEADHO,
        HEADMATCHHO
    };

    bool is_sparsemap()
    {
        auto attn = get_attn();
        return (attn == Attn::HEAD ||
                attn == Attn::HEADMATCH ||
                attn == Attn::HEADHO ||
                attn == Attn::HEADMATCHHO);
    }

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--attn") {
                assert(i + 1 < argc);
                attn_str = argv[i + 1];
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    Attn get_attn() const
    {
        if (attn_str == "softmax")
            return Attn::SOFTMAX;
        else if (attn_str == "sparsemax")
            return Attn::SPARSEMAX;
        else if (attn_str == "head")
            return Attn::HEAD;
        else if (attn_str == "headmatch")
            return Attn::HEADMATCH;
        else if (attn_str == "head-ho")
            return Attn::HEADHO;
        else if (attn_str == "headmatch-ho")
            return Attn::HEADMATCHHO;
        else {
            std::cerr << "Invalid attention type." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    virtual std::string get_filename() const override
    {
        std::ostringstream fn;
        fn << "_attn_" << attn_str;
        return fn.str();
    }

    virtual std::ostream& print(std::ostream& o) const override
    {
        o << " Attention settings\n"
          << "     Attn type: " << attn_str << '\n';
        return o;
    }

    std::string attn_str = "softmax";
};


struct DecompOpts : public BaseOpts
{

    std::string dataset;
    int lstm_layers = 0;
    bool update_embed = false;
    bool normalize_embed = false;
    float dropout = .0f;

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--dataset") {
                assert(i + 1 < argc);
                dataset = argv[i + 1];
                i += 2;
            } else if (arg == "--drop") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> dropout;
                i += 2;
            } else if (arg == "--update-embed") {
                update_embed = true;
                i += 1;
            } else if (arg == "--normalize-embed") {
                normalize_embed = true;
                i += 1;
            } else if (arg == "--lstm-layers") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> lstm_layers;
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    virtual std::string get_filename() const override
    {
        std::ostringstream fn;
        fn << "_Decomp_" << dataset
           << "_drop_" << dropout
           << "_lstm_" << lstm_layers
           << "_update_" << update_embed
           << "_nrm_" << normalize_embed;
        return fn.str();
    }

    virtual std::ostream& print(std::ostream& o) const override
    {
        o << " Decomp settings\n"
          << "     Dataset: " << dataset << '\n'
          << "     Dropout: " << dropout << '\n'
          << "     LSTM layers: " << lstm_layers << '\n'
          << "     Update emb: " << update_embed << '\n'
          << "     Normlz emb: " << normalize_embed << '\n';
        return o;
    }
};

struct ESIMOpts : public BaseOpts
{

    std::string dataset;
    float dropout = 0.5;
    int lstm_layers;

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--dataset") {
                assert(i + 1 < argc);
                dataset = argv[i + 1];
                i += 2;
            } else if (arg == "--drop") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> dropout;
                i += 2;
            } else if (arg == "--lstm-layers") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> lstm_layers;
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    virtual std::string get_filename() const override
    {
        std::ostringstream fn;
        fn << "_ESIM_" << dataset << "_drop_"
           << dropout;
        return fn.str();
    }

    virtual std::ostream& print(std::ostream& o) const override
    {
        o << " ESIM settings\n"
          << " LSTM layers: " << lstm_layers << '\n'
          << "     Dataset: " << dataset << '\n'
          << "     Dropout: " << dropout << '\n';
        return o;
    }
};

struct SparseMAPOpts : BaseOpts
{
    dynet::SparseMAPOpts sm_opts;

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "--sparsemap-max-iter") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> sm_opts.max_iter;
                i += 2;
            } else if (arg == "--sparsemap-eta") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> sm_opts.eta;
                i += 2;
            } else if (arg == "--sparsemap-adapt-eta") {
                sm_opts.adapt_eta = true;
                i += 1;
            } else if (arg == "--sparsemap-residual-thr") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> sm_opts.residual_thr;
                i += 2;
            } else if (arg == "--sparsemap-max-iter-bw") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> sm_opts.max_iter_backward;
                i += 2;
            } else if (arg == "--sparsemap-atol-thr-bw") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> sm_opts.atol_thr_backward;
                i += 2;
            } else if (arg == "--sparsemap-max-active-set-iter") {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> sm_opts.max_active_set_iter;
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    virtual std::string get_filename() const override
    {
        std::ostringstream fn;
        fn << "sparsemap_max_iter_" << sm_opts.max_iter << "_thr_"
           << sm_opts.residual_thr << "_bw_" << sm_opts.max_iter_backward
           << "_atol_" << sm_opts.atol_thr_backward
           << "_";
        return fn.str();
    }

    virtual std::ostream& print(std::ostream& o) const override
    {
        o << " SparseMAP settings\n"
          << "    Max iter: " << sm_opts.max_iter << '\n'
          << "Residual thr: " << sm_opts.residual_thr << '\n'
          << "         eta: " << sm_opts.eta << '\n'
          << "   Adapt eta: " << sm_opts.adapt_eta << '\n'
          << "BW: max iter: " << sm_opts.max_iter_backward << '\n'
          << "BW: atol thr: " << sm_opts.atol_thr_backward << '\n'
          << "Act.set iter: " << sm_opts.max_active_set_iter << '\n';
        return o;
    }
};

std::ostream&
operator<<(std::ostream& o, const BaseOpts& opts)
{
    return opts.print(o);
}

