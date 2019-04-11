#include <dynet/io.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>

#include <vector>
#include <string>
#include <fstream>

#include "utils.h"
#include "builders/bilstm.h"

namespace dy = dynet;

struct BaseModel
{
    dy::ParameterCollection p;

    explicit
    BaseModel(dy::ParameterCollection p) : p(p) {};

    void save(const std::string filename)
    {
        dy::TextFileSaver s(filename);
        s.save(p);
    }

    void load(const std::string filename)
    { dy::TextFileLoader l(filename);
        l.populate(p);
    }
};


struct BaseEmbedBiLSTMModel : BaseModel
{
    dy::LookupParameter p_emb;

    //std::unique_ptr<BiLSTMBuilder> bilstm;
    BiLSTMSettings bilstm_settings;
    BiLSTMBuilder bilstm;

    unsigned vocab_size_;
    unsigned embed_dim_;
    unsigned hidden_dim_;
    unsigned stacks_;
    bool update_embed_;
    float dropout_;

    bool training_ = false;

    explicit
    BaseEmbedBiLSTMModel(
        dy::ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned stacks,
        bool update_embed,
        float dropout,
        std::string model_name="model")
    : BaseModel(params.add_subcollection(model_name))
    , bilstm_settings{stacks, /*layers=*/1, hidden_dim / 2}
    , bilstm(p, bilstm_settings, embed_dim)
    , vocab_size_(vocab_size)
    , embed_dim_(embed_dim)
    , hidden_dim_(hidden_dim)
    , stacks_(stacks)
    , update_embed_(update_embed)
    , dropout_(dropout)
    {
        if (update_embed)
            p_emb = p.add_lookup_parameters(vocab_size_, {embed_dim_});
        else
            p_emb = params.add_lookup_parameters(vocab_size_, {embed_dim_});

        assert(hidden_dim_ % 2 == 0);
    }

    void
    set_train_time()
    {
        training_ = true;
        bilstm.set_dropout(dropout_);
    }

    void
    set_test_time()
    {
        training_ = false;
        bilstm.disable_dropout();
    }

    void
    load_embeddings(
        const std::string filename)
    {
        std::cerr << "~ ~ loading embeddings... ~ ~ ";
        std::ifstream in(filename);
        assert(in);
        std::string line;
        vector<float> embed(embed_dim_);
        unsigned ix = 0;

        while (getline(in, line))
        {
            std::istringstream lin(line);

            for (unsigned i = 0; i < embed_dim_; ++i)
                lin >> embed[i];

            p_emb.initialize(ix, embed);
            ix += 1;
        }
        std::cerr << "done." << std::endl;
    }

    std::vector<dy::Expression>
    embed_ctx_sent(
        dy::ComputationGraph&cg,
        const Sentence& sent)
    {
        std::vector<dy::Expression> embeds(sent.size());

        auto sent_sz = sent.size();

        for (size_t i = 0; i < sent_sz; ++i)
        {
            auto w = sent.word_ixs[i];
            embeds[i] = update_embed_ ? dy::lookup(cg, p_emb, w)
                                      : dy::const_lookup(cg, p_emb, w);
        }

        auto enc = bilstm(embeds);

        return enc;
    }
};
