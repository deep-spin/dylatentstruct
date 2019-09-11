#pragma once

#include <dynet/io.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>

#include <vector>
#include <string>
#include <fstream>

#include "data.h"
#include "utils.h"
#include "builders/bilstm.h"

namespace dy = dynet;

struct BaseModel
{
    explicit
    BaseModel(dy::ParameterCollection&& p) : p(p) {};

    virtual void
    set_train_time()
    {
        training_ = true;
    }

    virtual void
    set_test_time()
    {
        training_ = false;
    }


    void save(const std::string filename)
    {
        dy::TextFileSaver s(filename);
        s.save(p);
    }

    void load(const std::string filename)
    { dy::TextFileLoader l(filename);
        l.populate(p);
    }

    dy::ParameterCollection p;
    bool training_;
};


struct BaseEmbedModel : BaseModel
{
    unsigned vocab_size_;
    unsigned embed_dim_;
    bool update_embed_;
    dy::LookupParameter p_emb;

    explicit
    BaseEmbedModel(
        dy::ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        bool update_embed,
        std::string model_name="model")
    : BaseModel{params.add_subcollection(model_name)}
    , vocab_size_{vocab_size}
    , embed_dim_{embed_dim}
    , update_embed_{update_embed}
    {
        if (update_embed)
            p_emb = p.add_lookup_parameters(vocab_size_, {embed_dim_});
        else
            p_emb = params.add_lookup_parameters(vocab_size_, {embed_dim_});
    }

    void
    load_embeddings(
        const std::string filename,
        bool normalize=false)
    {
        std::cerr << "~ ~ loading embeddings... ~ ~ ";
        std::ifstream in(filename);
        assert(in);
        std::string line;
        std::vector<float> embed(embed_dim_);
        unsigned ix = 0;

        while (getline(in, line))
        {
            std::istringstream lin(line);

            for (unsigned i = 0; i < embed_dim_; ++i)
                lin >> embed[i];

            if (normalize)
                normalize_vector(embed);

            p_emb.initialize(ix, embed);
            ix += 1;
        }
        std::cerr << "done." << std::endl;
    }

    std::vector<dy::Expression>
    embed_sent(
        dy::ComputationGraph& cg,
        const Sentence& sent)
    {
        auto sent_sz = sent.size();
        std::vector<dy::Expression> embeds(sent_sz);
        for (size_t i = 0; i < sent_sz; ++i)
        {
            auto w = sent.word_ixs[i];
            embeds[i] = update_embed_ ? dy::lookup(cg, p_emb, w)
                                      : dy::const_lookup(cg, p_emb, w);
        }
        return embeds;
    }
};


struct BaseEmbedBiLSTMModel : BaseEmbedModel
{

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
    : BaseEmbedModel{params, vocab_size, embed_dim, update_embed, model_name}
    , bilstm_settings{stacks, /*layers=*/1, hidden_dim / 2}
    , bilstm(p, bilstm_settings, embed_dim)
    , hidden_dim_(hidden_dim)
    , stacks_(stacks)
    , dropout_(dropout)
    {
        assert(hidden_dim_ % 2 == 0);
    }

    virtual void
    set_train_time()
    override
    {
        training_ = true;
        BaseModel::set_train_time();
        bilstm.set_dropout(dropout_);
    }

    virtual void
    set_test_time()
    override
    {
        training_ = false;
        BaseModel::set_test_time();
        bilstm.disable_dropout();
    }

    std::vector<dy::Expression>
    embed_ctx_sent(
        dy::ComputationGraph& cg,
        const Sentence& sent)
    {
        auto embeds = embed_sent(cg, sent);
        auto enc = bilstm(embeds);
        return enc;
    }

    BiLSTMSettings bilstm_settings;
    BiLSTMBuilder bilstm;

    unsigned hidden_dim_;
    unsigned stacks_;
    float dropout_;
};
