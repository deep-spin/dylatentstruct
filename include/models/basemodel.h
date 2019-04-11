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

    void save(const std::string filename)
    {
        dy::TextFileSaver s(filename);
        s.save(p);
    }

    void load(const std::string filename)
    {
        dy::TextFileLoader l(filename);
        l.populate(p);
    }
};


struct BaseEmbedBiLSTMModel : BaseModel
{
    dy::LookupParameter p_emb;

    std::unique_ptr<BiLSTMBuilder> bilstm;

    unsigned vocab_size_;
    unsigned embed_dim_;
    unsigned hidden_dim_;
    unsigned stacks_;

    bool update_embed_;

    explicit
    BaseEmbedBiLSTMModel(
        dy::ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned stacks,
        bool update_embed,
        std::string model_name="model")
    : vocab_size_(vocab_size)
    , embed_dim_(embed_dim)
    , hidden_dim_(hidden_dim)
    , stacks_(stacks)
    , update_embed_(update_embed)
    {
        p = params.add_subcollection(model_name);

        if (update_embed)
            p_emb = p.add_lookup_parameters(vocab_size_, {embed_dim_});
        else
            p_emb = params.add_lookup_parameters(vocab_size_, {embed_dim_});

        assert(hidden_dim_ % 2 == 0);

        BiLSTMSettings bl_settings = {stacks_, /*layers=*/1, hidden_dim_ / 2};
        bilstm.reset(new BiLSTMBuilder(p, bl_settings, embed_dim_));
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

        auto enc = (*bilstm)(embeds);

        return enc;
    }
};
