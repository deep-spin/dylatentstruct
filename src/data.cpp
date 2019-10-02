#include <sstream>

#include "data.h"

std::istream&
operator>>(std::istream& in, LabeledSentence& data)
{
    std::string target_buf, ixs_buf, heads_buf;
    std::getline(in, target_buf, '\t');
    std::getline(in, ixs_buf, '\t');
    std::getline(in, heads_buf);
    if (!in)  // failed
        return in;

    {
        std::stringstream target_ss(target_buf);
        target_ss >> data.target;
    }

    {
        std::stringstream ixs(ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.sentence.word_ixs.push_back(tmp);
    }

    {
        std::stringstream heads(heads_buf);
        int tmp;
        while(heads >> tmp)
            data.sentence.heads.push_back(tmp);
    }

    return in;
}

std::istream&
operator>>(std::istream& in, TaggedSentence& data)
{
    std::string sent_buf, ixs_buf, tags_buf, heads_buf;
    std::getline(in, sent_buf, '\t');
    std::getline(in, ixs_buf, '\t');
    std::getline(in, tags_buf, '\t');
    std::getline(in, heads_buf);

    if (!in)  // failed
        return in;

    {
        std::stringstream ixs(ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.sentence.word_ixs.push_back(tmp);
    }

    {
        std::stringstream heads(heads_buf);
        int tmp;
        while(heads >> tmp)
            data.sentence.heads.push_back(tmp);
    }

    {
        std::stringstream tags(tags_buf);
        int tmp;
        while(tags >> tmp)
            data.tags.push_back(tmp);
    }

    if (data.sentence.size() != data.tags.size()) {
        std::cerr << "Error: mismatched tag length in sentence " << sent_buf
                  << std::endl;
        std::abort();
    }

    return in;
}

std::istream&
operator>>(std::istream& in, NLIPair& data)
{
    std::string target_buf;
    std::string prem_ixs_buf, prem_heads_buf;
    std::string hypo_ixs_buf, hypo_heads_buf;
    std::getline(in, target_buf, '\t');
    std::getline(in, prem_ixs_buf, '\t');
    std::getline(in, prem_heads_buf, '\t');
    std::getline(in, hypo_ixs_buf, '\t');
    std::getline(in, hypo_heads_buf, '\n');

    if (!in)  // failed
        return in;

    {
        std::stringstream target_ss(target_buf);
        target_ss >> data.target;
    }

    {
        std::stringstream ixs(prem_ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.prem.word_ixs.push_back(tmp);
    }

    {
        std::stringstream heads(prem_heads_buf);
        int tmp;
        while(heads >> tmp)
            data.prem.heads.push_back(tmp);
    }

    {
        std::stringstream ixs(hypo_ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.hypo.word_ixs.push_back(tmp);
    }

    {
        std::stringstream heads(hypo_heads_buf);
        int tmp;
        while(heads >> tmp)
            data.hypo.heads.push_back(tmp);
    }

    return in;
}

std::istream&
operator>>(std::istream& in, MultiLabelInstance& data)
{
    std::string target_buf, feats_buf;
    std::getline(in, target_buf, '\t');
    std::getline(in, feats_buf, '\n');

    if (!in)  // failed
        return in;

    {
        std::stringstream ixs(target_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.labels.push_back(tmp);
    }

    {
        std::stringstream ixs(feats_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.features.push_back(tmp);
    }

    return in;
}
