#pragma once

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>

struct Sentence
{

    std::vector<unsigned> word_ixs;
    std::vector<int> heads;
    size_t size() const { return word_ixs.size(); }
};

struct LabeledSentence
{
    Sentence sentence;
    unsigned target;

    size_t size() const { return sentence.size(); }
};

struct TaggedSentence
{
    Sentence sentence;
    std::vector<int> tags;

    size_t size() const { return sentence.size(); }
};

struct NLIPair
{
    Sentence prem;
    Sentence hypo;
    unsigned target;

    size_t size()
    const
    { return prem.word_ixs.size() + hypo.word_ixs.size(); }
};

struct MultiLabelInstance
{
    std::vector<int> features;
    std::vector<int> labels;
    size_t size() const { return features.size(); }
};

typedef std::vector<LabeledSentence> SentBatch;
typedef std::vector<TaggedSentence> TaggedBatch;
typedef std::vector<NLIPair> NLIBatch;
typedef std::vector<MultiLabelInstance> MLBatch;

std::istream& operator>>(std::istream& in, LabeledSentence& data);
std::istream& operator>>(std::istream& in, TaggedSentence& data);
std::istream& operator>>(std::istream& in, NLIPair& data);
std::istream& operator>>(std::istream& in, MultiLabelInstance& data);


template<typename T>
std::vector<std::vector<T> >
read_batches(const std::string& filename, unsigned batch_size)
{
    std::vector<std::vector<T> > batches;

    std::ifstream in(filename);
    assert(in);

    std::string line;
    std::vector<T> curr_batch;

    while(in)
    {
        T s;
        in >> s;
        if (!in) break;

        if (curr_batch.size() == batch_size)
        {
            batches.push_back(curr_batch);
            curr_batch.clear();
        }
        curr_batch.push_back(s);
    }

    // leftover batch
    if (curr_batch.size() > 0)
        batches.push_back(curr_batch);

    // test
    unsigned total_samples = 0;
    unsigned total_words = 0;
    for (auto& batch : batches)
    {
        total_samples += batch.size();
        for (auto& s : batch)
            total_words += s.size();
    }
    std::cerr << batches.size() << " batches, "
              << total_samples << " samples, "
              << total_words << " words\n";

    return batches;
}
