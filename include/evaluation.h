#pragma once

#include <vector>
#include <ostream>

struct PRFResult
{
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> fscore;

    double average_fscore() {
        double total = 0;
        for (auto && f : fscore)
            total += f;
        return total / fscore.size();
    }
};

class ConfusionMatrix
{
    public:
    explicit ConfusionMatrix(int n_classes)
        : n_classes{ n_classes }
        , counts(n_classes, std::vector<int>(n_classes, 0))
    {};

    void insert(int y_true, int y_pred) {
        if (y_true >= 0) // negative indices mean mask
            counts.at(y_true).at(y_pred) += 1;
    }

    double accuracy() {
        int correct = 0;
        int total = 0;
        for(int y_true = 0; y_true < n_classes; ++y_true) {
            for(int y_pred = 0; y_pred < n_classes; ++y_pred) {
                total += counts.at(y_true).at(y_pred);
                if (y_true == y_pred)
                    correct += counts.at(y_true).at(y_pred);
            }
        }
        return static_cast<double>(correct) / static_cast<double>(total);
    }

    PRFResult precision_recall_f1() {

        auto out = PRFResult{};

        auto pred_cls = std::vector<int>(n_classes, 0);
        auto true_cls = std::vector<int>(n_classes, 0);

        for(int y_true = 0; y_true < n_classes; ++y_true) {
            for(int y_pred = 0; y_pred < n_classes; ++y_pred) {
                int count = counts.at(y_true).at(y_pred);
                pred_cls.at(y_pred) += count;
                true_cls.at(y_true) += count;
            }
        }

        for(int y = 0; y < n_classes; ++y) {
            double tp = counts.at(y).at(y);
            double tp_plus_fp = pred_cls.at(y);
            double tp_plus_fn = true_cls.at(y);
            double p = tp / (tp_plus_fp + 1e-30);
            double r = tp / (tp_plus_fn + 1e-30);
            double f = (2 * p * r) / (p + r + 1e-30);

            out.precision.push_back(p);
            out.recall.push_back(r);
            out.fscore.push_back(f);
        }

        return out;
    }

    void operator+=(const ConfusionMatrix& cm) {
        for(int y_true = 0; y_true < cm.n_classes; ++y_true) {
            for(int y_pred = 0; y_pred < cm.n_classes; ++y_pred) {
                int other = cm.counts.at(y_true).at(y_pred);
                counts.at(y_true).at(y_pred) += other;
            }
        }
    }

    private:
    int n_classes;
    std::vector<std::vector<int> > counts;

    friend std::ostream& operator<<(std::ostream& out,
                                    const ConfusionMatrix& cm);
    friend ConfusionMatrix operator+(const ConfusionMatrix& cm1,
                                     const ConfusionMatrix& cm2);

};


std::ostream& operator<<(std::ostream& out, const ConfusionMatrix& cm) {
    for(auto&& row : cm.counts) {
        for(auto&& val : row)
            out << val << '\t';
        out << '\n';
    }
    return out;
}
