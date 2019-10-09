#include <vector>

#include <ad3/FactorGraph.h>
#include "factors/FactorTreeTurbo.h"


int main(int argc, char** argv)
{
    auto fg = std::make_unique<AD3::FactorGraph>();
    std::vector<AD3::BinaryVariable*> vars;
    std::vector<std::tuple<int, int>> arcs;

    bool projective = true;
    //bool projective = false;
    auto sz = 1;

    std::vector<double> scores;
    std::vector<double> add;

    int k = 1;

    for (size_t m = 1; m < sz; ++m) {
        for (size_t h = 0; h < sz; ++h) {
            if (h != m) {
                arcs.push_back(std::make_tuple(h, m));
                auto var = fg->CreateBinaryVariable();
                vars.push_back(var);
                scores.push_back(0.5 + k);
                ++k;
            }
        }
    }

    for (auto & s : scores)
       std::cout << s << " ";
    std::cout << std::endl;

    auto* tree_factor = new AD3::FactorTreeTurbo;
    fg->DeclareFactor(static_cast<AD3::Factor*>(tree_factor), vars, /*pass_ownership=*/true);
    tree_factor->Initialize(projective, sz, arcs);
    auto cfg = tree_factor->CreateConfiguration();
    double value = 0;
    tree_factor->Maximize(scores, add, cfg, &value);

    vector<int>* heads = static_cast<vector<int>*>(cfg);
    for (auto i : *heads)
        std::cout << i << " ";
    std::cout << std::endl;

    return 0;
}


