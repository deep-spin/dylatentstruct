#pragma once

// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.1.
//
// AD3 2.1 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.1 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.1.  If not, see <http://www.gnu.org/licenses/>.

#include <ostream>
#include "ad3/GenericFactor.h"


namespace AD3 {

class FactorTree : public GenericFactor
{
  public:
    FactorTree() {}
    virtual ~FactorTree() { ClearActiveSet(); }

    void RunCLE(const vector<double>& scores,
                vector<int>* heads,
                double* value);

    // Compute the score of a given assignment.
    // Note: additional_log_potentials is empty and is ignored.
    void Maximize(const vector<double>& variable_log_potentials,
                  const vector<double>&,
                  Configuration& configuration,
                  double* value)
    {
        vector<int>* heads = static_cast<vector<int>*>(configuration);
        RunCLE(variable_log_potentials, heads, value);
    }

    // Compute the score of a given assignment.
    // Note: additional_log_potentials is empty and is ignored.
    void Evaluate(const vector<double>& variable_log_potentials,
                  const vector<double>&,
                  const Configuration configuration,
                  double* value)
    {
        const vector<int>* heads =
          static_cast<const vector<int>*>(configuration);
        // Heads belong to {0,1,2,...}
        *value = 0.0;
        for (size_t m = 1; m < heads->size(); ++m) {
            int h = (*heads)[m];
            int ix = index_arcs_[h][m];
            *value += variable_log_potentials[ix];
        }
    }

    // Given a configuration with a probability (weight),
    // increment the vectors of variable and additional posteriors.
    // Note: additional_log_potentials is empty and is ignored.
    void UpdateMarginalsFromConfiguration(const Configuration& configuration,
                                          double weight,
                                          vector<double>* variable_posteriors,
                                          vector<double>*)
    {
        const vector<int>* heads =
          static_cast<const vector<int>*>(configuration);
        for (size_t m = 1; m < heads->size(); ++m) {
            int h = (*heads)[m];
            int ix = index_arcs_[h][m];
            (*variable_posteriors)[ix] += weight;
        }
    }

    // Count how many common values two configurations have.
    int CountCommonValues(const Configuration& configuration1,
                          const Configuration& configuration2)
    {
        const vector<int>* heads1 =
          static_cast<const vector<int>*>(configuration1);
        const vector<int>* heads2 =
          static_cast<const vector<int>*>(configuration2);
        int count = 0;
        for (size_t i = 1; i < heads1->size(); ++i) {
            if ((*heads1)[i] == (*heads2)[i]) {
                ++count;
            }
        }
        return count;
    }

    // Check if two configurations are the same.
    bool SameConfiguration(const Configuration& configuration1,
                           const Configuration& configuration2)
    {
        const vector<int>* heads1 =
          static_cast<const vector<int>*>(configuration1);
        const vector<int>* heads2 =
          static_cast<const vector<int>*>(configuration2);
        for (size_t i = 1; i < heads1->size(); ++i) {
            if ((*heads1)[i] != (*heads2)[i])
                return false;
        }
        return true;
    }

    // Delete configuration.
    void DeleteConfiguration(Configuration configuration)
    {
        vector<int>* heads = static_cast<vector<int>*>(configuration);
        delete heads;
    }

    // Create configuration.
    Configuration CreateConfiguration()
    {
        vector<int>* heads = new vector<int>(length_);
        return static_cast<Configuration>(heads);
    }

    void Initialize(int length, const vector<std::tuple<int, int>>& arcs)
    {
        length_ = length;
        index_arcs_.assign(length, vector<int>(length, -1));
        for (size_t k = 0; k < arcs.size(); ++k) {
            int h, m;
            std::tie(h, m) = arcs[k];
            index_arcs_[h][m] = k;
        }
    }

    virtual void
    PrintConfiguration(std::ostream& out, const Configuration y) override
    {
        vector<int>* heads = static_cast<vector<int>*>(y);
        for (auto && h : *heads)
            out << h << " ";
    }

  private:
    void RunChuLiuEdmondsIteration(vector<bool>* disabled,
                                   vector<vector<int>>* candidate_heads,
                                   vector<vector<double>>* candidate_scores,
                                   vector<int>* heads,
                                   double* value);

  protected:
    int length_; // Sentence length (including root symbol).
    vector<vector<int>> index_arcs_;
};

} // namespace AD3
