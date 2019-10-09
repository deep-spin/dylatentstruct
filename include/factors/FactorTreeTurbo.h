#pragma once

// Copyright (c) 2012-2015 Andre Martins
// All Rights Reserved.
//
// This file is part of TurboParser 2.3.
//
// TurboParser 2.3 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TurboParser 2.3 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with TurboParser 2.3.  If not, see <http://www.gnu.org/licenses/>.


#include "DependencyDecoder.h"
#include "ad3/GenericFactor.h"

namespace AD3 {
class FactorTreeTurbo : public GenericFactor {
public:
  FactorTreeTurbo() {}
  virtual ~FactorTreeTurbo() {
    ClearActiveSet();
  }

  // Print as a string.
  void Print(ostream& stream) {
    stream << "ARBORESCENCE";
    Factor::Print(stream);
    stream << endl;
  }

  // Compute the score of a given assignment.
  // Note: additional_log_potentials is empty and is ignored.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    vector<int>* heads = static_cast<vector<int>*>(configuration);

    if (length_ == 1) {
        heads->at(0) = -1;
        return;
    }

    if (projective_) {
      decoder.RunEisner(length_, num_arcs_, index_arcs_, variable_log_potentials,
                          heads, value);
    } else {
      decoder.RunChuLiuEdmonds(length_, index_arcs_, variable_log_potentials,
                                 heads, value);
    }
  }

  // Compute the score of a given assignment.
  // Note: additional_log_potentials is empty and is ignored.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<int> *heads = static_cast<const vector<int>*>(configuration);
    // Heads belong to {0,1,2,...}
    *value = 0.0;
    for (int m = 1; m < heads->size(); ++m) {
      int h = (*heads)[m];
      int index = index_arcs_[h][m];
      *value += variable_log_potentials[index];
    }
  }

  // Given a configuration with a probability (weight),
  // increment the vectors of variable and additional posteriors.
  // Note: additional_log_potentials is empty and is ignored.
  void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    const vector<int> *heads = static_cast<const vector<int>*>(configuration);
    for (int m = 1; m < heads->size(); ++m) {
      int h = (*heads)[m];
      int index = index_arcs_[h][m];
      (*variable_posteriors)[index] += weight;
    }
  }

  // Count how many common values two configurations have.
  int CountCommonValues(const Configuration &configuration1,
                        const Configuration &configuration2) {
    const vector<int> *heads1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *heads2 = static_cast<const vector<int>*>(configuration2);
    int count = 0;
    for (int i = 1; i < heads1->size(); ++i) {
      if ((*heads1)[i] == (*heads2)[i]) {
        ++count;
      }
    }
    return count;
  }

  // Check if two configurations are the same.
  bool SameConfiguration(
    const Configuration &configuration1,
    const Configuration &configuration2) {
    const vector<int> *heads1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *heads2 = static_cast<const vector<int>*>(configuration2);
    for (int i = 1; i < heads1->size(); ++i) {
      if ((*heads1)[i] != (*heads2)[i]) return false;
    }
    return true;
  }

  // Delete configuration.
  void DeleteConfiguration(
    Configuration configuration) {
    vector<int> *heads = static_cast<vector<int>*>(configuration);
    delete heads;
  }

  // Create configuration.
  Configuration CreateConfiguration() {
    vector<int>* heads = new vector<int>(length_);
    return static_cast<Configuration>(heads);
  }

public:
  void Initialize(bool projective, int length,
                  const vector<std::tuple<int, int>>& arcs) {
    projective_ = projective;
    length_ = length;
    num_arcs_ = arcs.size();
    index_arcs_.assign(length, vector<int>(length, -1));
    for (int k = 0; k < arcs.size(); ++k) {
        int h, m;
        std::tie(h, m) = arcs[k];
        index_arcs_[h][m] = k;
    }
  }

private:
  bool projective_; // If true, assume projective trees.
  int length_; // Sentence length (including root symbol).
  int num_arcs_;
  vector<vector<int> > index_arcs_;
  DependencyDecoder decoder;
};
} // namespace AD3
