// Copyright 2025 Francesco Cavaliere
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OR_TOOLS_ALGORITHMS_SET_COVER_CORE_MODEL_H_
#define OR_TOOLS_ALGORITHMS_SET_COVER_CORE_MODEL_H_

#include "ortools/algorithms/set_cover/reduced_costs.h"
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif  // defined(_MSC_VER)

#include <tuple>

#include "../set_cover_model.h"

namespace operations_research::scp {

// CoreModel extends SetCoverModel to work on a subset of the model, updating it
// periodically using dual/Lagrangian multipliers.
//
// For very large SCP instances with millions of columns (subsets), limiting the
// algorithms to a subset of columns (called the core model) and refining it
// iteratively can achieve great speedups.
//
// This interface also enables column-generation based approaches, where the
// full set of columns is represented only implicitly by solving the
// "column-generation" problems that find the most promising columns to add to
// the core model. This approach is useful when the number of columns is
// prohibitively large.
// Note: Column generation methods are problem-specific, so they are not
// included in this interface.
// TODO(c4v4): provided an example of column generation.
class CoreModel {
 public:
  CoreModel() = default;
  CoreModel(const Model& core_model) : core_model_(core_model) {}
  CoreModel(Model&& core_model) : core_model_(std::move(core_model)) {}
  virtual ~CoreModel() = default;

  Model& core_model() { return core_model_; }
  const Model& core_model() const { return core_model_; }

  Cost lower_bound() const { return lower_bound_; }
  const ElementCostVector& multipliers() const { return multipliers_; }
  void UpdateLbAndMultipliers(Cost lower_bound,
                              ElementCostVector& multipliers) {
    if (lower_bound < lower_bound_) {
      lower_bound_ = lower_bound;
      multipliers_ = multipliers;
    }
  }

  // Invoked at every subgradient iteration, provides a customization point to
  // update the core model.
  virtual Cost UpdateCoreModel(Cost upper_bound,
                               ElementCostVector& multipliers) = 0;

 private:
  Model core_model_;
  // The best lower bound and multipliers are stored here because they become
  // invalid when core_model_ is updated. All information related to the core
  // model should be stored within the core model itself, so it can be updated
  // or reset as needed.
  Cost lower_bound_;
  ElementCostVector multipliers_;
};

// IdentityModel is a trivial implementation of CoreModel that does not update.
// The full model is the core model.
class IdentityModel final : public CoreModel {
 public:
  template <typename... Args>
  IdentityModel(Args&&... args) : CoreModel(std::forward<Args>(args)...) {}

  Cost UpdateCoreModel(Cost upper_bound,
                       ElementCostVector& multipliers) override {
    return this->lower_bound();
  }
};

// CoreFromFullModel is a CoreModel implementation that updates the core model
// by selecting the most promising columns from the full model. This specific
// implementation follows the approach described in:
// [1] Caprara, Alberto, Matteo Fischetti, and Paolo Toth. 1999. “A Heuristic
// Method for the Set Covering Problem.” Operations Research 47 (5): 730–43.
// https://www.jstor.org/stable/223097
class CoreFromFullModel final : public CoreModel {
  static constexpr size_t min_row_coverage = 5;

 public:
  // The core model is automatically constructed as in [1]
  CoreFromFullModel(const Model* full_model);

  // Externally provided initial core model
  CoreFromFullModel(const Model& core_model, Model* full_model)
      : CoreModel(core_model), full_model_(full_model) {}
  CoreFromFullModel(Model&& core_model, Model* full_model)
      : CoreModel(std::move(core_model)), full_model_(full_model) {}

  const Model& full_model() { return *full_model_; }
  const Model& full_model() const { return *full_model_; }

  void BuildFirstCoreModel(const Model& full_model);
  Cost UpdateCoreModel(Cost upper_bound,
                       ElementCostVector& multipliers) override;

 private:
  const Model* full_model_;
  SubsetMapVector columns_map_;
  ReducedCosts reduced_costs_;
  // Managing the pricing period
  size_t period;
  size_t countdown;
  size_t max_countdown;
};

}  // namespace operations_research::scp

#endif  // OR_TOOLS_ALGORITHMS_SET_COVER_CORE_MODEL_H_
