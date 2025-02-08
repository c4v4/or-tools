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

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif  // defined(_MSC_VER)

#include <tuple>

#include "../set_cover_model.h"

namespace operations_research::scp {

// TODO(c4v4): move SetCover* into scp namespace and rename them.
using Model = SetCoverModel;

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

  virtual std::tuple<Cost, bool> UpdateModelAndLowerBound(
      ElementCostVector& multipliers, Cost lower_bound, Cost upper_bound) = 0;

 private:
  Model core_model_;
};

// IdentityModel is a trivial implementation of CoreModel that does not update.
// The full model is the core model.
class IdentityModel final : public CoreModel {
 public:
  template <typename... Args>
  IdentityModel(Args&&... args) : CoreModel(std::forward<Args>(args)...) {}

  std::tuple<Cost, bool> UpdateModelAndLowerBound(
      ElementCostVector& multipliers, Cost lower_bound,
      Cost upper_bound) override {
    return std::make_tuple(lower_bound, false);
  }
};

// CoreFromFullModel is a CoreModel implementation that updates the core model
// by selecting the most promising columns from the full model. This specific
// implementation follows the approach described in:
// [1] Caprara, Alberto, Matteo Fischetti, and Paolo Toth. 1999. “A Heuristic
// Method for the Set Covering Problem.” Operations Research 47 (5): 730–43.
// https://www.jstor.org/stable/223097
class CoreFromFullModel final : public CoreModel {
 public:
  // The core model is automatically constructed as in [1]
  CoreFromFullModel(Model* full_model);

  // Externally provided initial core model
  CoreFromFullModel(const Model& core_model, Model* full_model)
      : CoreModel(core_model), full_model_(full_model) {}
  CoreFromFullModel(Model&& core_model, Model* full_model)
      : CoreModel(std::move(core_model)), full_model_(full_model) {}

  Model& full_model() { return *full_model_; }
  const Model& full_model() const { return *full_model_; }

  std::tuple<Cost, bool> UpdateModelAndLowerBound(
      ElementCostVector& multipliers, Cost lower_bound,
      Cost upper_bound) override;

 private:
  Model* full_model_;
  // TODO(c4v4): defines the core->full mappings
};

}  // namespace operations_research::scp

#endif  // OR_TOOLS_ALGORITHMS_SET_COVER_CORE_MODEL_H_
