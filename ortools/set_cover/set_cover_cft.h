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

#ifndef CAV_OR_TOOLS_ORTOOLS_SET_COVER_SET_COVER_CFT_H
#define CAV_OR_TOOLS_ORTOOLS_SET_COVER_SET_COVER_CFT_H

#include <absl/base/internal/pretty_function.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ortools/algorithms/set_cover_model.h"
#include "ortools/base/accurate_sum.h"
#include "ortools/base/strong_vector.h"

namespace operations_research::scp {

////////////////////////////////////////////////////////////////////////
////////////////////////// COMMON DEFINITIONS //////////////////////////
////////////////////////////////////////////////////////////////////////

// Mappings to translate btween models with different indices.
using SubsetMapVector = util_intops::StrongVector<SubsetIndex, SubsetIndex>;
using ElementMapVector = util_intops::StrongVector<ElementIndex, ElementIndex>;
using Model = SetCoverModel;

class Solution {
 public:
  double cost() const { return cost_; }
  const std::vector<SubsetIndex>& subsets() const { return subsets_; }
  std::vector<SubsetIndex>& subsets() { return subsets_; }
  void AddSubset(SubsetIndex subset, Cost cost) {
    subsets_.push_back(subset);
    cost_ += cost;
  }
  bool Empty() const { return subsets_.empty(); }
  void Clear() {
    cost_ = 0.0;
    subsets_.clear();
  }

 private:
  Cost cost_;
  std::vector<SubsetIndex> subsets_;
};

class DualState {
 public:
  DualState(const Model& model);
  Cost lower_bound() const { return lower_bound_.Value(); }
  const ElementCostVector& multipliers() const { return multipliers_; }
  const SubsetCostVector& reduced_costs() const { return reduced_costs_; }
  void UpdateMultipliers(const Model& model,
                         const ElementCostVector& multipliers_delta);
  void SetMultipliers(const Model& model,
                      const ElementCostVector& multipliers_delta);

 private:
  template <typename Op>
  void DualUpdate(const Model& model, Op multiplier_generator);

 private:
  AccurateSum<Cost> lower_bound_;
  ElementCostVector multipliers_;
  SubsetCostVector reduced_costs_;
};

struct PrimalDualState {
  Solution solution;
  DualState dual_state;
};

class CoreModel : public Model {
 public:
  template <typename... Args>
  CoreModel(Args&&... args) : Model(std::forward<Args>(args)...) {}
  virtual bool UpdateCore(PrimalDualState& state) = 0;
  virtual ~CoreModel() = default;
};

// In the narrow scope of the CFT subgradient, there are often divisions
// between non-negative quantities (e.g., to compute a relative gap). In these
// specific cases, the denominator should always be greater than the
// numerator. This function checks that.
inline Cost DivideIfGE0(Cost numerator, Cost denominator) {
  DCHECK_GE(numerator, .0);
  if (numerator < 1e-6) {
    return 0.0;
  }
  return numerator / denominator;
}

absl::Status ValidateModel(const Model& model);
absl::Status ValidateFeasibleSolution(const Model& model,
                                      const Solution& solution,
                                      Cost tolerance = 1e-6);

///////////////////////////////////////////////////////////////////////
///////////////////////////// SUBGRADIENT /////////////////////////////
///////////////////////////////////////////////////////////////////////

struct SubgradientContext {
  const Model& model;
  const DualState& current_dual_state;
  const DualState& best_dual_state;
  const Solution& best_solution;
  const ElementCostVector& subgradient;
};

class SubgradientCBs {
 public:
  virtual bool ExitCondition(const SubgradientContext&) = 0;
  virtual void RunHeuristic(const SubgradientContext&, Solution&) = 0;
  virtual void ComputeMultipliersDelta(const SubgradientContext&,
                                       ElementCostVector& delta_mults) = 0;
  virtual bool UpdateCoreModel(CoreModel&, PrimalDualState&) = 0;
  virtual ~SubgradientCBs() = default;
};

class BoundCBs : public SubgradientCBs {
 public:
  static constexpr Cost kTol = 1e-6;

  BoundCBs(const Model& model);
  Cost step_size() const { return step_size_; }
  bool ExitCondition(const SubgradientContext& context) override;
  void ComputeMultipliersDelta(const SubgradientContext& context,
                               ElementCostVector& delta_mults) override;
  void RunHeuristic(const SubgradientContext& context,
                    Solution& solution) override {
    solution.Clear();
  }
  bool UpdateCoreModel(CoreModel& core_model,
                       PrimalDualState& best_state) override;

 private:
  void MakeMinimalCoverageSubgradient(const SubgradientContext& context,
                                      ElementCostVector& subgradient);

 private:
  Cost squared_norm_;
  ElementCostVector direction_;  // stabilized subgradient
  std::vector<SubsetIndex> lagrangian_solution_;

  // Stopping condition
  Cost prev_best_lb_;
  BaseInt max_iter_countdown_;
  BaseInt exit_test_countdown_;
  BaseInt exit_test_period_;

  // Step size
  void UpdateStepSize(Cost lower_bound);
  Cost step_size_;
  Cost last_min_lb_seen_;
  Cost last_max_lb_seen_;
  BaseInt step_size_update_countdown_;
  BaseInt step_size_update_period_;
};

absl::Status SubgradientOptimization(CoreModel& core_model, SubgradientCBs& cbs,
                                     PrimalDualState& best_state);

///////////////////////////////////////////////////////////////////////
//////////////////////// FULL TO CORE PRICING /////////////////////////
///////////////////////////////////////////////////////////////////////

// TODO(cava): implement core-model/full-model like in [1]
class FullToCoreModel : public CoreModel {
  struct UpdateTrigger {
    BaseInt countdown;
    BaseInt period;
    BaseInt max_period;
  };

 public:
  FullToCoreModel(Model&& full_model);

  template <typename... Args>
  FullToCoreModel(Args&&... args)
      : FullToCoreModel(Model(std::forward<Args>(args)...)) {}

  bool UpdateCore(PrimalDualState& core_state) override;

 private:
  void UpdatePricingPeriod(const DualState& full_dual_state,
                           const PrimalDualState& core_state);

  SubsetMapVector columns_map_;
  Model full_model_;
  DualState full_dual_state_;

  BaseInt update_countdown_;
  BaseInt update_period_;
  BaseInt update_max_period_;
};

////////////////////////////////////////////////////////////////////////
/////////////////////// MULTIPLIERS BASED GREEDY ///////////////////////
////////////////////////////////////////////////////////////////////////

void CompleteGreedySolution(const Model& model, Solution& solution);
void CompleteGreedySolution(const Model& model, const DualState& dual_state,
                            Cost cost_cutoff, BaseInt size_cutoff,
                            Solution& solution);

///////////////////////////////////////////////////////////////////////
//////////////////////// THREE PHASE ALGORITHM ////////////////////////
///////////////////////////////////////////////////////////////////////

class HeuristicCBs : public SubgradientCBs {
 public:
  HeuristicCBs() : step_size_(0.1), countdown_(250) {};
  void set_step_size(Cost step_size) { step_size_ = step_size; }
  bool ExitCondition(const SubgradientContext& context) override {
    return --countdown_ <= 0;
  }
  void RunHeuristic(const SubgradientContext& context,
                    Solution& solution) override;
  void ComputeMultipliersDelta(const SubgradientContext& context,
                               ElementCostVector& delta_mults) override;
  bool UpdateCoreModel(CoreModel& model, PrimalDualState& state) override {
    return false;
  }

 private:
  Cost step_size_;
  BaseInt countdown_;
};

absl::StatusOr<PrimalDualState> RunThreePhase(
    CoreModel& model, const Solution& init_solution = {});

}  // namespace operations_research::scp

#endif /* CAV_OR_TOOLS_ORTOOLS_SET_COVER_SET_COVER_CFT_H */
