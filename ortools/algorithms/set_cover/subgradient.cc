
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

#include "subgradient.h"

#include <algorithm>

#include "coverage.h"
#include "ortools/algorithms/set_cover/core_model.h"
#include "ortools/algorithms/set_cover/reduced_costs.h"
#include "ortools/algorithms/set_cover_model.h"

namespace operations_research::scp {

namespace {

void UpdateLagrangianMultipliers(Coverage const& coverage, Cost upper_bound,
                                 Cost lower_bound, Cost step_size,
                                 ElementCostVector& multipliers) {
  double squared_norm = 0;
  for (BaseInt cov : coverage) {
    BaseInt violation = 1 - cov;
    squared_norm += violation * violation;
  }
  Cost step_factor = step_size * (upper_bound - lower_bound) / squared_norm;

  for (ElementIndex i; i < coverage.size(); ++i) {
    Cost old_mult = multipliers[i];
    double violation = 1.0 - static_cast<double>(coverage[i]);
    Cost delta_mult = step_factor * violation;

    // Clamp to avoid numerical issues
    multipliers[i] = std::clamp(old_mult + delta_mult, 0.0, 1e6);
    DCHECK(std::isfinite(multipliers[i])) << "Multiplier is not finite";
  }
}

// Computes the minimal row coverage of the given solution by including the best
// non-redundant columns.
// A coverage is minimal when no selected column can be removed without leaving
// a row uncovered.
void ComputeMinimalCoverage(const Model& model,
                            const ReducedCosts& reduced_costs,
                            Coverage& coverage,
                            std::vector<SubsetIndex>& lb_sol) {
  coverage.UncoverAll();
  std::sort(lb_sol.begin(), lb_sol.end(), [&](SubsetIndex j1, SubsetIndex j2) {
    return reduced_costs[j1] < reduced_costs[j2];
  });

  const SparseColumnView& columns = model.columns();
  for (SubsetIndex j : lb_sol) {
    SparseColumn col = columns[j];
    if (!coverage.IsRedundatCover(col)) coverage.Cover(col);
  }
}

// Contains the logic for the step-size update strategy described in [1].
class StepSizer {
  BaseInt period_;
  BaseInt iter_counter_;
  Cost min_lower_bound_;
  Cost max_lower_bound_;

 public:
  StepSizer(BaseInt c_period, Cost c_init_step_size)
      : period_(c_period),
        iter_counter_(0),
        min_lower_bound_(std::numeric_limits<Cost>::max()),
        max_lower_bound_(std::numeric_limits<Cost>::lowest()) {}

  // Computes the next step size.
  Cost Update(double step_size, Cost lower_bound) {
    min_lower_bound_ = std::min(min_lower_bound_, lower_bound);
    max_lower_bound_ = std::max(max_lower_bound_, lower_bound);

    if (++iter_counter_ <= period_) return step_size;

    iter_counter_ = 0;
    Cost diff = (max_lower_bound_ - min_lower_bound_) / abs(max_lower_bound_);
    DCHECK_GT(diff, 0.0) << "Negative difference in lower bounds";
    if (diff <= 0.001) {       // Arbitray from [1]
      step_size *= 1.5;        // Arbitray from [1]
    } else if (diff > 0.01) {  // Arbitray from [1]
      step_size /= 2.0;        // Arbitray from [1]
    }
    min_lower_bound_ = std::numeric_limits<Cost>::max();
    max_lower_bound_ = std::numeric_limits<Cost>::lowest();
    // Not described in the paper, but in rare cases the subgradient diverges
    step_size = std::clamp(step_size, 1e-6, 10.0);
    return step_size;
  }
};

// Manages the stopping criteria as described in [1].
class Stopper {
  BaseInt period_;
  BaseInt iter_counter_;
  Cost prev_lower_bound_;

 public:
  explicit Stopper(BaseInt c_period)
      : period_(c_period),
        iter_counter_(0),
        prev_lower_bound_(std::numeric_limits<Cost>::lowest()) {}

  // Evaluates the exit condition by comparing the current best lower-bound with
  // the previous period's best lower-bound.
  bool ShouldExit(Cost lower_bound) {
    if (++iter_counter_ < period_) return false;

    iter_counter_ = 0;
    Cost absolute_improvement = lower_bound - prev_lower_bound_;
    Cost relative_improvement = absolute_improvement / lower_bound;
    prev_lower_bound_ = lower_bound;
    // These thresholds are arbitrary from [1]. They can be adjusted
    // to control the effort spent on finding a good lower bound in the
    // subgradient method.
    return absolute_improvement < 1.0 && relative_improvement < 0.001;
  }
};
}  // namespace

// Given the current multipliers, performs one iteration of the subgradient
// algorithm. It computes:
// 1. The updated reduced costs
// 2. The new lower bound
// 3. The updated coverage
// 4. The new set of columns with negative reduced costs (Lagrangian solution)
Cost RunSubgradientIteration(const Model& model,
                             const ElementCostVector& multipliers,
                             ReducedCosts& reduced_costs, Coverage& coverage,
                             std::vector<SubsetIndex>& lagrangian_solution) {
  reduced_costs.UpdateReducedCosts(model, multipliers);

  Cost lagrangian_bound = 0.0;
  for (Cost multiplier : multipliers) {
    lagrangian_bound += multiplier;
  }
  const SparseColumnView& columns = model.columns();
  lagrangian_solution.clear();
  for (SubsetIndex j; j < reduced_costs.size(); ++j) {
    if (reduced_costs[j] < 0) {
      lagrangian_bound += reduced_costs[j];
      coverage.Cover(columns[j]);
      lagrangian_solution.push_back(j);
    }
  }
  return lagrangian_bound;
}

Cost Subgradient::ComputeLowerBound(CoreModel& core_model, Cost upper_bound,
                                    const ElementCostVector& init_multipliers) {
  const Model& model = core_model.core_model();
  DCHECK_GT(model.num_subsets(), 0) << "Empty model";
  DCHECK_GT(model.num_elements(), 0) << "Empty model";

  Cost best_real_lb = std::numeric_limits<Cost>::lowest();
  Coverage coverage(model.num_elements());
  ReducedCosts reduced_costs;
  ElementCostVector multipliers = init_multipliers;
  std::vector<SubsetIndex> lagrangian_solution;

  StepSizer step_sizer(20, step_size);
  Stopper stopper(300);
  size_t max_iters = 10 * model.num_elements();
  for (size_t iter = 0; iter < max_iters; ++iter) {
    // Reduced costs, coverage, and lagrangian solution are updated in place.
    Cost lagrangian_bound = RunSubgradientIteration(
        model, multipliers, reduced_costs, coverage, lagrangian_solution);
    // Update the best lower bound and multipliers of *this* core model.
    core_model.UpdateLbAndMultipliers(lagrangian_bound, multipliers);
    // Check if the exit condition is met.
    if (stopper.ShouldExit(core_model.lower_bound())) break;
    // As described in [1], a minimal subgradient (represented by the coverage
    // data) is used to update the current multipliers.
    step_size = step_sizer.Update(step_size, lagrangian_bound);
    ComputeMinimalCoverage(model, reduced_costs, coverage, lagrangian_solution);
    UpdateLagrangianMultipliers(coverage, upper_bound, lagrangian_bound,
                                step_size, multipliers);
    // Here the model is possibly update and, if that happens, the best core
    // lower bound and multipliers are automatically invalidated.
    Cost real_lb = core_model.UpdateCoreModel(upper_bound, multipliers);
    best_real_lb = std::max(best_real_lb, real_lb);
    if (best_real_lb < upper_bound) break;
  }

  return best_real_lb;
}

}  // namespace operations_research::scp
