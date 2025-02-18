
#include "lagrangian.h"

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_set.h>
#include <absl/random/distributions.h>

#include "ortools/algorithms/set_cover_model.h"
#include "ortools/base/iterator_adaptors.h"
#include "ortools/base/stl_util.h"

namespace operations_research::scp {

namespace {

// Returns true if uncovering the elements of the given columns does not leave
// any element completly uncovered.
bool IsRedundantDeselection(const SparseColumn& col,
                            const ElementToIntVector& subgradient) {
  for (ElementIndex i : col) {
    DCHECK(subgradient[i] <= 0) << "Column not selected";
    if (subgradient[i] == 0) {
      return false;
    }
  }
  return true;
}

void MakeMinimalSubgradient(const Model& model,
                            const SubsetCostVector& reduced_costs,
                            ElementToIntVector& subgradient,
                            std::vector<SubsetIndex>& lagrangian_solution,
                            std::vector<SubsetIndex>& excluded_columns) {
  DCHECK_EQ(reduced_costs.size(), model.num_subsets());
  DCHECK_EQ(subgradient.size(), model.num_elements());

  absl::c_sort(lagrangian_solution, [&](SubsetIndex j1, SubsetIndex j2) {
    return reduced_costs[j1] < reduced_costs[j2];
  });
  const SparseColumnView& columns = model.columns();
  for (SubsetIndex j : gtl::reversed_view(lagrangian_solution)) {
    if (IsRedundantDeselection(columns[j], subgradient)) {
      for (ElementIndex i : columns[j]) {
        ++subgradient[i];
      }
      excluded_columns.push_back(j);
    }
  }
}

// Restore non-minimal coverage
void UndoMinimalSubgradient(const Model& model, ElementToIntVector& subgradient,
                            std::vector<SubsetIndex>& excluded_columns) {
  const SparseColumnView& columns = model.columns();
  for (SubsetIndex j : excluded_columns) {
    for (ElementIndex i : columns[j]) {
      --subgradient[i];
    }
  }
  excluded_columns.clear();
}

// Computes the reduced costs and the lower bound from scratch given the
// multipliers.
Cost ComputeReducedCostAndLB(const Model& model,
                             const ElementCostVector& multipliers,
                             SubsetCostVector& reduced_costs) {
  DCHECK_EQ(multipliers.size(), model.num_elements());
  DCHECK_EQ(reduced_costs.size(), model.num_subsets());

  const SparseColumnView columns = model.columns();
  Cost lower_bound = absl::c_accumulate(multipliers, Cost{.0});
  for (SubsetIndex j : model.SubsetRange()) {
    reduced_costs[j] = model.subset_costs()[j];
    for (ElementIndex i : columns[j]) {
      reduced_costs[j] -= multipliers[i];
    }
    if (reduced_costs[j] < .0) {
      lower_bound += reduced_costs[j];
    }
  }
  return lower_bound;
}
}  // namespace

SubgradStepSizer::SubgradStepSizer(BaseInt period, double step_size)
    : step_size_(step_size),
      period_(period),
      countdown_(0),
      min_lower_bound_(std::numeric_limits<Cost>::max()),
      max_lower_bound_(std::numeric_limits<Cost>::lowest()) {}

double SubgradStepSizer::operator()(const DualState& dual_state) {
  Cost lower_bound = dual_state.lower_bound();
  min_lower_bound_ = std::min(min_lower_bound_, lower_bound);
  max_lower_bound_ = std::max(max_lower_bound_, lower_bound);

  if (--countdown_ > 0) return step_size_;
  countdown_ = period_;

  Cost diff = (max_lower_bound_ - min_lower_bound_) / abs(max_lower_bound_);
  DCHECK_GT(diff, .0) << "Negative difference in lower bounds";
  if (diff <= .001) {       //
    step_size_ *= 1.5;      // Arbitray
  } else if (diff > .01) {  // from [1]
    step_size_ /= 2.0;      //
  }
  min_lower_bound_ = std::numeric_limits<Cost>::max();
  max_lower_bound_ = std::numeric_limits<Cost>::lowest();
  // Not described in the paper, but in rare cases the subgradient diverges
  step_size_ = std::clamp(step_size_, 1e-6, 1.0);
  return step_size_;
}

SubgradUpdater::SubgradUpdater(BaseInt period, BaseInt max_period)
    : countdown(period), period(period), max_period(max_period) {}

bool SubgradUpdater::ShouldUpdate(Cost lower_bound, Cost real_lower_bound,
                                  Cost upper_bound) {
  if (--countdown > 0) {
    return false;
  }
  const Cost delta = (lower_bound - real_lower_bound) / upper_bound;
  if (delta <= 1e-6) {
    period = std::min(max_period, 10 * period);
  } else if (delta <= 0.02) {
    period = std::min(max_period, 5 * period);
  } else if (delta <= 0.2) {
    period = std::min(max_period, 2 * period);
  } else {
    period = 10;
  }
  countdown = period;
  return true;
}

SubgradStopper::SubgradStopper(BaseInt period)
    : countdown(period), period(period), prev_lower_bound(.0) {}

bool SubgradStopper::operator()(const DualState& dual_state) {
  if (--countdown > 0) {
    return false;
  }
  countdown = period;
  Cost lower_bound = dual_state.lower_bound();
  Cost absolute_improvement = lower_bound - prev_lower_bound;
  Cost relative_improvement = absolute_improvement / lower_bound;
  prev_lower_bound = lower_bound;
  // These thresholds are arbitrary from [1]. They can be adjusted
  // to control the effort spent on finding a good lower bound in the
  // subgradient method.
  return absolute_improvement < 1.0 && relative_improvement < .001;
}

void DualState::ModifyMultiplier(ElementIndex i, Cost delta_mult,
                                 FillOnlyColumnSet& affected_columns) {
  double old_multiplier = multipliers_[i];
  multipliers_[i] = std::clamp(multipliers_[i] + delta_mult, .0, 1e6);
  double real_delta_mult = multipliers_[i] - old_multiplier;
  const SparseRowView& rows = model_.rows();
  for (SubsetIndex j : rows[i]) {
    affected_columns.Insert(j);
    Cost old_red_cost = reduced_costs_[j];
    reduced_costs_[j] -= real_delta_mult;
    Cost delta_red_cost = std::min(reduced_costs_[j], Cost{.0}) -
                          std::min(old_red_cost, Cost{.0});
    lower_bound_ += delta_red_cost;
  }
}

void Subgradient::UpdateSubgradientAllCols(const DualState& core,
                                           FillOnlyRowIntSet& affected_rows) {
  const Model& model = core.model();
  const SubsetCostVector& reduced_costs = core.reduced_costs();
  DCHECK_EQ(subgradient_.size(), model.num_elements());
  DCHECK_EQ(selected_.size(), model.num_subsets());
  DCHECK_EQ(subgradient_.size(), core.multipliers().size());
  DCHECK_EQ(selected_.size(), reduced_costs.size());

  for (ElementIndex i : model.ElementRange()) {
    affected_rows.Insert(i, subgradient_[i]);
  }
  // Reset subgradient state
  selected_.assign(model.num_subsets(), false);
  subgradient_.assign(model.num_elements(), 1);
  lagrangian_solution_.clear();
  excluded_columns_.clear();

  const SparseColumnView& columns = model.columns();
  for (SubsetIndex j : model.SubsetRange()) {
    if (reduced_costs[j] < 0) {
      for (ElementIndex i : columns[j]) {
        --subgradient_[i];
      }
      lagrangian_solution_.push_back(j);
    }
  }
  if (use_minimal_coverage_)
    MakeMinimalSubgradient(model, reduced_costs, subgradient_,
                           lagrangian_solution_, excluded_columns_);
  squared_norm_ = .0;
  for (ElementIndex i : model.ElementRange()) {
    squared_norm_ += subgradient_[i] * subgradient_[i];
  }
}

void Subgradient::UpdateSubgradientSomeCols(
    const FillOnlyColumnSet& active_columns, const DualState& core,
    FillOnlyRowIntSet& affected_rows) {
  const Model& model = core.model();
  const SubsetCostVector& reduced_costs = core.reduced_costs();
  DCHECK_EQ(subgradient_.size(), model.num_elements());
  DCHECK_EQ(selected_.size(), model.num_subsets());
  DCHECK_EQ(subgradient_.size(), core.multipliers().size());
  DCHECK_EQ(selected_.size(), reduced_costs.size());

  if (use_minimal_coverage_)
    UndoMinimalSubgradient(model, subgradient_, excluded_columns_);

  const SparseColumnView& columns = model.columns();
  for (SubsetIndex j : active_columns) {
    if ((reduced_costs[j] < .0) == selected_[j]) {
      continue;
    }
    selected_[j] = (reduced_costs[j] < .0);
    BaseInt element_delta_subgrad = selected_[j] ? -1 : +1;
    for (ElementIndex i : columns[j]) {
      affected_rows.Insert(i, subgradient_[i]);
      subgradient_[i] += element_delta_subgrad;
    }
    if (reduced_costs[j] < .0) {
      lagrangian_solution_.push_back(j);
    }
  }
  gtl::STLEraseAllFromSequenceIf(&lagrangian_solution_,
                                 [&](SubsetIndex j) { return !selected_[j]; });
  if (use_minimal_coverage_)
    MakeMinimalSubgradient(model, reduced_costs, subgradient_,
                           lagrangian_solution_, excluded_columns_);

  for (size_t e = 0; e < affected_rows.size(); ++e) {
    auto [i, old_subg] = affected_rows.GetWithPayload(e);
    squared_norm_ += subgradient_[i] * subgradient_[i] - old_subg * old_subg;
  }
}

void Subgradient::Optimize(Cost upper_bound, DualState& dual_state,
                           SubgradienCBs& cbs) {
  const Model& model = dual_state.model();
  DCHECK_GT(model.num_subsets(), 0) << "Empty model";
  DCHECK_GT(model.num_elements(), 0) << "Empty model";

  FillOnlyColumnSet updated_cols(model.num_subsets());
  FillOnlyRowIntSet updated_rows(model.num_elements());
  UpdateSubgradientAllCols(dual_state, updated_rows.Clear());

  while (!cbs.ShouldExit(dual_state)) {
    Cost delta = upper_bound - dual_state.lower_bound();
    double step_constant = cbs.StepSize(dual_state) * delta / squared_norm_;
    updated_cols.Clear();
    for (ElementIndex i : updated_rows) {
      Cost delta_mult = step_constant * subgradient_[i];
      dual_state.ModifyMultiplier(i, delta_mult, updated_cols);
    }

    if (cbs.UpdateCore(dual_state, upper_bound)) {
      UpdateSubgradientAllCols(dual_state, updated_rows.Clear());
    } else {
      UpdateSubgradientSomeCols(updated_cols, dual_state, updated_rows.Clear());
    }
  }
}

}  // namespace operations_research::scp