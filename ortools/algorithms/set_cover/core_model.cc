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

#include "core_model.h"

#include "ortools/algorithms/set_cover_model.h"
#include "ortools/base/stl_util.h"
namespace operations_research::scp {

namespace {
void ExtractCoreModel(const Model& full_model,
                      const SubsetMapVector& columns_map, Model& core_model) {
  // Fill core model with the selected columns
  const SparseColumnView& full_columns = full_model.columns();
  const SubsetCostVector& full_costs = full_model.subset_costs();
  core_model.ReserveNumSubsets(columns_map.size());
  for (const SubsetIndex full_j : columns_map) {
    SubsetIndex core_j(core_model.num_subsets());
    core_model.AddEmptySubset(full_costs[full_j]);
    for (const ElementIndex i : full_columns[full_j]) {
      core_model.AddElementToLastSubset(i);
    }
  }
  core_model.CreateSparseRowView();
}
}  // namespace
void CoreFromFullModel::BuildFirstCoreModel(const Model& full_model) {
  this->full_model_ = &full_model;
  SubsetMapVector& columns_map = this->columns_map_;
  columns_map.clear();
  columns_map.reserve(full_model.num_elements() * min_row_coverage);

  // Select the first min_row_coverage columns for each row
  const SparseRowView& rows = full_model.rows();
  for (const SparseRow& row : rows)
    for (RowEntryIndex n; n < std::min(row.size(), min_row_coverage); ++n) {
      columns_map.push_back(row[n]);
    }
  gtl::STLSortAndRemoveDuplicates(&columns_map);
  ExtractCoreModel(full_model, columns_map, this->core_model());
}

CoreFromFullModel::CoreFromFullModel(const Model* full_model)
    : full_model_(full_model) {
  BuildFirstCoreModel(*full_model_);
}

namespace {
// Pricing period as of [1].
size_t ComputeNextUpdatePeriod(size_t period, size_t max_countdown,
                               Cost lower_bound, Cost real_lower_bound,
                               Cost upper_bound) {
  const Cost delta = (lower_bound - real_lower_bound) / upper_bound;
  if (delta <= 1e-6) return std::min(max_countdown, 10 * period);
  if (delta <= 0.02) return std::min(max_countdown, 5 * period);
  if (delta <= 0.2) return std::min(max_countdown, 2 * period);
  return 10;
}

void SelecteMinRedCostColumns(const Model& full_model,
                              const ReducedCosts& reduced_costs,
                              SubsetMapVector& columns_map,
                              SubsetBoolVector& selected) {
  // TODO(c4v4): implement
}

void SelectMinRedCostByRow(const Model& full_model,
                           const ReducedCosts& reduced_costs,
                           SubsetMapVector& columns_map,
                           SubsetBoolVector& selected) {
  // TODO(c4v4): implement
}
}  // namespace

Cost CoreFromFullModel::UpdateCoreModel(Cost upper_bound,
                                        ElementCostVector& multipliers) {
  const Model& full_model = *this->full_model_;
  if (--this->countdown > 0) return std::numeric_limits<Cost>::lowest();
  Cost real_lower_bound = 0.0;
  for (Cost multiplier : multipliers) {
    real_lower_bound += multiplier;
  }
  reduced_costs_.UpdateReducedCosts(full_model, multipliers);
  const SparseColumnView& columns = full_model.columns();
  for (SubsetIndex j; j < reduced_costs_.size(); ++j) {
    if (reduced_costs_[j] < 0) real_lower_bound += reduced_costs_[j];
  }

  SubsetBoolVector selected(full_model.num_subsets(), false);
  SubsetMapVector& columns_map = this->columns_map_;
  columns_map.clear();
  SelecteMinRedCostColumns(full_model, reduced_costs_, columns_map, selected);
  SelectMinRedCostByRow(full_model, reduced_costs_, columns_map, selected);
  ExtractCoreModel(full_model, columns_map, this->core_model());
  this->countdown = ComputeNextUpdatePeriod(
      countdown, max_countdown, lower_bound(), real_lower_bound, upper_bound);
  return real_lower_bound;
}

}  // namespace operations_research::scp