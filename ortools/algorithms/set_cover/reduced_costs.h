#ifndef CAV_ORTOOLS_ALGORITHMS_SET_COVER_REDUCED_COSTS_H
#define CAV_ORTOOLS_ALGORITHMS_SET_COVER_REDUCED_COSTS_H

#include "ortools/algorithms/set_cover_model.h"

namespace operations_research::scp {
// `ReducedCosts` is currently just a vector of costs.
// In the future, we plan to update only the costs that have changed.
// This class is designed to support that future improvement.
class ReducedCosts {
 public:
  // Const vector-like access interface
  size_t size() const { return reduced_costs_.size(); }
  auto begin() const { return reduced_costs_.begin(); }
  auto end() const { return reduced_costs_.end(); }
  BaseInt operator[](SubsetIndex i) const { return reduced_costs_[i]; }

  // TODO(c4v4): store last multipliers, compare then with new ones, locally
  // update only the changed reduced costs (Note: needs row-view).
  // TODO(anyone): specialize the hell out of this, this is one of the hottest
  // spots in the algorithm.
  void UpdateReducedCosts(const Model& model,
                          const ElementCostVector& multipliers) {
    const SubsetCostVector costs = model.subset_costs();
    const SparseColumnView columns = model.columns();
    reduced_costs_ = costs;
    for (SubsetIndex j; j < costs.size(); ++j) {
      for (ElementIndex i : columns[j]) {
        reduced_costs_[j] -= multipliers[i];
      }
    }
  }

 private:
  SubsetCostVector reduced_costs_;
  // TODO(c4v4): ElementCostVector last_multipliers;
};
}  // namespace operations_research::scp

#endif /* CAV_ORTOOLS_ALGORITHMS_SET_COVER_REDUCED_COSTS_H */
