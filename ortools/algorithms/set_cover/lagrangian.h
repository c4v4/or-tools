#ifndef CAV_ORTOOLS_ALGORITHMS_SET_COVER_LAGRANGIAN_H
#define CAV_ORTOOLS_ALGORITHMS_SET_COVER_LAGRANGIAN_H

#include <tuple>
#include <utility>

#include "../set_cover_model.h"

namespace operations_research::scp {

template <typename T, typename... PayloadTs>
class FillOnlySet {
  static constexpr BaseInt kNotInSet = std::numeric_limits<BaseInt>::max();
  static constexpr auto kIndexes = std::index_sequence_for<PayloadTs...>{};

 public:
  FillOnlySet(BaseInt num_subsets) : positions_(num_subsets, kNotInSet) {}

  auto begin() const { return elements_.begin(); }
  auto end() const { return elements_.end(); }
  size_t size() const { return elements_.size(); }
  size_t max_size() const { return positions_.size(); }
  T operator[](BaseInt pos) const { return elements_[pos]; }

  auto GetWithPayload(BaseInt pos) const {
    return FoldPayloads(payloads_, kIndexes, [&](auto&... payload) {
      return std::forward_as_tuple(elements_[pos], payload[pos]...);
    });
  }
  FillOnlySet& Clear() {
    for (T e : elements_) {
      positions_[e] = kNotInSet;
    }
    elements_.clear();
    FoldPayloads(payloads_, kIndexes,
                 [&](auto&... payload) { (payload.clear(), ...); });
    return *this;
  }
  bool Insert(T e, const PayloadTs&... vals) {
    if (positions_[e] != kNotInSet) {
      return false;
    }
    positions_[e] = elements_.size();
    elements_.push_back(e);
    FoldPayloads(payloads_, kIndexes,
                 [&](auto&... payload) { (payload.push_back(vals), ...); });
    return true;
  }
  void InsertAll(const PayloadTs&... default_vals) {
    absl::c_iota(positions_, 0);
    elements_.resize(max_size());
    absl::c_iota(elements_, 0);
    FoldPayloads(payloads_, kIndexes, [&](auto&... payload) {
      (payload.assign(max_size(), default_vals), ...);
    });
  }

 private:
  template <typename PT, size_t... Is, typename Op>
  static auto FoldPayloads(PT&& payload, std::index_sequence<Is...>,
                           Op lambda) {
    return lambda(std::get<Is>(std::forward<PT>(payload))...);
  }

 private:
  std::vector<T> elements_;
  SubsetToIntVector positions_;
  std::tuple<std::vector<PayloadTs>...> payloads_;
};

using FillOnlyColumnSet = FillOnlySet<SubsetIndex>;
using FillOnlyRowIntSet = FillOnlySet<ElementIndex, BaseInt>;

class DualState {
 public:
  DualState(Model&& model)
      : model_(model),
        lower_bound_(0.0),
        multipliers_(model.num_elements(), 0.0),
        reduced_costs_(model.subset_costs()) {}

  Cost lower_bound() const { return lower_bound_; }
  const Model& model() const { return model_; }
  const ElementCostVector& multipliers() const { return multipliers_; }
  const SubsetCostVector& reduced_costs() const { return reduced_costs_; }
  void ModifyMultiplier(ElementIndex i, Cost delta_mult,
                        FillOnlyColumnSet& affected_columns);

 protected:
  Model model_;
  Cost lower_bound_;
  ElementCostVector multipliers_;
  SubsetCostVector reduced_costs_;
};

class SubgradienCBs : protected DualState {
 public:
  SubgradienCBs(Model&& model) : DualState(std::move(model)) {}
  DualState& dual_state() { return *this; }
  const DualState& dual_state() const { return *this; }

  virtual bool ShouldExit(const DualState& dual_state) = 0;
  virtual double StepSize(const DualState& dual_state) = 0;
  virtual bool UpdateCore(DualState& dual_state, Cost upper_bound) = 0;
  virtual ~SubgradienCBs() = default;
};

template <typename UpdaterT, typename StopperT, typename StepSizerT>
class SubgradientCBsAggregator : SubgradienCBs {
  static constexpr size_t kMinRowCoverage = 5;

  SubgradientCBsAggregator(const UpdaterT& updater, const StopperT& stopper,
                           const StepSizerT& stepsizer);
  SubgradientCBsAggregator(UpdaterT&& update_trigger, StopperT&& stopper,
                           StepSizerT&& stepsizer);

  bool UpdateCore(DualState& dual_state, Cost upper_bound) override {
    return updater_(dual_state, upper_bound);
  }
  bool ShouldExit(const DualState& dual_state) override {
    return stopper_(dual_state);
  }
  double StepSize(const DualState& dual_state) override {
    return stepsizer_(dual_state);
  }

  const UpdaterT& updater() { return updater_; }
  const StopperT& stopper() { return stopper_; }
  const StepSizerT& stepsizer() { return stepsizer_; }

 private:
  UpdaterT updater_;
  StopperT stopper_;
  StepSizerT stepsizer_;
};

class SubgradUpdater {
 public:
  static constexpr size_t kMinRowCoverage = 5;

  SubgradUpdater(BaseInt period, BaseInt max_period);
  bool operator()(const DualState& dual_state, Cost upper_bound);
  bool ShouldUpdate(Cost lower_bound, Cost real_lower_bound, Cost upper_bound);

  Cost full_lower_bound() const { return full_lower_bound_; }
  const ElementCostVector& full_multipliers() const {
    return full_multipliers_;
  }
  const SubsetCostVector& full_reduced_costs() const {
    return full_reduced_costs_;
  }

 private:
  BaseInt countdown;
  BaseInt period;
  BaseInt max_period;

  SubsetMapVector columns_map_;
  const Model* full_model_;
  Cost full_lower_bound_;
  ElementCostVector full_multipliers_;
  SubsetCostVector full_reduced_costs_;
};

class SubgradStepSizer {
 public:
  SubgradStepSizer(BaseInt period, double step_size);
  double step_size() const { return step_size_; }
  double operator()(const DualState& dual_state);

 private:
  double step_size_;
  BaseInt period_;
  BaseInt countdown_;
  Cost min_lower_bound_;
  Cost max_lower_bound_;
};

class SubgradStopper {
 public:
  SubgradStopper(BaseInt period);
  bool operator()(const DualState& dual_state);

 private:
  BaseInt countdown;
  BaseInt period;
  Cost prev_lower_bound;
};

// CftSubgradientCore is a SubgradientCore implementation that updates the
// restricted model by selecting the most promising columns from the full model.
// This specific implementation follows the approach described in:
// [1] Caprara, Alberto, Matteo Fischetti, and Paolo Toth. 1999. “A Heuristic
// Method for the Set Covering Problem.” Operations Research 47 (5): 730–43.
// https://www.jstor.org/stable/223097
//
// NOTE: A REFERENCE TO THE FULL MODEL IS KEPT, BUT THE FULL MODEL IS NOT OWNED
// In theory, a cleaner design would own the full model internally (moving into
// it in the constructor), however, since the full model might be used in
// other algorithms prior and after this one, it is kept only as a pointer for
// better ergonomics.
using CftSubgradientCore =
    SubgradientCBsAggregator<SubgradUpdater, SubgradStopper, SubgradStepSizer>;

// SubgradientState is a class that maintains the current state of the
// subgradient.
// NOTE: This class was named SubgradientInvariant mimicking the
// `SetCoverInvariant` data structure, however:
//  1. The term "invariant" is not ideal here. states are properties, but
//      this class holds data. It is true that this data maintains certain
//      states, which are enforced by correctly using the class's public
//      interface. But this is a basic property of classes in OOP. If this were
//      not the case, it would be a struct with public members modified by
//      external functions.
//  2. Keeping a pointer to the model within the class can be convenient and
//      ensures that the state always refers to that model. However, this
//      approach introduces hidden external dependencies for the class user. In
//      this context, the model can (and does) change and gets updated, so the
//      risk of having an outdated dual state pointing to an updated Model is
//      high. Since the model changes, whoever changes it must also update the
//      dual state. By not storing the model within the class, we make this
//      dependency explicit, since member function signatures will require a
//      model.
class Subgradient {
 public:
  Subgradient(const Model& model, bool use_minimal_coverage = true)
      : selected_(model.num_subsets(), false),
        lagrangian_solution_(),
        excluded_columns_(),
        subgradient_(model.num_elements(), 1),
        use_minimal_coverage_(use_minimal_coverage) {}

  bool selected(SubsetIndex j) const { return selected_[j]; }
  const ElementToIntVector& subgradient() const { return subgradient_; }
  double squared_norm() const { return squared_norm_; }
  const std::vector<SubsetIndex>& lagrangian_solution() const {
    return lagrangian_solution_;
  }
  const std::vector<SubsetIndex>& excluded_columns() const {
    return excluded_columns_;
  }
  void set_use_minimal_coverage(bool use_minimal_coverage) {
    use_minimal_coverage_ = use_minimal_coverage;
  }

  // Updated the subgradient current state using the current dual information.
  void UpdateSubgradientAllCols(const DualState& core,
                             FillOnlyRowIntSet& affected_rows);

  // Localized update of the subgradient state.
  void UpdateSubgradientSomeCols(const FillOnlyColumnSet& active_columns,
                               const DualState& core,
                               FillOnlyRowIntSet& affected_rows);

  void Optimize(Cost upper_bound, DualState& dual_state, SubgradienCBs& core);

 private:
  SubsetBoolVector selected_;
  std::vector<SubsetIndex> lagrangian_solution_;
  std::vector<SubsetIndex> excluded_columns_;
  ElementToIntVector subgradient_;
  double squared_norm_;
  bool use_minimal_coverage_;
};

class GreedyUpdater {
 public:
  // TODO(cava): implement greedy algorithm
  constexpr bool operator()(DualState& /*dual_state*/, Cost /*upper_bound*/) {
    return false;
  }
};

class GreedyStepSizer {
 public:
  double step_size;
  double operator()(const DualState& /*dual_state*/) { return step_size; }
};

class GreedyStopper {
 public:
  BaseInt countdown;
  bool operator()(const DualState& /*dual_state*/) { return --countdown < 0; }
};

// During the heuristic phase CFT, we run some more subgradient iterations.
// However, this time we keep the core model fixed and only jump around a bit to
// explore different multipliers. The updater does not update anything it just
// call the CGT greedy heuristic and keep track of the best solution found.
using GreedySubgradientCore =
    SubgradientCBsAggregator<SubgradUpdater, SubgradStopper, SubgradStepSizer>;

}  // namespace operations_research::scp

#endif /* CAV_ORTOOLS_ALGORITHMS_SET_COVER_LAGRANGIAN_H */
