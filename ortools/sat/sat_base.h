// Copyright 2010-2025 Google LLC
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

// Basic types and classes used by the sat solver.

#ifndef OR_TOOLS_SAT_SAT_BASE_H_
#define OR_TOOLS_SAT_SAT_BASE_H_

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "ortools/base/logging.h"
#include "ortools/base/strong_vector.h"
#include "ortools/util/bitset.h"
#include "ortools/util/strong_integers.h"

namespace operations_research {
namespace sat {

// Index of a variable (>= 0).
DEFINE_STRONG_INDEX_TYPE(BooleanVariable);
const BooleanVariable kNoBooleanVariable(-1);

// Index of a literal (>= 0), see Literal below.
DEFINE_STRONG_INDEX_TYPE(LiteralIndex);
const LiteralIndex kNoLiteralIndex(-1);

// Special values used in some API to indicate a literal that is always true
// or always false.
const LiteralIndex kTrueLiteralIndex(-2);
const LiteralIndex kFalseLiteralIndex(-3);

// A literal is used to represent a variable or its negation. If it represents
// the variable it is said to be positive. If it represent its negation, it is
// said to be negative. We support two representations as an integer.
//
// The "signed" encoding of a literal is convenient for input/output and is used
// in the cnf file format. For a 0-based variable index x, (x + 1) represent the
// variable x and -(x + 1) represent its negation. The signed value 0 is an
// undefined literal and this class can never contain it.
//
// The "index" encoding of a literal is convenient as an index to an array
// and is the one used internally for efficiency. It is always positive or zero,
// and for a 0-based variable index x, (x << 1) encode the variable x and the
// same number XOR 1 encode its negation.
class Literal {
 public:
  explicit Literal(int signed_value)
      : index_(signed_value > 0 ? ((signed_value - 1) << 1)
                                : ((-signed_value - 1) << 1) ^ 1) {
    CHECK_NE(signed_value, 0);
  }

  Literal() = default;
  explicit Literal(LiteralIndex index) : index_(index.value()) {}
  Literal(BooleanVariable variable, bool is_positive)
      : index_(is_positive ? (variable.value() << 1)
                           : (variable.value() << 1) ^ 1) {}

  // We want a literal to be implicitly converted to a LiteralIndex().
  // Before this, we used to have many literal.Index() that didn't add anything.
  //
  // TODO(user): LiteralIndex might not even be needed, but because of the
  // signed value business, it is still safer with it.
  operator LiteralIndex() const { return Index(); }  // NOLINT

  BooleanVariable Variable() const { return BooleanVariable(index_ >> 1); }
  bool IsPositive() const { return !(index_ & 1); }
  bool IsNegative() const { return (index_ & 1); }

  LiteralIndex Index() const { return LiteralIndex(index_); }
  LiteralIndex NegatedIndex() const { return LiteralIndex(index_ ^ 1); }

  int SignedValue() const {
    return (index_ & 1) ? -((index_ >> 1) + 1) : ((index_ >> 1) + 1);
  }

  Literal Negated() const { return Literal(NegatedIndex()); }

  std::string DebugString() const {
    return absl::StrFormat("%+d", SignedValue());
  }

  bool operator==(Literal other) const { return index_ == other.index_; }
  bool operator!=(Literal other) const { return index_ != other.index_; }
  bool operator<(const Literal& other) const { return index_ < other.index_; }

  template <typename H>
  friend H AbslHashValue(H h, Literal literal) {
    return H::combine(std::move(h), literal.index_);
  }

 private:
  int index_;
};

inline std::ostream& operator<<(std::ostream& os, Literal literal) {
  os << literal.DebugString();
  return os;
}

template <typename Sink, typename... T>
void AbslStringify(Sink& sink, Literal arg) {
  absl::Format(&sink, "%s", arg.DebugString());
}

inline std::ostream& operator<<(std::ostream& os,
                                absl::Span<const Literal> literals) {
  os << "[";
  bool first = true;
  for (const Literal literal : literals) {
    if (first) {
      first = false;
    } else {
      os << ",";
    }
    os << literal.DebugString();
  }
  os << "]";
  return os;
}

// Only used for testing to use the classical SAT notation for a literal. This
// allows to write Literals({+1, -4, +3}) for the clause with BooleanVariable 0
// and 2 appearing positively and 3 negatively.
inline std::vector<Literal> Literals(absl::Span<const int> input) {
  std::vector<Literal> result(input.size());
  for (int i = 0; i < result.size(); ++i) {
    result[i] = Literal(input[i]);
  }
  return result;
}

// Holds the current variable assignment of the solver.
// Each variable can be unassigned or be assigned to true or false.
class VariablesAssignment {
 public:
  VariablesAssignment() = default;
  explicit VariablesAssignment(int num_variables) { Resize(num_variables); }

  // This type is neither copyable nor movable.
  VariablesAssignment(const VariablesAssignment&) = delete;
  VariablesAssignment& operator=(const VariablesAssignment&) = delete;
  void Resize(int num_variables) {
    assignment_.Resize(LiteralIndex(num_variables << 1));
  }

  // Makes the given literal true by assigning its underlying variable to either
  // true or false depending on the literal sign. This can only be called on an
  // unassigned variable.
  void AssignFromTrueLiteral(Literal literal) {
    DCHECK(!VariableIsAssigned(literal.Variable()));
    assignment_.Set(literal.Index());
  }

  // Unassign the variable corresponding to the given literal.
  // This can only be called on an assigned variable.
  void UnassignLiteral(Literal literal) {
    DCHECK(VariableIsAssigned(literal.Variable()));
    assignment_.ClearTwoBits(literal.Index());
  }

  // Literal getters. Note that both can be false, in which case the
  // corresponding variable is not assigned.
  bool LiteralIsFalse(Literal literal) const {
    return assignment_.IsSet(literal.NegatedIndex());
  }
  bool LiteralIsTrue(Literal literal) const {
    return assignment_.IsSet(literal.Index());
  }
  bool LiteralIsAssigned(Literal literal) const {
    return assignment_.AreOneOfTwoBitsSet(literal.Index());
  }

  // Returns true iff the given variable is assigned.
  bool VariableIsAssigned(BooleanVariable var) const {
    return assignment_.AreOneOfTwoBitsSet(LiteralIndex(var.value() << 1));
  }

  // Returns the literal of the given variable that is assigned to true.
  // That is, depending on the variable, it can be the positive literal or the
  // negative one. Only call this on an assigned variable.
  Literal GetTrueLiteralForAssignedVariable(BooleanVariable var) const {
    DCHECK(VariableIsAssigned(var));
    return Literal(var, assignment_.IsSet(LiteralIndex(var.value() << 1)));
  }

  int NumberOfVariables() const { return assignment_.size().value() / 2; }

 private:
  // The encoding is as follows:
  // - assignment_.IsSet(literal.Index()) means literal is true.
  // - assignment_.IsSet(literal.Index() ^ 1]) means literal is false.
  // - If both are false, then the variable (and the literal) is unassigned.
  Bitset64<LiteralIndex> assignment_;

  friend class AssignmentView;
};

// For "hot" loop, it is better not to reload the Bitset64 pointer on each
// check.
class AssignmentView {
 public:
  explicit AssignmentView(const VariablesAssignment& assignment)
      : view_(assignment.assignment_.const_view()) {}

  bool LiteralIsFalse(Literal literal) const {
    return view_[literal.NegatedIndex()];
  }

  bool LiteralIsTrue(Literal literal) const { return view_[literal.Index()]; }

 private:
  Bitset64<LiteralIndex>::ConstView view_;
};

// Forward declaration.
class SatClause;
class SatPropagator;

// Information about a variable assignment.
struct AssignmentInfo {
  // The decision level at which this assignment was made. This starts at 0 and
  // increases each time the solver takes a search decision.
  //
  // TODO(user): We may be able to get rid of that for faster enqueues. Most of
  // the code only need to know if this is 0 or the highest level, and for the
  // LBD computation, the literal of the conflict are already ordered by level,
  // so we could do it fairly efficiently.
  //
  // TODO(user): We currently don't support more than 2^28 decision levels. That
  // should be enough for most practical problem, but we should fail properly if
  // this limit is reached.
  uint32_t level : 28;

  // The type of assignment (see AssignmentType below).
  //
  // Note(user): We currently don't support more than 16 types of assignment.
  // This is checked in RegisterPropagator().
  mutable uint32_t type : 4;

  // The index of this assignment in the trail.
  int32_t trail_index;

  std::string DebugString() const {
    return absl::StrFormat("level:%d type:%d trail_index:%d", level, type,
                           trail_index);
  }
};
static_assert(sizeof(AssignmentInfo) == 8,
              "ERROR_AssignmentInfo_is_not_well_compacted");

// Each literal on the trail will have an associated propagation "type" which is
// either one of these special types or the id of a propagator.
struct AssignmentType {
  static constexpr int kCachedReason = 0;
  static constexpr int kUnitReason = 1;
  static constexpr int kSearchDecision = 2;
  static constexpr int kSameReasonAs = 3;

  // Propagator ids starts from there and are created dynamically.
  static constexpr int kFirstFreePropagationId = 4;
};

// The solver trail stores the assignment made by the solver in order.
// This class is responsible for maintaining the assignment of each variable
// and the information of each assignment.
class Trail {
 public:
  Trail() {
    current_info_.trail_index = 0;
    current_info_.level = 0;
  }

  // This type is neither copyable nor movable.
  Trail(const Trail&) = delete;
  Trail& operator=(const Trail&) = delete;

  void Resize(int num_variables);

  // Registers a propagator. This assigns a unique id to this propagator and
  // calls SetPropagatorId() on it.
  void RegisterPropagator(SatPropagator* propagator);

  // Enqueues the assignment that make the given literal true on the trail. This
  // should only be called on unassigned variables.
  void SetCurrentPropagatorId(int propagator_id) {
    current_info_.type = propagator_id;
  }
  void FastEnqueue(Literal true_literal) {
    DCHECK(!assignment_.VariableIsAssigned(true_literal.Variable()));
    trail_[current_info_.trail_index] = true_literal;
    info_[true_literal.Variable()] = current_info_;
    assignment_.AssignFromTrueLiteral(true_literal);
    ++current_info_.trail_index;
  }
  void Enqueue(Literal true_literal, int propagator_id) {
    SetCurrentPropagatorId(propagator_id);
    FastEnqueue(true_literal);
  }

  // Specific Enqueue() version for the search decision.
  void EnqueueSearchDecision(Literal true_literal) {
    Enqueue(true_literal, AssignmentType::kSearchDecision);
  }

  // Specific Enqueue() version for a fixed variable.
  void EnqueueWithUnitReason(Literal true_literal) {
    Enqueue(true_literal, AssignmentType::kUnitReason);
  }

  // Some constraints propagate a lot of literals at once. In these cases, it is
  // more efficient to have all the propagated literals except the first one
  // referring to the reason of the first of them.
  void EnqueueWithSameReasonAs(Literal true_literal,
                               BooleanVariable reference_var) {
    reference_var_with_same_reason_as_[true_literal.Variable()] = reference_var;
    Enqueue(true_literal, AssignmentType::kSameReasonAs);
  }

  // Enqueues the given literal using the current content of
  // GetEmptyVectorToStoreReason() as the reason. This API is a bit more
  // leanient and does not require the literal to be unassigned. If it is
  // already assigned to false, then MutableConflict() will be set appropriately
  // and this will return false otherwise this will enqueue the literal and
  // returns true.
  ABSL_MUST_USE_RESULT bool EnqueueWithStoredReason(Literal true_literal) {
    if (assignment_.LiteralIsTrue(true_literal)) return true;
    if (assignment_.LiteralIsFalse(true_literal)) {
      *MutableConflict() = reasons_repository_[Index()];
      MutableConflict()->push_back(true_literal);
      return false;
    }

    Enqueue(true_literal, AssignmentType::kCachedReason);
    const BooleanVariable var = true_literal.Variable();
    reasons_[var] = reasons_repository_[info_[var].trail_index];
    old_type_[var] = info_[var].type;
    info_[var].type = AssignmentType::kCachedReason;
    return true;
  }

  // Returns the reason why this variable was assigned.
  //
  // Note that this shouldn't be called on a variable at level zero, because we
  // don't cleanup the reason data for these variables but the underlying
  // clauses may have been deleted.
  //
  // If conflict_id >= 0, this indicate that this was called as part of the
  // first-UIP procedure. It has a few implication:
  //  - The reason do not need to be cached and can be adapted to the current
  //    conflict.
  //  - Some data can be reused between two calls about the same conflict.
  //  - Note however that if the reason is a simple clause, we shouldn't adapt
  //    it because we rely on extra fact in the first UIP code where we detect
  //    subsumed clauses for instance.
  absl::Span<const Literal> Reason(BooleanVariable var,
                                   int64_t conflict_id = -1) const;

  // Returns the "type" of an assignment (see AssignmentType). Note that this
  // function never returns kSameReasonAs or kCachedReason, it instead returns
  // the initial type that caused this assignment. As such, it is different
  // from Info(var).type and the latter should not be used outside this class.
  int AssignmentType(BooleanVariable var) const;

  // If a variable was propagated with EnqueueWithSameReasonAs(), returns its
  // reference variable. Otherwise return the given variable.
  BooleanVariable ReferenceVarWithSameReason(BooleanVariable var) const;

  // This can be used to get a location at which the reason for the literal
  // at trail_index on the trail can be stored. This clears the vector before
  // returning it.
  std::vector<Literal>* GetEmptyVectorToStoreReason(int trail_index) const {
    if (trail_index >= reasons_repository_.size()) {
      reasons_repository_.resize(trail_index + 1);
    }
    reasons_repository_[trail_index].clear();
    return &reasons_repository_[trail_index];
  }

  // Shortcut for GetEmptyVectorToStoreReason(Index()).
  std::vector<Literal>* GetEmptyVectorToStoreReason() const {
    return GetEmptyVectorToStoreReason(Index());
  }

  // Explicitly overwrite the reason so that the given propagator will be
  // asked for it. This is currently only used by the BinaryImplicationGraph.
  void ChangeReason(int trail_index, int propagator_id) {
    const BooleanVariable var = trail_[trail_index].Variable();
    info_[var].type = propagator_id;
    old_type_[var] = propagator_id;
  }

  // Reverts the trail and underlying assignment to the given target trail
  // index. Note that we do not touch the assignment info.
  void Untrail(int target_trail_index) {
    const int index = Index();
    num_untrailed_enqueues_ += index - target_trail_index;
    for (int i = target_trail_index; i < index; ++i) {
      assignment_.UnassignLiteral(trail_[i]);
    }
    current_info_.trail_index = target_trail_index;
  }

  // Changes the decision level used by the next Enqueue().
  void SetDecisionLevel(int level) { current_info_.level = level; }
  int CurrentDecisionLevel() const { return current_info_.level; }

  // Generic interface to set the current failing clause.
  //
  // Returns the address of a vector where a client can store the current
  // conflict. This vector will be returned by the FailingClause() call.
  std::vector<Literal>* MutableConflict() {
    failing_sat_clause_ = nullptr;
    return &conflict_;
  }

  // Returns the last conflict.
  absl::Span<const Literal> FailingClause() const {
    if (DEBUG_MODE && debug_checker_ != nullptr) {
      CHECK(debug_checker_(conflict_));
    }
    return conflict_;
  }

  // Specific SatClause interface so we can update the conflict clause activity.
  // Note that MutableConflict() automatically sets this to nullptr, so we can
  // know whether or not the last conflict was caused by a clause.
  void SetFailingSatClause(SatClause* clause) { failing_sat_clause_ = clause; }
  SatClause* FailingSatClause() const { return failing_sat_clause_; }

  // Getters.
  int NumVariables() const { return trail_.size(); }
  int64_t NumberOfEnqueues() const { return num_untrailed_enqueues_ + Index(); }
  int Index() const { return current_info_.trail_index; }
  // This accessor can return trail_.end(). operator[] cannot. This allows
  // normal std:vector operations, such as assign(begin, end).
  std::vector<Literal>::const_iterator IteratorAt(int index) const {
    return trail_.begin() + index;
  }
  const Literal& operator[](int index) const { return trail_[index]; }
  const VariablesAssignment& Assignment() const { return assignment_; }
  const AssignmentInfo& Info(BooleanVariable var) const {
    DCHECK_GE(var, 0);
    DCHECK_LT(var, info_.size());
    return info_[var];
  }

  // Print the current literals on the trail.
  std::string DebugString() const {
    std::string result;
    for (int i = 0; i < current_info_.trail_index; ++i) {
      if (!result.empty()) result += " ";
      result += trail_[i].DebugString();
    }
    return result;
  }

  void RegisterDebugChecker(
      std::function<bool(absl::Span<const Literal> clause)> checker) {
    debug_checker_ = std::move(checker);
  }

 private:
  int64_t num_untrailed_enqueues_ = 0;
  AssignmentInfo current_info_;
  VariablesAssignment assignment_;
  std::vector<Literal> trail_;
  std::vector<Literal> conflict_;
  util_intops::StrongVector<BooleanVariable, AssignmentInfo> info_;
  SatClause* failing_sat_clause_;

  // Data used by EnqueueWithSameReasonAs().
  util_intops::StrongVector<BooleanVariable, BooleanVariable>
      reference_var_with_same_reason_as_;

  // Reason cache. Mutable since we want the API to be the same whether the
  // reason are cached or not.
  //
  // When a reason is computed for the first time, we change the type of the
  // variable assignment to kCachedReason so that we know that if it is needed
  // again the reason can just be retrieved by a direct access to reasons_. The
  // old type is saved in old_type_ and can be retrieved by
  // AssignmentType().
  //
  // Note(user): Changing the type is not "clean" but it is efficient. The idea
  // is that it is important to do as little as possible when pushing/popping
  // literals on the trail. Computing the reason happens a lot less often, so it
  // is okay to do slightly more work then. Note also, that we don't need to
  // do anything on "untrail", the kCachedReason type will be overwritten when
  // the same variable is assigned again.
  //
  // TODO(user): An alternative would be to change the sign of the type. This
  // would remove the need for a separate old_type_ vector, but it requires
  // more bits for the type filed in AssignmentInfo.
  //
  // Note that we use a deque for the reason repository so that if we add
  // variables, the memory address of the vectors (kept in reasons_) are still
  // valid.
  mutable std::deque<std::vector<Literal>> reasons_repository_;
  mutable util_intops::StrongVector<BooleanVariable, absl::Span<const Literal>>
      reasons_;
  mutable util_intops::StrongVector<BooleanVariable, int> old_type_;

  // This is used by RegisterPropagator() and Reason().
  std::vector<SatPropagator*> propagators_;

  std::function<bool(absl::Span<const Literal> clause)> debug_checker_ =
      nullptr;
};

// Base class for all the SAT constraints.
class SatPropagator {
 public:
  explicit SatPropagator(const std::string& name)
      : name_(name), propagator_id_(-1), propagation_trail_index_(0) {}

  // This type is neither copyable nor movable.
  SatPropagator(const SatPropagator&) = delete;
  SatPropagator& operator=(const SatPropagator&) = delete;
  virtual ~SatPropagator() = default;

  // Sets/Gets this propagator unique id.
  void SetPropagatorId(int id) { propagator_id_ = id; }
  int PropagatorId() const { return propagator_id_; }

  // Inspects the trail from propagation_trail_index_ until at least one literal
  // is propagated. Returns false iff a conflict is detected (in which case
  // trail->SetFailingClause() must be called).
  //
  // This must update propagation_trail_index_ so that all the literals before
  // it have been propagated. In particular, if nothing was propagated, then
  // PropagationIsDone() must return true.
  virtual bool Propagate(Trail* trail) = 0;

  // Reverts the state so that all the literals with a trail index greater or
  // equal to the given one are not processed for propagation. Note that the
  // trail current decision level is already reverted before this is called.
  //
  // TODO(user): Currently this is called at each Backtrack(), but we could
  // bundle the calls in case multiple conflict one after the other are detected
  // even before the Propagate() call of a SatPropagator is called.
  //
  // TODO(user): It is not yet 100% the case, but this can be guaranteed to be
  // called with a trail index that will always be the start of a new decision
  // level.
  virtual void Untrail(const Trail& /*trail*/, int trail_index) {
    propagation_trail_index_ = std::min(propagation_trail_index_, trail_index);
  }

  // Explains why the literal at given trail_index was propagated by returning a
  // reason for this propagation. This will only be called for literals that are
  // on the trail and were propagated by this class.
  //
  // The interpretation is that because all the literals of a reason were
  // assigned to false, we could deduce the assignment of the given variable.
  //
  // The returned Span has to be valid until the literal is untrailed. A client
  // can use trail_.GetEmptyVectorToStoreReason() if it doesn't have a memory
  // location that already contains the reason.
  //
  // If conlict id is positive, then this is called during first UIP resolution
  // and we will backtrack over this literal right away, so we don't need to
  // have a span that survive more than once.
  virtual absl::Span<const Literal> Reason(const Trail& /*trail*/,
                                           int /*trail_index*/,
                                           int64_t /*conflict_id*/) const {
    LOG(FATAL) << "Not implemented.";
    return {};
  }

  // Returns true if all the preconditions for Propagate() are satisfied.
  // This is just meant to be used in a DCHECK.
  bool PropagatePreconditionsAreSatisfied(const Trail& trail) const;

  // Returns true iff all the trail was inspected by this propagator.
  bool PropagationIsDone(const Trail& trail) const {
    return propagation_trail_index_ == trail.Index();
  }

  // Small optimization: If a propagator does not contain any "constraints"
  // there is no point calling propagate on it. Before each propagation, the
  // solver will checks for emptiness, and construct an optimized list of
  // propagator before looping many time over the list.
  virtual bool IsEmpty() const { return false; }

 protected:
  const std::string name_;
  int propagator_id_;
  int propagation_trail_index_;
};

// ########################  Implementations below  ########################

// TODO(user): A few of these method should be moved in a .cc

inline bool SatPropagator::PropagatePreconditionsAreSatisfied(
    const Trail& trail) const {
  if (propagation_trail_index_ > trail.Index()) {
    LOG(INFO) << "Issue in '" << name_ << ":"
              << " propagation_trail_index_=" << propagation_trail_index_
              << " trail_.Index()=" << trail.Index();
    return false;
  }
  if (propagation_trail_index_ < trail.Index() &&
      trail.Info(trail[propagation_trail_index_].Variable()).level !=
          trail.CurrentDecisionLevel()) {
    LOG(INFO) << "Issue in '" << name_ << "':"
              << " propagation_trail_index_=" << propagation_trail_index_
              << " trail_.Index()=" << trail.Index()
              << " level_at_propagation_index="
              << trail.Info(trail[propagation_trail_index_].Variable()).level
              << " current_decision_level=" << trail.CurrentDecisionLevel();
    return false;
  }
  return true;
}

inline void Trail::Resize(int num_variables) {
  assignment_.Resize(num_variables);
  info_.resize(num_variables);
  trail_.resize(num_variables);
  reasons_.resize(num_variables);

  // TODO(user): these vectors are not always used. Initialize them
  // dynamically.
  old_type_.resize(num_variables);
  reference_var_with_same_reason_as_.resize(num_variables);
}

inline void Trail::RegisterPropagator(SatPropagator* propagator) {
  if (propagators_.empty()) {
    propagators_.resize(AssignmentType::kFirstFreePropagationId);
  }
  CHECK_LT(propagators_.size(), 16);
  propagator->SetPropagatorId(propagators_.size());
  propagators_.push_back(propagator);
}

inline BooleanVariable Trail::ReferenceVarWithSameReason(
    BooleanVariable var) const {
  DCHECK(Assignment().VariableIsAssigned(var));
  // Note that we don't use AssignmentType() here.
  if (info_[var].type == AssignmentType::kSameReasonAs) {
    var = reference_var_with_same_reason_as_[var];
    DCHECK(Assignment().VariableIsAssigned(var));
    DCHECK_NE(info_[var].type, AssignmentType::kSameReasonAs);
  }
  return var;
}

inline int Trail::AssignmentType(BooleanVariable var) const {
  if (info_[var].type == AssignmentType::kSameReasonAs) {
    var = reference_var_with_same_reason_as_[var];
    DCHECK_NE(info_[var].type, AssignmentType::kSameReasonAs);
  }
  const int type = info_[var].type;
  return type != AssignmentType::kCachedReason ? type : old_type_[var];
}

inline absl::Span<const Literal> Trail::Reason(BooleanVariable var,
                                               int64_t conflict_id) const {
  // Special case for AssignmentType::kSameReasonAs to avoid a recursive call.
  var = ReferenceVarWithSameReason(var);

  // Fast-track for cached reason.
  if (info_[var].type == AssignmentType::kCachedReason) {
    if (DEBUG_MODE && debug_checker_ != nullptr) {
      std::vector<Literal> clause;
      clause.assign(reasons_[var].begin(), reasons_[var].end());
      clause.push_back(assignment_.GetTrueLiteralForAssignedVariable(var));
      CHECK(debug_checker_(clause));
    }
    return reasons_[var];
  }

  const AssignmentInfo& info = info_[var];
  if (info.type == AssignmentType::kUnitReason ||
      info.type == AssignmentType::kSearchDecision) {
    reasons_[var] = {};
  } else {
    DCHECK_LT(info.type, propagators_.size());
    DCHECK(propagators_[info.type] != nullptr) << info.type;
    reasons_[var] =
        propagators_[info.type]->Reason(*this, info.trail_index, conflict_id);
  }
  old_type_[var] = info.type;
  info_[var].type = AssignmentType::kCachedReason;
  if (DEBUG_MODE && debug_checker_ != nullptr) {
    std::vector<Literal> clause;
    clause.assign(reasons_[var].begin(), reasons_[var].end());
    clause.push_back(assignment_.GetTrueLiteralForAssignedVariable(var));
    CHECK(debug_checker_(clause));
  }
  return reasons_[var];
}

}  // namespace sat
}  // namespace operations_research

#endif  // OR_TOOLS_SAT_SAT_BASE_H_
