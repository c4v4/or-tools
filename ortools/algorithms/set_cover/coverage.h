#ifndef CAV_ORTOOLS_ALGORITHMS_SET_COVER_COVERAGE_H
#define CAV_ORTOOLS_ALGORITHMS_SET_COVER_COVERAGE_H

#include "ortools/algorithms/set_cover_model.h"

namespace operations_research::scp {

// Tracks the coverage of rows by columns.
// This overlaps with `SetCoverInvariant` but is isolated to only provide
// coverage information.
class Coverage {
 public:
  // const vector-like access interface
  Coverage() = default;
  Coverage(size_t num_elements) : coverage_(num_elements, 0) {}
  void resize(size_t num_elements) { coverage_.resize(num_elements, 0); }
  size_t size() const { return coverage_.size(); }
  auto begin() const { return coverage_.begin(); }
  auto end() const { return coverage_.end(); }
  BaseInt operator[](ElementIndex i) const { return coverage_[i]; }

  // Coverage interface
  void UncoverAll() { coverage_.assign(coverage_.size(), 0); }

  // Cover the elementes in the given column and return the number of newly
  // covered elements.
  // NOTE: The number of newly covered elements is returned and not stored.
  // By doing so, when the function gets inlined and `new_covered` is ignored,
  // the compiler can optimize away the `new_covered` computation, obtaining a
  // simpler loop. https://godbolt.org/z/WMsKEb8nG
  BaseInt Cover(const SparseColumn& col) {
    BaseInt new_covered = 0;
    for (ElementIndex i : col) {
      if (coverage_[i] == 0) new_covered++;
      coverage_[i]++;
    }
    return new_covered;
  }

  // Uncover the elements in the given column and return the number of now
  // uncovered elements.
  BaseInt UnCover(const SparseColumn& col) {
    BaseInt uncovered = 0;
    for (ElementIndex i : col) {
      coverage_[i]--;
      if (coverage_[i] == 0) uncovered++;
    }
    return uncovered;
  }

  // Returns true if covering the elements of the given columns does not covern
  // any new element.
  bool IsRedundatCover(const SparseColumn& col) const {
    for (ElementIndex i : col) {
      if (coverage_[i] == 0) return false;
    }
    return true;
  }

  // Returns true if uncovering the elements of the given columns does not leave
  // any element completlly uncovered.
  bool IsRedundantUncover(const SparseColumn& col) const {
    for (ElementIndex i : col) {
      if (coverage_[i] == 1) return false;
    }
    return true;
  }

 private:
  ElementToIntVector coverage_;
};

}  // namespace operations_research::scp

#endif /* CAV_ORTOOLS_ALGORITHMS_SET_COVER_COVERAGE_H */
