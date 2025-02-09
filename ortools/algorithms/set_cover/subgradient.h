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

#ifndef CAV_ORTOOLS_ALGORITHMS_SET_COVER_SUBGRADIENT_H
#define CAV_ORTOOLS_ALGORITHMS_SET_COVER_SUBGRADIENT_H

#include "core_model.h"

namespace operations_research::scp {

// The Subgradient class optimizes the Lagrangian dual of the Set Cover Problem.
// It follows the method described in [1], maximizing the Lagrangian bound using
// a minimal-cover subgradient to stabilize and speed up convergence.
class Subgradient {
 public:
  // Executes the subgradient algorithm and returns the best (real) lower bound
  // found.
  //    - `core_model` is a restricted SCP model updated during the algorithm.
  //    After execution, core_model contains a refined set of high-quality
  //    columns (subsets). The returned lower bound is the best bound returned
  //    by the core_model after every update.
  //    - `upper_bound` is a real or estimated upper bound for the solution of
  //    the Lagraingian dual. Note that once the upper bound is reached, the
  //    algorithm stops.
  //    - `init_multipliers` is the initial set of Lagrangian multipliers.
  // Use the getters to access data from the last subgradient execution.
  Cost ComputeLowerBound(CoreModel& core_model, Cost upper_bound,
                         const ElementCostVector& init_multipliers);

  // The final step size at the end of the subgradient, it is used to initialize
  // the heuristic phase of the CFT.
  double GetFinalStepSize() const { return step_size; }

 private:
  double step_size;
};

}  // namespace operations_research::scp

#endif /* CAV_ORTOOLS_ALGORITHMS_SET_COVER_SUBGRADIENT_H */
