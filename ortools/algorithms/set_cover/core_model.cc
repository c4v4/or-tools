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
namespace operations_research::scp {

// TODO(c4v4): implement the pricing + column selection + mappings
std::tuple<Cost, bool> CoreFromFullModel::UpdateModelAndLowerBound(
    ElementCostVector& multipliers, Cost lower_bound, Cost upper_bound) {
  return std::make_tuple(lower_bound, false);
}

}  // namespace operations_research::scp