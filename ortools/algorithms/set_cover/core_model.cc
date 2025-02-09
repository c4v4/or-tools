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

namespace {

// TODO(c4v4): implement the initial tentative core model from [1]
void BuildFirstCoreModel(const Model& full_model, Model& core_model) {
  // 1. Clear core_model from previoys data
  // 2. Clear core->full mappings

  // 3. Select the first n columns of each row (there might be duplicates)
  // 4. Sort the column list to detect duplicates

  // 5. Add columns to core model
  // 6. Create core->full mappings
  // 7. Fill core_model RowView

  // Stub
  core_model = full_model;
}

}  // namespace

CoreFromFullModel::CoreFromFullModel(const Model* full_model)
    : full_model_(full_model) {
  BuildFirstCoreModel(*full_model_, this->core_model());
}

// TODO(c4v4): implement the pricing + column selection + mappings
Cost CoreFromFullModel::UpdateCoreModel(Cost upper_bound,
                                        ElementCostVector& multipliers) {
  return lower_bound();
}

}  // namespace operations_research::scp