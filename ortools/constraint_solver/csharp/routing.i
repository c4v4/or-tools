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

// TODO(user): Refactor this file to adhere to the SWIG style guide.

%typemap(csimports) SWIGTYPE %{
using System;
using System.Runtime.InteropServices;
using System.Collections;
using System.Collections.Generic;
%}

%include "ortools/base/base.i"
%template(IntBoolPair) std::pair<int, bool>;
%include "ortools/constraint_solver/csharp/constraint_solver.i"
%import "ortools/util/csharp/sorted_interval_list.i"  // Domain

%include "ortools/constraint_solver/csharp/routing_index_manager.i"

// We need to forward-declare the proto here, so that PROTO_INPUT involving it
// works correctly. The order matters very much: this declaration needs to be
// before the %{ #include ".../routing.h" %}.
namespace operations_research {
class RoutingModelParameters;
class RoutingSearchParameters;
class RoutingSearchStatus;
}  // namespace operations_research

// Include the file we want to wrap a first time.
%{
#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/constraint_solver/routing_parameters.h"
#include "ortools/constraint_solver/routing_parameters.pb.h"
#include "ortools/constraint_solver/routing_types.h"
%}

%module(directors="1") operations_research;

// RoutingModel methods.
DEFINE_INDEX_TYPE_TYPEDEF(
    operations_research::RoutingCostClassIndex,
    operations_research::RoutingModel::CostClassIndex);
DEFINE_INDEX_TYPE_TYPEDEF(
    operations_research::RoutingDimensionIndex,
    operations_research::RoutingModel::DimensionIndex);
DEFINE_INDEX_TYPE_TYPEDEF(
    operations_research::RoutingDisjunctionIndex,
    operations_research::RoutingModel::DisjunctionIndex);
DEFINE_INDEX_TYPE_TYPEDEF(
    operations_research::RoutingVehicleClassIndex,
    operations_research::RoutingModel::VehicleClassIndex);
DEFINE_INDEX_TYPE_TYPEDEF(
    operations_research::RoutingResourceClassIndex,
    operations_research::RoutingModel::ResourceClassIndex);

namespace operations_research {

// GlobalVehicleBreaksConstraint
%unignore GlobalVehicleBreaksConstraint;
%typemap(csimports) GlobalVehicleBreaksConstraint %{
%}

// PathsMetadata
%unignore PathsMetadata;

// RoutingDimension
%unignore RoutingDimension;
%typemap(csimports) RoutingDimension %{
using System;
using System.Collections.Generic;
%}
%typemap(cscode) RoutingDimension %{
  // Keep reference to delegate to avoid GC to collect them early.
  private List<IntIntToLong> limitCallbacks;
  private IntIntToLong StoreIntIntToLong(IntIntToLong limit) {
    if (limitCallbacks == null)
      limitCallbacks = new List<IntIntToLong>();
    limitCallbacks.Add(limit);
    return limit;
  }

  private List<LongLongToLong> groupDelayCallbacks;
  private LongLongToLong StoreLongLongToLong(LongLongToLong groupDelay) {
    if (groupDelayCallbacks == null)
      groupDelayCallbacks = new List<LongLongToLong>();
    groupDelayCallbacks.Add(groupDelay);
    return groupDelay;
  }
%}
%ignore RoutingDimension::GetBreakDistanceDurationOfVehicle;

// RoutingModel
%unignore RoutingModel;
%typemap(csimports) RoutingModel %{
using System;
using System.Collections.Generic;
using Domain = Google.OrTools.Util.Domain;
%}
%typemap(cscode) RoutingModel %{
  // Keep reference to delegate to avoid GC to collect them early.
  private List<LongToLong> unaryTransitCallbacks;
  private LongToLong StoreLongToLong(LongToLong c) {
    if (unaryTransitCallbacks == null)
      unaryTransitCallbacks = new List<LongToLong>();
    unaryTransitCallbacks.Add(c);
    return c;
  }

  private List<LongLongToLong> transitCallbacks;
  private LongLongToLong StoreLongLongToLong(LongLongToLong c) {
    if (transitCallbacks == null)
      transitCallbacks = new List<LongLongToLong>();
    transitCallbacks.Add(c);
    return c;
  }

  private List<VoidToVoid> solutionCallbacks;
  private VoidToVoid StoreVoidToVoid(VoidToVoid c) {
    if (solutionCallbacks == null)
      solutionCallbacks = new List<VoidToVoid>();
    solutionCallbacks.Add(c);
    return c;
  }
%}
%rename("GetStatus") RoutingModel::status;
// Ignored:
%ignore RoutingModel::AddDimensionDependentDimensionWithVehicleCapacity;

%unignore RoutingModel::RegisterUnaryTransitVector;
%unignore RoutingModel::RegisterTransitMatrix;

%unignore RoutingModel::AddVectorDimension;
%unignore RoutingModel::AddMatrixDimension;

%ignore RoutingModel::AddSameVehicleRequiredTypeAlternatives;
%ignore RoutingModel::GetAllDimensionNames;
%ignore RoutingModel::GetAutomaticFirstSolutionStrategy;
%ignore RoutingModel::GetDeliveryIndexPairs;
%ignore RoutingModel::GetDimensions;
%ignore RoutingModel::GetDimensionsWithSoftAndSpanCosts;
%ignore RoutingModel::GetDimensionsWithSoftOrSpanCosts;
%ignore RoutingModel::GetGlobalDimensionCumulOptimizers;
%ignore RoutingModel::GetHardTypeIncompatibilitiesOfType;
%ignore RoutingModel::GetLocalDimensionCumulMPOptimizers;
%ignore RoutingModel::GetLocalDimensionCumulOptimizers;
%ignore RoutingModel::GetMutableGlobalCumulOptimizer;
%ignore RoutingModel::GetMutableLocalCumulOptimizer;
%ignore RoutingModel::GetMutableLocalCumulMPOptimizer;
%ignore RoutingModel::GetPerfectBinaryDisjunctions;
%ignore RoutingModel::GetPickupIndexPairs;
%ignore RoutingModel::HasTypeRegulations;
%ignore RoutingModel::MakeStateDependentTransit;
%ignore RoutingModel::PackCumulsOfOptimizerDimensionsFromAssignment;
%ignore RoutingModel::RegisterStateDependentTransitCallback;
%ignore RoutingModel::RemainingTime;
%ignore RoutingModel::StateDependentTransitCallback;
%ignore RoutingModel::SolveWithParameters(
    const RoutingSearchParameters& search_parameters,
    std::vector<const Assignment*>* solutions);
%ignore RoutingModel::SolveFromAssignmentWithParameters(
      const Assignment* assignment,
      const RoutingSearchParameters& search_parameters,
      std::vector<const Assignment*>* solutions);
%ignore RoutingModel::TransitCallback;
%ignore RoutingModel::UnaryTransitCallbackOrNull;

// RoutingModelVisitor
%unignore RoutingModelVisitor;
%typemap(csimports) RoutingModelVisitor %{
%}

// SimpleBoundCosts
%unignore BoundCost;
%unignore SimpleBoundCosts;
%rename("GetBoundCost") SimpleBoundCosts::bound_cost;
%rename("GetSize") SimpleBoundCosts::Size;

// TypeRegulationsConstraint
%unignore TypeRegulationsConstraint;
%typemap(csimports) TypeRegulationsConstraint %{
%}

// TypeRegulationsChecker
%unignore TypeRegulationsChecker;
%ignore TypeRegulationsChecker::CheckVehicle;

}  // namespace operations_research

%rename("%(camelcase)s", %$isfunction) "";

// Protobuf support
PROTO_INPUT(operations_research::RoutingSearchParameters,
            Google.OrTools.ConstraintSolver.RoutingSearchParameters,
            search_parameters)
PROTO_INPUT(operations_research::RoutingModelParameters,
            Google.OrTools.ConstraintSolver.RoutingModelParameters,
            parameters)
PROTO2_RETURN(operations_research::RoutingSearchParameters,
              Google.OrTools.ConstraintSolver.RoutingSearchParameters)
PROTO2_RETURN(operations_research::RoutingModelParameters,
              Google.OrTools.ConstraintSolver.RoutingModelParameters)
PROTO_ENUM_RETURN(operations_research::RoutingSearchStatus::Value,
                  Google.OrTools.ConstraintSolver.RoutingSearchStatus.Types.Value)

// Wrap routing includes
// TODO(user): Replace with %ignoreall/%unignoreall
//swiglint: disable include-h-allglobals
%include "ortools/constraint_solver/routing_parameters.h"
%include "ortools/constraint_solver/routing.h"
