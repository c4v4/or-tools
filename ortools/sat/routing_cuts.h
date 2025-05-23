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

#ifndef OR_TOOLS_SAT_ROUTING_CUTS_H_
#define OR_TOOLS_SAT_ROUTING_CUTS_H_

#include <stdint.h>

#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cuts.h"
#include "ortools/sat/integer.h"
#include "ortools/sat/integer_base.h"
#include "ortools/sat/model.h"
#include "ortools/sat/precedences.h"
#include "ortools/sat/sat_base.h"

namespace operations_research {
namespace sat {

// Helper to compute the minimum flow going out of a subset of nodes, for a
// given RoutesConstraint.
class MinOutgoingFlowHelper {
 public:
  MinOutgoingFlowHelper(int num_nodes, const std::vector<int>& tails,
                        const std::vector<int>& heads,
                        const std::vector<Literal>& literals, Model* model);

  // Returns the minimum flow going out of `subset`, based on a conservative
  // estimate of the maximum number of nodes of a feasible path inside this
  // subset. `subset` must not be empty and must not contain the depot (node 0).
  // Paths are approximated by their length and their last node, and can thus
  // contain cycles. The complexity is O(subset.size() ^ 3).
  int ComputeMinOutgoingFlow(absl::Span<const int> subset);

  // Same as above, but uses less conservative estimates (paths are approximated
  // by their set of nodes and their last node -- hence they can't contain
  // cycles). The complexity is O(2 ^ subset.size()).
  int ComputeTightMinOutgoingFlow(absl::Span<const int> subset);

 private:
  // Returns the minimum flow going out of a subset of size `subset_size`,
  // assuming that the longest feasible path inside this subset has
  // `longest_path_length` nodes and that there are at most `max_longest_paths`
  // such paths.
  int GetMinOutgoingFlow(int subset_size, int longest_path_length,
                         int max_longest_paths);

  const std::vector<int>& tails_;
  const std::vector<int>& heads_;
  const std::vector<Literal>& literals_;
  const BinaryRelationRepository& binary_relation_repository_;
  const Trail& trail_;
  const IntegerTrail& integer_trail_;

  // Temporary data used by ComputeMinOutgoingFlow(). Always contain default
  // values, except while ComputeMinOutgoingFlow() is running.
  // ComputeMinOutgoingFlow() computes, for each i in [0, |subset|), whether
  // each node n in the subset could appear at position i in a feasible path.
  // It computes whether n can appear at position i based on which nodes can
  // appear at position i-1, and based on the arc literals and some linear
  // constraints they enforce. To save memory, it only stores data about two
  // consecutive positions at a time: a "current" position i, and a "next"
  // position i+1.

  std::vector<bool> in_subset_;
  std::vector<int> index_in_subset_;
  // For each node n, the indices (in tails_, heads_) of the m->n and n->m arcs
  // inside the subset (self arcs excepted).
  std::vector<std::vector<int>> incoming_arc_indices_;
  std::vector<std::vector<int>> outgoing_arc_indices_;
  // For each node n, whether it can appear at the current and next position in
  // a feasible path.
  std::vector<bool> reachable_;
  std::vector<bool> next_reachable_;
  // For each node n, the lower bound of each variable (appearing in a linear
  // constraint enforced by some incoming arc literal), if n appears at the
  // current and next position in a feasible path. Variables not appearing in
  // these maps have no tighter lower bound that the one from the IntegerTrail
  // (at decision level 0).
  std::vector<absl::flat_hash_map<IntegerVariable, IntegerValue>>
      node_var_lower_bounds_;
  std::vector<absl::flat_hash_map<IntegerVariable, IntegerValue>>
      next_node_var_lower_bounds_;
};

// Given a graph with nodes in [0, num_nodes) and a set of arcs (the order is
// important), this will:
//   - Start with each nodes in separate "subsets".
//   - Consider the arc in order, and each time one connects two separate
//     subsets, merge the two subsets into a new one.
//   - Stops when there is only 2 subset left.
//   - Output all subsets generated this way (at most 2 * num_nodes). The
//     subsets spans will point in the subset_data vector (which will be of size
//     exactly num_nodes).
//
// This is an heuristic to generate interesting cuts for TSP or other graph
// based constraints. We roughly follow the algorithm described in section 6 of
// "The Traveling Salesman Problem, A computational Study", David L. Applegate,
// Robert E. Bixby, Vasek Chvatal, William J. Cook.
//
// Note that this is mainly a "symmetric" case algo, but it does still work for
// the asymmetric case.
void GenerateInterestingSubsets(int num_nodes,
                                absl::Span<const std::pair<int, int>> arcs,
                                int stop_at_num_components,
                                std::vector<int>* subset_data,
                                std::vector<absl::Span<const int>>* subsets);

// Given a set of rooted tree on n nodes represented by the parent vector,
// returns the n sets of nodes corresponding to all the possible subtree. Note
// that the output memory is just n as all subset will point into the same
// vector.
//
// This assumes no cycles, otherwise it will not crash but the result will not
// be correct.
//
// In the TSP context, if the tree is a Gomory-Hu cut tree, this will returns
// a set of "min-cut" that contains a min-cut for all node pairs.
//
// TODO(user): This also allocate O(n) memory internally, we could reuse it from
// call to call if needed.
void ExtractAllSubsetsFromForest(
    absl::Span<const int> parent, std::vector<int>* subset_data,
    std::vector<absl::Span<const int>>* subsets,
    int node_limit = std::numeric_limits<int>::max());

// In the routing context, we usually always have lp_value in [0, 1] and only
// looks at arcs with a lp_value that is not too close to zero.
struct ArcWithLpValue {
  int tail;
  int head;
  double lp_value;

  bool operator==(const ArcWithLpValue& o) const {
    return tail == o.tail && head == o.head && lp_value == o.lp_value;
  }
};

// Regroups and sum the lp values on duplicate arcs or reversed arcs
// (tail->head) and (head->tail). As a side effect, we will always
// have tail <= head.
void SymmetrizeArcs(std::vector<ArcWithLpValue>* arcs);

// Given a set of arcs on a directed graph with n nodes (in [0, num_nodes)),
// returns a "parent" vector of size n encoding a rooted Gomory-Hu tree.
//
// Note that usually each edge in the tree is attached a max-flow value (its
// weight), but we don't need it here. It can be added if needed. This tree as
// the property that for all (s, t) pair of nodes, if you take the minimum
// weight edge on the path from s to t and split the tree in two, then this is a
// min-cut for that pair.
//
// IMPORTANT: This algorithm currently "symmetrize" the graph, so we will
// actually have all the min-cuts that minimize sum incoming + sum outgoing lp
// values. The algo do not work as is on an asymmetric graph. Note however that
// because of flow conservation, our outgoing lp values should be the same as
// our incoming one on a circuit/route constraint.
//
// We use a simple implementation described in "Very Simple Methods for All
// Pairs Network Flow Analysis", Dan Gusfield, 1990,
// https://ranger.uta.edu/~weems/NOTES5311/LAB/LAB2SPR21/gusfield.huGomory.pdf
std::vector<int> ComputeGomoryHuTree(
    int num_nodes, absl::Span<const ArcWithLpValue> relevant_arcs);

// Cut generator for the circuit constraint, where in any feasible solution, the
// arcs that are present (variable at 1) must form a circuit through all the
// nodes of the graph. Self arc are forbidden in this case.
//
// In more generality, this currently enforce the resulting graph to be strongly
// connected. Note that we already assume basic constraint to be in the lp, so
// we do not add any cuts for components of size 1.
CutGenerator CreateStronglyConnectedGraphCutGenerator(
    int num_nodes, absl::Span<const int> tails, absl::Span<const int> heads,
    absl::Span<const Literal> literals, Model* model);

// Almost the same as CreateStronglyConnectedGraphCutGenerator() but for each
// components, computes the demand needed to serves it, and depending on whether
// it contains the depot (node zero) or not, compute the minimum number of
// vehicle that needs to cross the component border.
CutGenerator CreateCVRPCutGenerator(int num_nodes, absl::Span<const int> tails,
                                    absl::Span<const int> heads,
                                    absl::Span<const Literal> literals,
                                    absl::Span<const int64_t> demands,
                                    int64_t capacity, Model* model);

// Try to find a subset where the current LP capacity of the outgoing or
// incoming arc is not enough to satisfy the demands.
//
// We support the special value -1 for tail or head that means that the arc
// comes from (or is going to) outside the nodes in [0, num_nodes). Such arc
// must still have a capacity assigned to it.
//
// TODO(user): Support general linear expression for capacities.
// TODO(user): Some model applies the same capacity to both an arc and its
// reverse. Also support this case.
CutGenerator CreateFlowCutGenerator(
    int num_nodes, const std::vector<int>& tails, const std::vector<int>& heads,
    const std::vector<AffineExpression>& arc_capacities,
    std::function<void(const std::vector<bool>& in_subset,
                       IntegerValue* min_incoming_flow,
                       IntegerValue* min_outgoing_flow)>
        get_flows,
    Model* model);

}  // namespace sat
}  // namespace operations_research

#endif  // OR_TOOLS_SAT_ROUTING_CUTS_H_
