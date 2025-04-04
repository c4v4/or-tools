#!/usr/bin/env python3
# Copyright 2010-2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single machine jobshop with setup times, release dates and due dates."""

from typing import Sequence
from absl import app
from absl import flags
from google.protobuf import text_format
from ortools.sat.python import cp_model

# ----------------------------------------------------------------------------
# Command line arguments.
_OUTPUT_PROTO = flags.DEFINE_string(
    "output_proto", "", "Output file to write the cp_model proto to."
)
_PARAMS = flags.DEFINE_string(
    "params",
    "num_search_workers:16,log_search_progress:true,max_time_in_seconds:45",
    "Sat solver parameters.",
)
_PREPROCESS = flags.DEFINE_bool(
    "--preprocess_times", True, "Preprocess setup times and durations"
)


# ----------------------------------------------------------------------------
# Intermediate solution printer
class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        """Called at each new solution."""
        print(
            f"Solution {self.__solution_count}, time = {self.wall_time} s,"
            f" objective = {self.objective_value}"
        )


def single_machine_scheduling():
    """Solves a complex single machine jobshop scheduling problem."""

    parameters = _PARAMS.value
    output_proto_file = _OUTPUT_PROTO.value

    # ----------------------------------------------------------------------------
    # Data.

    job_durations = [
        2546,
        8589,
        5953,
        3710,
        3630,
        3016,
        4148,
        8706,
        1604,
        5502,
        9983,
        6209,
        9920,
        7860,
        2176,
    ]

    setup_times = [
        [
            3559,
            1638,
            2000,
            3676,
            2741,
            2439,
            2406,
            1526,
            1600,
            3356,
            4324,
            1923,
            3663,
            4103,
            2215,
        ],
        [
            1442,
            3010,
            1641,
            4490,
            2060,
            2143,
            3376,
            3891,
            3513,
            2855,
            2653,
            1471,
            2257,
            1186,
            2354,
        ],
        [
            1728,
            3583,
            3243,
            4080,
            2191,
            3644,
            4023,
            3510,
            2135,
            1346,
            1410,
            3565,
            3181,
            1126,
            4169,
        ],
        [
            1291,
            1703,
            3103,
            4001,
            1712,
            1137,
            3341,
            3485,
            2557,
            2435,
            1972,
            1986,
            1522,
            4734,
            2520,
        ],
        [
            4134,
            2200,
            1502,
            3995,
            1277,
            1808,
            1020,
            2078,
            2999,
            1605,
            1697,
            2323,
            2268,
            2288,
            4856,
        ],
        [
            4974,
            2480,
            2492,
            4088,
            2587,
            4652,
            1478,
            3942,
            1222,
            3305,
            1206,
            1024,
            2605,
            3080,
            3516,
        ],
        [
            1903,
            2584,
            2104,
            1609,
            4745,
            2691,
            1539,
            2544,
            2499,
            2074,
            4793,
            1756,
            2190,
            1298,
            2605,
        ],
        [
            1407,
            2536,
            2296,
            1769,
            1449,
            3386,
            3046,
            1180,
            4132,
            4783,
            3386,
            3429,
            2450,
            3376,
            3719,
        ],
        [
            3026,
            1637,
            3628,
            3096,
            1498,
            4947,
            1912,
            3703,
            4107,
            4730,
            1805,
            2189,
            1789,
            1985,
            3586,
        ],
        [
            3940,
            1342,
            1601,
            2737,
            1748,
            3771,
            4052,
            1619,
            2558,
            3782,
            4383,
            3451,
            4904,
            1108,
            1750,
        ],
        [
            1348,
            3162,
            1507,
            3936,
            1453,
            2953,
            4182,
            2968,
            3134,
            1042,
            3175,
            2805,
            4901,
            1735,
            1654,
        ],
        [
            1099,
            1711,
            1245,
            1067,
            4343,
            3407,
            1108,
            1784,
            4803,
            2342,
            3377,
            2037,
            3563,
            1621,
            2840,
        ],
        [
            2573,
            4222,
            3164,
            2563,
            3231,
            4731,
            2395,
            1033,
            4795,
            3288,
            2335,
            4935,
            4066,
            1440,
            4979,
        ],
        [
            3321,
            1666,
            3573,
            2377,
            4649,
            4600,
            1065,
            2475,
            3658,
            3374,
            1138,
            4367,
            4728,
            3032,
            2198,
        ],
        [
            2986,
            1180,
            4095,
            3132,
            3987,
            3880,
            3526,
            1460,
            4885,
            3827,
            4945,
            4419,
            3486,
            3805,
            3804,
        ],
        [
            4163,
            3441,
            1217,
            2941,
            1210,
            3794,
            1779,
            1904,
            4255,
            4967,
            4003,
            3873,
            1002,
            2055,
            4295,
        ],
    ]

    due_dates = [
        -1,
        -1,
        28569,
        -1,
        98104,
        27644,
        55274,
        57364,
        -1,
        -1,
        60875,
        96637,
        77888,
        -1,
        -1,
    ]
    release_dates = [0, 0, 0, 0, 19380, 0, 0, 48657, 0, 27932, 0, 0, 24876, 0, 0]

    precedences = [(0, 2), (1, 2)]

    # ----------------------------------------------------------------------------
    # Helper data.
    num_jobs = len(job_durations)
    all_jobs = range(num_jobs)

    # ----------------------------------------------------------------------------
    # Preprocess.
    if _PREPROCESS.value:
        for job_id in all_jobs:
            min_incoming_setup = min(
                setup_times[j][job_id] for j in range(num_jobs + 1)
            )
            if release_dates[job_id] != 0:
                min_incoming_setup = min(min_incoming_setup, release_dates[job_id])
            if min_incoming_setup == 0:
                continue

            print(f"job {job_id} has a min incoming setup of {min_incoming_setup}")
            # We can transfer some setup times to the duration of the job.
            job_durations[job_id] += min_incoming_setup
            # Decrease corresponding incoming setup times.
            for j in range(num_jobs + 1):
                setup_times[j][job_id] -= min_incoming_setup
            # Adjust release dates if needed.
            if release_dates[job_id] != 0:
                release_dates[job_id] -= min_incoming_setup

    # ----------------------------------------------------------------------------
    # Model.
    model = cp_model.CpModel()

    # ----------------------------------------------------------------------------
    # Compute a maximum makespan greedily.
    horizon = sum(job_durations) + sum(
        max(setup_times[i][j] for i in range(num_jobs + 1)) for j in range(num_jobs)
    )
    print(f"Greedy horizon = {horizon}")

    # ----------------------------------------------------------------------------
    # Global storage of variables.
    intervals = []
    starts = []
    ends = []

    # ----------------------------------------------------------------------------
    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        duration = job_durations[job_id]
        release_date = release_dates[job_id]
        due_date = due_dates[job_id] if due_dates[job_id] != -1 else horizon
        print(
            f"job {job_id:2}: start = {release_date:5}, duration = {duration:4},"
            f" end = {due_date:6}"
        )
        name_suffix = f"_{job_id}"
        start = model.new_int_var(release_date, due_date, "s" + name_suffix)
        end = model.new_int_var(release_date, due_date, "e" + name_suffix)
        interval = model.new_interval_var(start, duration, end, "i" + name_suffix)
        starts.append(start)
        ends.append(end)
        intervals.append(interval)

    # No overlap constraint.
    model.add_no_overlap(intervals)

    # ----------------------------------------------------------------------------
    # Transition times using a circuit constraint.
    arcs = []
    for i in all_jobs:
        # Initial arc from the dummy node (0) to a task.
        start_lit = model.new_bool_var("")
        arcs.append((0, i + 1, start_lit))
        # If this task is the first, set to minimum starting time.
        min_start_time = max(release_dates[i], setup_times[0][i])
        model.add(starts[i] == min_start_time).only_enforce_if(start_lit)
        # Final arc from an arc to the dummy node.
        arcs.append((i + 1, 0, model.new_bool_var("")))

        for j in all_jobs:
            if i == j:
                continue

            lit = model.new_bool_var(f"{j} follows {i}")
            arcs.append((i + 1, j + 1, lit))

            # We add the reified precedence to link the literal with the times of the
            # two tasks.
            # If release_dates[j] == 0, we can strenghten this precedence into an
            # equality as we are minimizing the makespan.
            if release_dates[j] == 0:
                model.add(starts[j] == ends[i] + setup_times[i + 1][j]).only_enforce_if(
                    lit
                )
            else:
                model.add(starts[j] >= ends[i] + setup_times[i + 1][j]).only_enforce_if(
                    lit
                )

    model.add_circuit(arcs)

    # ----------------------------------------------------------------------------
    # Precedences.
    for before, after in precedences:
        print(f"job {after} is after job {before}")
        model.add(ends[before] <= starts[after])

    # ----------------------------------------------------------------------------
    # Objective.
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)

    # ----------------------------------------------------------------------------
    # Write problem to file.
    if output_proto_file:
        print(f"Writing proto to {output_proto_file}")
        with open(output_proto_file, "w") as text_file:
            text_file.write(str(model))

    # ----------------------------------------------------------------------------
    # Solve.
    solver = cp_model.CpSolver()
    if parameters:
        text_format.Parse(parameters, solver.parameters)
    solution_printer = SolutionPrinter()
    solver.best_bound_callback = lambda a: print(f"New objective lower bound: {a}")
    solver.solve(model, solution_printer)
    for job_id in all_jobs:
        print(
            f"job {job_id} starts at {solver.value(starts[job_id])} end ends at"
            f" {solver.value(ends[job_id])}"
        )


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    single_machine_scheduling()


if __name__ == "__main__":
    app.run(main)
