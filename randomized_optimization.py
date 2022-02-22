import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

# import mlrose as mh
import numpy as np
import math
from charting import random_hill_chart_lineplot, random_hill_chart_heatmap, chart_lineplot, chart_heatmap
from random import random


def get_four_peaks_problem(length=20):
    threshold_percentage = 0.15
    f_four_peaks = mh.FourPeaks(t_pct=threshold_percentage)

    maximum_fitness_values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36])

    four_peaks_problem = mh.DiscreteOpt(length, f_four_peaks)
    return four_peaks_problem, maximum_fitness_values[length - 1]


def get_knapsack_problem(length=20, max_weight_pct=0.35, max_val=2):

    master_weights = np.array([11, 10, 18, 16, 17, 7, 14, 15, 17, 10, 17, 15, 4, 12, 8, 3, 12, 15, 14, 2])
    master_values = np.array([10, 8, 8, 19, 6, 15, 9, 7, 3, 7, 11, 12, 13, 9, 10, 12, 13, 17, 16, 5])

    maximum_fitness_values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 117])

    weights = master_weights[0:length]
    values = master_values[0:length]

    f_knapsack = mh.Knapsack(weights, values, max_weight_pct=max_weight_pct)
    maximize = True
    if max_val is None:
        max_val = int(sum(weights) * max_weight_pct / min(weights)) + 1
    knapsack_problem = mh.DiscreteOpt(length, f_knapsack, max_val=max_val, maximize=maximize)
    return knapsack_problem, maximum_fitness_values[length - 1]


def create_kcolor_edges(node_count, connection_chance):
    nodes = list(range(node_count))
    max_edge_count = 0
    for i in range(1, node_count):
        max_edge_count += i

    for i in range(1, node_count + 1):
        max_edge_count + 1
    edges = []

    connected_nodes = set()
    for start in range(node_count):
        for end in range(start, node_count):
            if random() < connection_chance:
                edges.append((start, end))
                connected_nodes.add(start)
                connected_nodes.add(end)

    for node in nodes:
        if node not in connected_nodes:
            print(f"{node} not connected")
            edges.append((node, node))

    print(edges)
    print(f"{100* len(edges)/max_edge_count:5.2f} {len(edges)} out of {max_edge_count}")

    return edges


def get_k_colors_problem(max_val=2):
    length = 20

    # 57 out of 190
    edges = [
        (0, 0),
        (0, 1),
        (0, 3),
        (0, 9),
        (0, 16),
        (1, 4),
        (1, 6),
        (1, 7),
        (1, 10),
        (1, 12),
        (2, 4),
        (2, 5),
        (2, 7),
        (2, 10),
        (2, 15),
        (2, 17),
        (2, 18),
        (3, 6),
        (3, 9),
        (3, 11),
        (3, 17),
        (3, 19),
        (4, 6),
        (4, 7),
        (4, 15),
        (5, 12),
        (6, 7),
        (6, 18),
        (7, 7),
        (7, 11),
        (7, 13),
        (7, 14),
        (7, 16),
        (8, 9),
        (8, 12),
        (8, 15),
        (8, 18),
        (9, 11),
        (9, 12),
        (9, 14),
        (9, 17),
        (9, 18),
        (10, 10),
        (10, 13),
        (10, 14),
        (10, 15),
        (10, 17),
        (10, 18),
        (11, 13),
        (11, 15),
        (12, 12),
        (12, 19),
        (13, 13),
        (15, 18),
        (16, 16),
        (16, 17),
        (16, 18),
    ]
    f_max_k_color = mh.MaxKColor(edges)
    maximize = False
    max_fitness_values = 57
    min_fitness_value = 18
    knapsack_problem = mh.DiscreteOpt(length, f_max_k_color, max_val=max_val, maximize=maximize)
    return knapsack_problem, max_fitness_values, min_fitness_value


def get_queens_problem(length=8, max_val=8):
    f_queens = mh.Queens()
    queens_problem = mh.DiscreteOpt(length, f_queens, max_val=max_val, maximize=False)
    maximum_fitness_value = 0
    return queens_problem, maximum_fitness_value


def print_histogram(iteration, values, min_value=0, max_value=None, print_header=False, maximize=True):
    max_value = max_value if max_value is not None else max(values)

    bins, edges = np.histogram(values, range=(min_value, max_value))
    count_str = f"{iteration:3} mean:{values.mean():6.2f}   "
    edge_str = "                  "
    for i, bin_count in enumerate(bins):
        count_str += f" {bin_count:4} "
        edge_str += f" {edges[i + 1]:4.0f} "
    if print_header:
        print(edge_str)
    else:
        print(count_str)

    return bins


def get_max_knapsack_fitness_values(start, end):
    max_fitness_values = [0, 0, 0]
    for length in range(start, end):
        knapsack_problem, max_fitness_value = get_knapsack_problem(length)
        total_iterations = 100
        repetions_per_iteration = 100
        maximize = True
        rh_knapsack_values, max_fitness_value_at_length = random_hill(
            knapsack_problem,
            total_iterations=total_iterations,
            repetions_per_iteration=repetions_per_iteration,
            maximize=maximize,
            max_fitness_value=max_fitness_value,
        )
        max_fitness_values.append(max_fitness_value_at_length)
    print(max_fitness_values)
    return max_fitness_values


def random_hill(
    problem,
    total_iterations=100,
    repetions_per_iteration=1000,
    maximize=True,
    max_fitness_value=None,
    min_fitness_value=0,
):
    max_attempts = 2
    iteration_values = []
    overall_max_state = None
    overall_max_fitness = -100000
    overall_min_state = None
    overall_min_fitness = 100000
    rep_values = np.zeros((repetions_per_iteration))
    print_histogram(0, rep_values, max_value=max_fitness_value, print_header=True)
    for max_iters in range(total_iterations):
        rep_values = np.zeros((repetions_per_iteration))
        for i in range(repetions_per_iteration):
            best_state, best_fitness, _ = mh.random_hill_climb(
                problem, max_attempts=max_attempts, max_iters=max_iters, restarts=max_iters
            )

            if best_fitness > overall_max_fitness:
                overall_max_fitness = best_fitness
                overall_max_state = best_state

            if best_fitness < overall_min_fitness:
                overall_min_fitness = best_fitness
                overall_min_state = best_state

            rep_values[i] = best_fitness

        bin_counts = print_histogram(
            max_iters, rep_values, max_value=max_fitness_value, min_value=min_fitness_value, maximize=maximize
        )

        iteration_values.append([max_iters, rep_values.mean(), rep_values, bin_counts])
    print(f"Max {overall_max_fitness}:{overall_max_state}")
    print(f"Min {overall_min_fitness}:{overall_min_state}")
    return iteration_values, overall_best_fitness


### RANDOM HILL CLIMBING
four_peaks_problem, max_fitness_value = get_four_peaks_problem()

repetions_per_iteration = 100
total_iterations = 50
four_peaks_random_hill_values, overall_best_fitness = random_hill(
    four_peaks_problem,
    total_iterations=total_iterations,
    max_fitness_value=max_fitness_value,
    repetions_per_iteration=repetions_per_iteration,
)

rh_four_peaks_values, overall_best_fitness = iterate_algorithm(
    four_peaks_problem,
    "rh",
    total_iterations=total_iterations,
    max_fitness_value=max_fitness_value,
    repetions_per_iteration=repetions_per_iteration,
)


random_hill_chart_lineplot(rh_four_peaks_values, title="Four Peaks Problem")
random_hill_chart_heatmap(rh_four_peaks_values, title="Four Peaks Problem")


knapsack_problem, max_fitness_value = get_knapsack_problem()

total_iterations = 50
repetions_per_iteration = 100
maximize = True
rh_knapsack_values, max_fitness_value_at_length = random_hill(
    knapsack_problem,
    total_iterations=total_iterations,
    repetions_per_iteration=repetions_per_iteration,
    maximize=maximize,
    max_fitness_value=max_fitness_value,
)

random_hill_chart_lineplot(rh_knapsack_values, title="Knapsack")
random_hill_chart_heatmap(rh_knapsack_values, title="Knapsack")

color_problem, max_fitness_value, min_fitness_value = get_k_colors_problem()
total_iterations = 50
repetions_per_iteration = 100
maximize = False
rh_color_values, max_fitness_value_at_length = random_hill(
    color_problem,
    total_iterations=total_iterations,
    repetions_per_iteration=repetions_per_iteration,
    maximize=maximize,
    max_fitness_value=max_fitness_value,
    min_fitness_value=min_fitness_value,
)

random_hill_chart_lineplot(rh_color_values, title="2 Color Problem", maximize=maximize)
random_hill_chart_heatmap(rh_color_values, title="2 Color Problem", maximize=maximize)


## SIMULATED ANNEALING
def optimize_decay_schedule(
    problem,
    total_repetitions=3000,
    max_iters=50,
    start_temp=0.3,
    end_temp=None,
    temp_step=None,
    max_iters_start=50,
    max_iters_stop=None,
    iters_step=None,
):

    random_state = 0

    if end_temp is not None:
        init_temp_range = np.around(np.arange(start_temp, end_temp, temp_step), 2)
    else:
        init_temp_range = np.array([start_temp])

    if max_iters_stop is not None:
        max_iters_range = list(range(max_iters_start, max_iters_stop, iters_step))
    else:
        max_iters_range = [max_iters_start]

    for init_temp in init_temp_range:
        for max_iters in max_iters_range:
            fitness_scores = np.zeros((total_repetitions))
            fp_decay_schedule = mh.GeomDecay(init_temp=init_temp, decay=0.99, min_temp=0.0001)
            for i in range(total_repetitions):
                best_state, best_fitness, best_curve = mh.simulated_annealing(
                    problem,
                    fp_decay_schedule,
                    max_attempts=max_iters,
                    curve=True,
                    random_state=random_state,
                    max_iters=max_iters,
                )
                fitness_scores[i] = best_fitness
            print(
                f"max_iters:{max_iters} init_temp:{init_temp:0.2f}  mean:{fitness_scores.mean():7.4f} max:{fitness_scores.max()} min:{fitness_scores.min()}"
            )


def iterate_algorithm(
    problem,
    algorithm_type,
    decay_schedule=None,
    total_iterations=100,
    repetions_per_iteration=1000,
    maximize=True,
    max_fitness_value=None,
    min_fitness_value=0,
):
    if algorithm_type not in ["rh", "sa"]:
        print(f"Unsupported Algorithm type {algorithm_type}")
        return
    max_attempts = 2
    iteration_values = []
    overall_max_state = None
    overall_max_fitness = -100000
    overall_min_state = None
    overall_min_fitness = 100000
    rep_values = np.zeros((repetions_per_iteration))
    print_histogram(0, rep_values, max_value=max_fitness_value, print_header=True)
    for max_iters in range(total_iterations):
        rep_values = np.zeros((repetions_per_iteration))
        for i in range(repetions_per_iteration):
            if algorithm_type == "rh":
                best_state, best_fitness, _ = mh.random_hill_climb(
                    problem, max_attempts=max_attempts, max_iters=max_iters, restarts=max_iters
                )
            elif algorithm_type == "sa":
                best_state, best_fitness, _ = mh.simulated_annealing(
                    problem,
                    decay_schedule,
                    max_attempts=max_iters,
                    max_iters=max_iters,
                )
            else:
                raise Exception(f"Unsupported Algorithm Type {algorithm_type}")

            if best_fitness > overall_max_fitness:
                overall_max_fitness = best_fitness
                overall_max_state = best_state

            if best_fitness < overall_min_fitness:
                overall_min_fitness = best_fitness
                overall_min_state = best_state

            rep_values[i] = best_fitness

        bin_counts = print_histogram(
            max_iters, rep_values, max_value=max_fitness_value, min_value=min_fitness_value, maximize=maximize
        )

        iteration_values.append([max_iters, rep_values.mean(), rep_values, bin_counts])
    print(f"Max {overall_max_fitness}:{overall_max_state}")
    print(f"Min {overall_min_fitness}:{overall_min_state}")
    return iteration_values, overall_best_fitness


four_peaks_problem, max_fitness_value = get_four_peaks_problem()

fp_decay_schedule = mh.GeomDecay(init_temp=0.35, decay=0.99, min_temp=0.0001)
total_iterations = 100
repetions_per_iteration = 100
sa_four_peaks_values, max_fitness_value_at_length = iterate_algorithm(
    four_peaks_problem, "sa", fp_decay_schedule, max_fitness_value=max_fitness_value, total_iterations=total_iterations
)
chart_lineplot(sa_four_peaks_values, title="Four Peaks", suptitle="Simulated Annealing")
chart_heatmap(sa_four_peaks_values, title="Four Peaks", suptitle="Simulated Annealing")


optimize_decay_schedule(four_peaks_problem, max_iters_start=45, max_iters_stop=55)


optimize_decay_schedule(
    four_peaks_problem,
    max_iters_start=30,
    max_iters_stop=100,
    iters_step=10,
    start_temp=0.35,
)

optimize_decay_schedule(
    four_peaks_problem,
    max_iters_start=70,
    start_temp=0.4,
    end_temp=0.1,
    temp_step=-0.05,
)


T = 0.30
for t in range(1000):
    pr = fp_decay_schedule.evaluate(t)
    T_t = T * 0.99**t
    pr_1 = math.e ** (-1 / T_t)
    pr_10 = math.e ** (-10 / T_t)
    print(f"{t:3}:{pr:6.6f}   T:{T_t:0.6f}  {pr_1:0.6f} {pr_10:0.6f}")


for T in range(1, 10):
    for d in range(-1, -10, -1):
        pr = math.e ** (d / T)
        print(f"d:{d} T:{T}  {pr:6.3f}")
