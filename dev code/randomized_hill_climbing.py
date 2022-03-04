from operator import length_hint
import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose
import numpy as np
import math
from charting import (
    random_hill_chart,
    random_hill_chart_boxplot,
    random_hill_chart_seaborn_boxplot,
    random_hill_chart_lineplot,
    random_hill_chart_heatmap,
)


for i in range(10):
    threshold_percentage = i * 0.1
    f_four_peaks = mlrose.FourPeaks(t_pct=threshold_percentage)

    a = np.array([])

    f_four_peaks.evaluate(a)


threshold_percentage = 0.15
f_four_peaks = mlrose.FourPeaks(t_pct=threshold_percentage)
for a in range(2):
    for b in range(2):
        for c in range(2):
            for d in range(2):
                for e in range(2):
                    for f in range(2):
                        score = f_four_peaks.evaluate([a, b, c, d, e, f])
                        print(f"{a},{b},{c},{d},{e},{f}: {score}")


threshold_percentage = 0.15
f_four_peaks = mlrose.FourPeaks(t_pct=threshold_percentage)
length = 20
four_peaks_problem = mlrose.DiscreteOpt(length, f_four_peaks)
restarts = 5
max_attempts = 10
best_state, best_fitness, learning_curve = mlrose.random_hill_climb(
    four_peaks_problem, max_attempts=max_attempts, restarts=restarts, curve=True
)
print(f"{best_state}:{best_fitness}")
print(learning_curve)


def print_histogram(iteration, values, max_value=None, print_header=False):
    max_value = max_value if max_value is not None else max(values)
    range = (0, max_value)
    bins, edges = np.histogram(values, range=range)
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


def get_knapsack_problem(length=5, max_weight_pct=0.35, seed=None):
    if seed is not None:
        np.random.seed(seed)

    max_weight = length
    max_value = length
    weights = np.random.randint(1, high=max_weight, size=length)
    values = np.random.randint(1, high=max_value, size=length)
    print(weights)
    print(values)

    f_knapsack = mlrose.Knapsack(weights, values, max_weight_pct=max_weight_pct)
    maximize = True
    # Only use bit strings
    max_val = 2
    knapsack_problem = mlrose.DiscreteOpt(length, f_knapsack, max_val=max_val, maximize=maximize)
    return knapsack_problem


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
    show_best_state=False,
    maximize=True,
    max_fitness_value=None,
):
    max_attempts = 2
    iteration_values = []
    overall_best_state = None
    overall_best_fitness = -100000 if maximize else 100000
    rep_values = np.zeros((repetions_per_iteration))
    print_histogram(0, rep_values, max_value=max_fitness_value, print_header=True)
    for max_iters in range(total_iterations):
        rep_values = np.zeros((repetions_per_iteration))
        for i in range(repetions_per_iteration):
            best_state, best_fitness, learning_curve = mlrose.random_hill_climb(
                problem, max_attempts=max_attempts, max_iters=max_iters, restarts=max_iters, curve=True
            )
            if maximize:
                if best_fitness > overall_best_fitness:
                    overall_best_fitness = best_fitness
                    overall_best_state = best_state
            else:
                if best_fitness < overall_best_fitness:
                    overall_best_fitness = best_fitness
                    overall_best_state = best_state
            rep_values[i] = best_fitness

        bin_counts = print_histogram(max_iters, rep_values, max_value=max_fitness_value)

        iteration_values.append([max_iters, rep_values.mean(), rep_values, bin_counts])
    print(f"Best {overall_best_fitness}:{overall_best_state}")
    return iteration_values, overall_best_fitness


threshold_percentage = 0.15
f_four_peaks = mlrose.FourPeaks(t_pct=threshold_percentage)
length = 20
four_peaks_problem = mlrose.DiscreteOpt(length, f_four_peaks)
max_fitness_value = 36
repetions_per_iteration = 1000
total_iterations = 50
four_peaks_random_hill_values = random_hill(
    four_peaks_problem,
    total_iterations=total_iterations,
    max_fitness_value=max_fitness_value,
    repetions_per_iteration=repetions_per_iteration,
)

random_hill_chart(four_peaks_random_hill_values, title="Four Peaks Problem")
random_hill_chart_boxplot(four_peaks_random_hill_values, title="Four Peaks Problem")
random_hill_chart_seaborn_boxplot(four_peaks_random_hill_values, title="Four Peaks Problem")
random_hill_chart_lineplot(four_peaks_random_hill_values, title="Four Peaks Problem")
random_hill_chart_heatmap(four_peaks_random_hill_values, title="Four Peaks Problem")

f_queens = mlrose.Queens()
length = 8
max_val = 8
queens_problem = mlrose.DiscreteOpt(length, f_queens, max_val=max_val, maximize=False)
total_iterations = 50
repetions_per_iteration = 100
max_fitness_value = 28
rh_queens_values = random_hill(
    queens_problem,
    total_iterations=total_iterations,
    repetions_per_iteration=repetions_per_iteration,
    maximize=False,
    max_fitness_value=28,
)
# Switch Fitness
rh_queens_values[:, 1] = 28 - rh_queens_values[:, 1]
random_hill_chart(rh_queens_values, title="8 Queens")
random_hill_chart_seaborn_boxplot(rh_queens_values, title="8 Queens")
random_hill_chart_lineplot(rh_queens_values, title="8 Queens", maximize=False)
random_hill_chart_heatmap(rh_queens_values, title="8 Queens", maximize=False)

get_max_knapsack_fitness_values(8, 10)

length = 6
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
random_hill_chart(rh_knapsack_values, title="Knapsack")
random_hill_chart_boxplot(rh_knapsack_values, title="Knapsack")
random_hill_chart_seaborn_boxplot(rh_knapsack_values, title="Knapsack")

np.histogram(rh_knapsack_values[:, 1], bins="auto")


edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
f_max_k_color = mlrose.MaxKColor(edges)
f_max_k_color.evaluate(np.array([0, 2, 1, 2]))
