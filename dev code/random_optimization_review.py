from charting import random_hill_chart_lineplot, random_hill_chart_heatmap, chart_lineplot, chart_heatmap
from randomized_optimization import (
    get_four_peaks_problem,
    get_k_colors_problem,
    get_knapsack_problem,
    random_hill,
    iterate_algorithm,
    optimize_decay_schedule,
    get_decay_schedule,
)


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

## Knapsack
knapsack_problem, max_fitness_value, min_fitness_value = get_knapsack_problem()

optimize_decay_schedule(
    knapsack_problem,
    max_iters_start=100,
    start_temp=11,
    end_temp=5,
    temp_step=-0.25,
    total_repetitions=5000,
)

knapsack_decay_schedule = get_decay_schedule("knapsack")
total_iterations = 100
repetions_per_iteration = 100
sa_knapsack_values, knapsack_max_fitness, knapsack_min_fitness = iterate_algorithm(
    knapsack_problem,
    "sa",
    knapsack_decay_schedule,
    max_fitness_value=max_fitness_value,
    total_iterations=total_iterations,
)
chart_lineplot(sa_knapsack_values, title="Knapsack", suptitle="Simulated Annealing")
chart_heatmap(sa_knapsack_values, title="Knapsack", suptitle="Simulated Annealing")

## 2 Color Problem
color_problem, max_fitness_value, min_fitness_value = get_k_colors_problem()

optimize_decay_schedule(
    color_problem,
    max_iters_start=50,
    start_temp=0.2,
    end_temp=0,
    temp_step=-0.025,
    total_repetitions=3000,
)

color_problem, max_fitness_value, min_fitness_value = get_k_colors_problem()
color_decay_schedule = get_decay_schedule("color")
total_iterations = 100
repetions_per_iteration = 100
sa_color_values, color_max_fitness, color_min_fitness = iterate_algorithm(
    color_problem,
    "sa",
    color_decay_schedule,
    max_fitness_value=max_fitness_value,
    min_fitness_value=min_fitness_value,
    total_iterations=total_iterations,
)
chart_lineplot(sa_color_values, title="2 Color Problem", suptitle="Simulated Annealing", maximize=False)
chart_heatmap(sa_color_values, title="2 Color Problem", suptitle="Simulated Annealing", maximize=False)
