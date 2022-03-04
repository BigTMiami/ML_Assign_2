import string
import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

from problems import get_four_peaks_problem, get_knapsack_problem, get_k_colors_problem
import numpy as np
import argparse

import pandas as pd
from charting import fitness_chart, time_chart
from time import time


def run_algorithm_with_problem(
    problem_type, algorithm_type, length, seed=1, max_iterations=500, max_attempts=50, **kwargs
):
    maximize = True
    start_time = time()
    experiment_name = f"length_{length}"
    iteration_list = np.arange(0, max_iterations, max_iterations / 20)
    info_settings = {"l": length, "ma": max_attempts}

    output_directory = f"experiments/{problem_type}"
    if problem_type == "four_peaks":
        sup_title = f"Four Peaks (length={length})"
        problem = get_four_peaks_problem(length=length)
    elif problem_type == "knapsack":
        sup_title = f"Knapsack (length={length})"
        problem = get_knapsack_problem(length=length, seed=seed)
    elif problem_type == "k_color":
        sup_title = f"K Colors (length={length})"
        problem = get_k_colors_problem(length=length, seed=seed)
        maximize = False
    else:
        print(f"Unsupported Problem Type of {problem_type}")
        return

    if algorithm_type == "rhc":
        if "restarts" not in kwargs:
            print(f"RHC needs -restarts 1 2 3 ")
            return
        info_settings["r"] = kwargs["restarts"]

        rhc = mh.RHCRunner(
            problem=problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=seed,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            restart_list=kwargs["restarts"],
        )

        # the two data frames will contain the results
        df_run_stats, df_run_curves = rhc.run()
        title = "Random Hill Climbing"
        line_col = "Restarts"

    elif algorithm_type == "sa":
        if "temperatures" not in kwargs:
            print(f"SA needs -temperatures 1 2 3 ")
            return

        if "decays" not in kwargs:
            print(f"SA needs -decays geom arith exp ")
            return

        decay_list = []
        for decay_type in kwargs["decays"]:
            if decay_type == "geom":
                decay_list.append(mh.GeomDecay)
            elif decay_type == "arith":
                decay_list.append(mh.ArithDecay)
            elif decay_type == "exp":
                decay_list.append(mh.ExpDecay)
            else:
                print(f"Unsupported decay type {decay_type}")
                return

        info_settings["d"] = kwargs["decays"]
        info_settings["t"] = kwargs["temperatures"]

        sa = mh.SARunner(
            problem=problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=seed,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            temperature_list=kwargs["temperatures"],
            decay_list=decay_list,
        )

        # the two data frames will contain the results
        df_run_stats, df_run_curves = sa.run()
        title = "Simulated Annealing"
        if len(decay_list) > 1:
            line_col = "schedule_type"
        else:
            line_col = "Temperature"

    elif algorithm_type == "ga":
        if "populations" not in kwargs:
            print(f"GA needs -populations 50 100 200")
            return
        if "mutations" not in kwargs:
            print(f"GA needs -mutations 0.1 0.2 0.3")
            return

        info_settings["p"] = kwargs["populations"]
        info_settings["mu"] = kwargs["mutations"]

        ga = mh.GARunner(
            problem=problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=seed,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            population_sizes=kwargs["populations"],
            mutation_rates=kwargs["mutations"],
        )

        # the two data frames will contain the results
        df_run_stats, df_run_curves = ga.run()
        title = "Genetic Algorithm"
        if len(kwargs["populations"]) > 1:
            line_col = "Population Size"
        else:
            line_col = "Mutation Rate"

    elif algorithm_type == "mimic":
        if "keep_percents" not in kwargs:
            print(f"MIMIC needs -keep_percents 0.1 0.2 0.3")
            return
        if "populations" not in kwargs:
            print(f"MIMIC needs -populations 50 100 200")
            return

        info_settings["p"] = kwargs["populations"]
        info_settings["k"] = kwargs["keep_percents"]

        mimic = mh.MIMICRunner(
            problem=problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=seed,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            keep_percent_list=kwargs["keep_percents"],
            population_sizes=kwargs["populations"],
        )

        # the two data frames will contain the results
        df_run_stats, df_run_curves = mimic.run()
        title = "MIMIC"
        if len(kwargs["populations"]) > 1:
            line_col = "Population Size"
        else:
            line_col = "Keep Percent"

    stats_file = f"{output_directory}/{experiment_name}/{algorithm_type}__{experiment_name}__run_stats_df.csv"
    df = pd.read_csv(stats_file)
    fitness_chart(df, line_col, title=title, sup_title=sup_title, maximize=maximize, info_settings=info_settings)
    run_time = time() - start_time
    best_fitness = df["Fitness"].max() if maximize else df["Fitness"].min()
    print(f"Max Fitness: {best_fitness} Run Time of {run_time:0.2f}")


def combination_chart(problem_type, experiment_name, xscale_log=True):
    problem_dict = {"four_peaks": "Four Peaks", "knapsack": "Knapsack", "k_color": "K Colors"}
    if problem_type not in problem_dict:
        print(f"Unsupported problem type of {problem_type}")
        return
    problem_name = problem_dict[problem_type]
    output_directory = f"experiments/{problem_type}"
    algorithm_type = "sa"
    stats_file = f"{output_directory}/{experiment_name}/{algorithm_type}__{experiment_name}__run_stats_df.csv"
    df = pd.read_csv(stats_file)
    max_fitness = df["Fitness"].max()
    sa_time = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Time"].iloc[0]
    schedule_type = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["schedule_type"].iloc[0]
    Temperature = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Temperature"].iloc[0]
    df_sa = df[(df["Temperature"] == Temperature) & (df["schedule_type"] == schedule_type)][["Iteration", "Fitness"]]
    df_sa["Algorithm Type"] = "Simulated Annealing"

    algorithm_type = "rhc"
    stats_file = f"{output_directory}/{experiment_name}/{algorithm_type}__{experiment_name}__run_stats_df.csv"
    df = pd.read_csv(stats_file)
    max_fitness = df["Fitness"].max()
    rhc_time = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Time"].iloc[0]
    restarts = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Restarts"].iloc[0]
    df_rhc = df[df["Restarts"] == restarts][["Iteration", "Fitness"]]
    df_rhc["Algorithm Type"] = "Random Hill Climb"

    algorithm_type = "ga"
    stats_file = f"{output_directory}/{experiment_name}/{algorithm_type}__{experiment_name}__run_stats_df.csv"
    df = pd.read_csv(stats_file)
    max_fitness = df["Fitness"].max()
    ga_time = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Time"].iloc[0]
    Population_Size = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Population Size"].iloc[0]
    Mutation_Rate = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Mutation Rate"].iloc[0]
    df_ga = df[(df["Population Size"] == Population_Size) & (df["Mutation Rate"] == Mutation_Rate)][
        ["Iteration", "Fitness"]
    ]
    df_ga["Algorithm Type"] = "Genetic Algorithm"

    algorithm_type = "mimic"
    stats_file = f"{output_directory}/{experiment_name}/{algorithm_type}__{experiment_name}__run_stats_df.csv"
    df = pd.read_csv(stats_file)
    max_fitness = df["Fitness"].max()
    mimic_time = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Time"].iloc[0]
    Population_Size = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Population Size"].iloc[0]
    Keep_Percent = df[df["Fitness"] == max_fitness].sort_values(by=["Iteration"])["Keep Percent"].iloc[0]
    df_mimic = df[(df["Population Size"] == Population_Size) & (df["Keep Percent"] == Keep_Percent)][
        ["Iteration", "Fitness"]
    ]
    df_mimic["Algorithm Type"] = "MIMIC"

    df_master = pd.concat([df_sa, df_ga, df_rhc, df_mimic])
    sup_title = f"{problem_name} ({experiment_name})"
    fitness_chart(
        df_master,
        line_col="Algorithm Type",
        title=f"Algorithm Fitness Curves",
        sup_title=sup_title,
        xscale_log=xscale_log,
    )

    algorithms = ["Simulated Annealing", "Random Hill Climb", "Genetic Algorithm", "MIMIC"]
    times = [sa_time, rhc_time, ga_time, mimic_time]
    time_chart(algorithms, times, title=f"Algorithms by Time", sup_title=sup_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Optimization Solver")
    parser.add_argument(
        "problem",
        choices=["four_peaks", "knapsack", "k_color"],
        help="The algorithm type: four_peaks, knapsack, k_color",
    )
    parser.add_argument(
        "algorithm", choices=["rhc", "sa", "ga", "mimic"], help="The algorithm type: rhc, sa, ga, mimic"
    )
    parser.add_argument("length", type=int, help="The algorithm type: rhc, sa, ga, mimic")
    parser.add_argument("-restarts", nargs="+", type=int, help="Each restart value to be used for rhc")
    parser.add_argument("-temperatures", nargs="+", type=int, help="Each start temperature value to be used for sa")
    parser.add_argument("-decays", nargs="+", type=str, help="Each decay model type to be used for sa")
    parser.add_argument("-populations", nargs="+", type=int, help="Each popuation value to be used for ga")
    parser.add_argument("-mutations", nargs="+", type=float, help="Each mutation value to be used for ga")
    parser.add_argument("-keep_percents", nargs="+", type=float, help="Each keep percent value to be used for mimic")
    parser.add_argument("-max_iterations", type=int, default=500)
    parser.add_argument("-max_attempts", type=int, default=50)
    parser.add_argument("-seed", type=int, default=1)
    args = parser.parse_args()

    if args.algorithm == "rhc":
        if args.restarts is None:
            print(f"Restarts must be provided for rhc")
            exit()

    if args.algorithm == "sa":
        if args.temperatures is None:
            print(f"Temperatures must be provided for sa")
            exit()

        if args.decays is None:
            print(f"Decays must be provided for sa")
            exit()

    if args.algorithm == "ga":
        if args.populations is None:
            print(f"Populations must be provided for ga")
            exit()
        if args.mutations is None:
            print(f"Mutations must be provided for ga")
            exit()

    if args.algorithm == "mimic":
        if args.keep_percents is None:
            print(f"Keep Percents must be provided for mimic")
            exit()
        if args.populations is None:
            print(f"Populations must be provided for mimic")
            exit()

    run_algorithm_with_problem(
        args.problem,
        args.algorithm,
        args.length,
        seed=args.seed,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        restarts=args.restarts,
        temperatures=args.temperatures,
        populations=args.populations,
        mutations=args.mutations,
        keep_percents=args.keep_percents,
        decays=args.decays,
    )

    print("\a")
