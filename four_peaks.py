import string
import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

from randomized_optimization import get_four_peaks_problem
import numpy as np
import argparse

import pandas as pd
from charting_runner import fitness_chart
from time import time


def run_four_peaks(algorithm_type, length, SEED=1, max_iterations=500, max_attempts=50, **kwargs):
    start_time = time()

    output_directory = "experiments/four_peaks"
    sup_title = f"Four Peaks (length={length})"
    experiment_name = f"length_{length}"

    fp_problem = get_four_peaks_problem(length=length)
    iteration_list = np.arange(0, max_iterations, max_iterations / 20)

    if algorithm_type == "rhc":
        if "restarts" not in kwargs:
            print(f"RHC needs -restarts 1 2 3 ")
            return

        rhc = mh.RHCRunner(
            problem=fp_problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=SEED,
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

        sa = mh.SARunner(
            problem=fp_problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=SEED,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            temperature_list=kwargs["temperatures"],
        )

        # the two data frames will contain the results
        df_run_stats, df_run_curves = sa.run()
        title = "Simulated Annealing"
        line_col = "Temperature"

    elif algorithm_type == "ga":
        if "populations" not in kwargs:
            print(f"GA needs -populations 50 100 200")
            return
        if "mutations" not in kwargs:
            print(f"GA needs -mutations 0.1 0.2 0.3")
            return

        ga = mh.GARunner(
            problem=fp_problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=SEED,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            population_sizes=kwargs["populations"],
            mutation_rates=kwargs["mutations"],
        )

        # the two data frames will contain the results
        df_run_stats, df_run_curves = ga.run()
        title = "Genetic Algorithm"
        line_col = "Population Size"

    elif algorithm_type == "mimic":
        if "keep_percents" not in kwargs:
            print(f"MIMIC needs -keep_percents 0.1 0.2 0.3")
            return
        if "populations" not in kwargs:
            print(f"MIMIC needs -populations 50 100 200")
            return

        mimic = mh.MIMICRunner(
            problem=fp_problem,
            experiment_name=experiment_name,
            output_directory=output_directory,
            seed=SEED,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            keep_percent_list=kwargs["keep_percents"],
            population_sizes=kwargs["populations"],
        )

        # the two data frames will contain the results
        df_run_stats, df_run_curves = mimic.run()
        title = "MIMIC"
        line_col = "Population Size"

    print(f"IN MAIN: {line_col}")
    stats_file = f"{output_directory}/{experiment_name}/{algorithm_type}__{experiment_name}__run_stats_df.csv"
    df = pd.read_csv(stats_file)
    fitness_chart(df, line_col, title=title, sup_title=sup_title)
    run_time = time() - start_time
    print(f"Run Time of {run_time:0.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Four Peaks Parser")
    parser.add_argument(
        "algorithm", choices=["rhc", "sa", "ga", "mimic"], help="The algorithm type: rhc, sa, ga, mimic"
    )
    parser.add_argument("length", type=int, help="The algorithm type: rhc, sa, ga, mimic")
    parser.add_argument("-restarts", nargs="+", type=int, help="Each restart value to be used for rhc")
    parser.add_argument("-temperatures", nargs="+", type=int, help="Each start temperature value to be used for sa")
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

    run_four_peaks(
        args.algorithm,
        args.length,
        SEED=args.seed,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        restarts=args.restarts,
        temperatures=args.temperatures,
        populations=args.populations,
        mutations=args.mutations,
        keep_percents=args.keep_percents,
    )
