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
    title = "Four Peaks"
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
        sup_title = "Random Hill Climbing"
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
        sup_title = "Simulated Annealing"
        line_col = "Temperature"

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

    run_four_peaks(
        args.algorithm,
        args.length,
        SEED=args.seed,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        restarts=args.restarts,
        temperatures=args.temperatures,
    )
