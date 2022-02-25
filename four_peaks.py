import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

from randomized_optimization import get_four_peaks_problem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from charting_runner import fitness_chart

SEED = 1
output_directory = "experiments/four_peaks"
title = "Four Peaks"

length = 20
experiment_name = f"length_{length}"

fp_problem = get_four_peaks_problem(length=length)
iteration_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
iteration_list = np.arange(0, 120, 10)

rhc = mh.RHCRunner(
    problem=fp_problem,
    experiment_name=experiment_name,
    output_directory=output_directory,
    seed=SEED,
    iteration_list=iteration_list,
    max_attempts=50,
    restart_list=[20, 30, 40],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = rhc.run()

stats_file = f"{output_directory}/{experiment_name}/rhc__{experiment_name}__run_stats_df.csv"
df = pd.read_csv(stats_file)
df_max = df.groupby(["Iteration", "Restarts"]).agg({"Fitness": "max"}).reset_index()

line_col = "Restarts"
random_hill_chart_rhc(df, line_col)

sa = mh.SARunner(
    problem=fp_problem,
    experiment_name=experiment_name,
    output_directory=output_directory,
    seed=SEED,
    iteration_list=iteration_list,
    max_attempts=5000,
    temperature_list=[1, 10, 100],
    decay_list=[mh.GeomDecay],
)

df_run_stats, df_run_curves = sa.run()

stats_file = f"{output_directory}/{experiment_name}/sa__{experiment_name}__run_stats_df.csv"
df = pd.read_csv(stats_file)
line_col = "Temperature"
sup_title = "Simulated Annealing"
fitness_chart(df, line_col, title=title, sup_title=sup_title)
