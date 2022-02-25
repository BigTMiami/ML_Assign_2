import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

from randomized_optimization import get_four_peaks_problem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

SEED = 1
fp_problem = get_four_peaks_problem(length=24)
sa_experiment_name = "sa_example_experiment"
sa = mh.SARunner(
    problem=fp_problem,
    experiment_name=sa_experiment_name,
    output_directory="runner_output",
    seed=SEED,
    iteration_list=[2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    max_attempts=5000,
    temperature_list=[1, 10, 100],
    decay_list=[mh.GeomDecay],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = sa.run()

df = pd.read_csv("runner_output/sa_example_experiment/sa__sa_example_experiment__run_stats_df.csv")
sns.lineplot(data=df, x="Iteration", hue="Temperature", y="Fitness")
plt.show()


df.pivot(index="Iteration", columns="Temperature", values="Fitness")

df_run_stats.columns
df_run_stats.dtypes
df_run_stats = df_run_stats.drop(columns=["schedule_type"])
df_run_stats.columns

df_run_stats[["Iteration", "Temperature", "Fitness"]]

df_run_stats.pivot(index="Iteration", columns="Temperature", values="Fitness")

df_run_stats.pivot(index="Iteration", columns="Temperature", values="Fitness")

df_run_stats.pivot(index="Iteration", columns="Temperature", values="Fitness").plot.line()
plt.show()


rhc_experiment_name = "rhc_example_experiment"
SEED = 2
rhc = mh.RHCRunner(
    problem=fp_problem,
    experiment_name=rhc_experiment_name,
    output_directory="runner_output",
    seed=SEED,
    iteration_list=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    max_attempts=50,
    restart_list=[5, 10, 20],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = rhc.run()
df = pd.read_csv("runner_output/rhc_example_experiment/rhc__rhc_example_experiment__run_stats_df.csv")
sns.lineplot(data=df, x="Iteration", y="Fitness", hue="Restarts")
plt.show()
