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
output_directory = "experiments/four_peaks"

length = 20
fp_problem = get_four_peaks_problem(length=length)
experiment_name = f"length_{length}"
rhc = mh.RHCRunner(
    problem=fp_problem,
    experiment_name=experiment_name,
    output_directory=output_directory,
    seed=SEED,
    iteration_list=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    max_attempts=50,
    restart_list=[20, 40, 80],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = rhc.run()

stats_file = f"{output_directory}/{experiment_name}/rhc__{experiment_name}__run_stats_df.csv"
df = pd.read_csv(stats_file)

df_max = df.groupby(["Iteration", "Restarts"]).agg({"Fitness": "max"}).reset_index()

palette = sns.color_palette("hls", 3)
sns.lineplot(data=df_max, x="Iteration", y="Fitness", hue="Restarts", palette=palette)
plt.show()


sa_experiment_name = "sa_example_experiment"
sa = mh.SARunner(
    problem=fp_problem,
    experiment_name=sa_experiment_name,
    output_directory=output_directory,
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
