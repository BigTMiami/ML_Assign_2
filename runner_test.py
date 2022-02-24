import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

from randomized_optimization import get_four_peaks_problem
import numpy as np

SEED = 1
fp_problem, max_f = get_four_peaks_problem()
sa_experiment_name = "sa_example_experiment"
sa = mh.SARunner(
    problem=fp_problem,
    experiment_name=sa_experiment_name,
    output_directory="runner_output",
    seed=SEED,
    iteration_list=[2, 4, 6, 8, 16, 32, 64, 128],
    max_attempts=5000,
    temperature_list=[0.1, 1, 10, 100],
    decay_list=[mh.GeomDecay],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = sa.run()

rhc_experiment_name = "rhc_example_experiment"
rhc = mh.RHCRunner(
    problem=fp_problem,
    experiment_name=rhc_experiment_name,
    output_directory="runner_output",
    seed=SEED,
    iteration_list=[1, 2, 4, 8, 16, 32, 64],
    max_attempts=5,
    restart_list=[5, 10, 20],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = rhc.run()
