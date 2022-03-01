import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

# import mlrose as mh
import numpy as np


def get_four_peaks_problem(length=20, threshold_percentage=0.15):

    f_four_peaks = mh.FourPeaks(t_pct=threshold_percentage)

    four_peaks_problem = mh.DiscreteOpt(length, f_four_peaks)
    return four_peaks_problem


def get_knapsack_problem(length=5, max_weight_pct=0.35, seed=None):
    if seed is not None:
        np.random.seed(seed)

    max_weight = length
    max_value = length
    weights = np.random.randint(1, high=max_weight, size=length)
    values = np.random.randint(1, high=max_value, size=length)

    f_knapsack = mh.Knapsack(weights, values, max_weight_pct=max_weight_pct)
    maximize = True
    # Only use bit strings
    max_val = 2
    knapsack_problem = mh.DiscreteOpt(length, f_knapsack, max_val=max_val, maximize=maximize)
    return knapsack_problem


def get_k_colors_problem(length, edge_percentage=0.3, seed=None, max_val=2):
    if seed is not None:
        np.random.seed(seed)

    edges = []
    for start in range(length):
        for end in range(start, length):
            if np.random.random_sample() < edge_percentage:
                edges.append((start, end))
    f_max_k_color = mh.MaxKColor(edges)
    maximize = False
    k_color_problem = mh.DiscreteOpt(length, f_max_k_color, max_val=max_val, maximize=maximize)
    return k_color_problem
