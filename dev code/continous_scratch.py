import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

# import mlrose as mh
import numpy as np


def custom_fitness(state):
    w0 = 50
    w1 = 20
    w2 = 25
    return abs(w0 - state[0]) + (w1 - state[1]) ** 2 + abs(w2 - state[2])


st1 = np.array([34, 44, 3])

custom_fitness(st1)

fitness = mh.CustomFitness(custom_fitness, "continuous")

custom_problem = mh.ContinuousOpt(3, fitness, maximize=False, max_val=100)

for i in range(10):
    best_state, best_fitness, fitness_curve = mh.random_hill_climb(
        custom_problem, max_attempts=2, max_iters=5000, restarts=5000, curve=True
    )
    print(f"{best_fitness:6.2f}: [{best_state[0]:6.2f}, {best_state[1]:6.2f}, {best_state[2]:6.2f}]")
print(fitness_curve)
