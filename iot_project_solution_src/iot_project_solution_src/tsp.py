import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import math_utils


def tsp_fun(ass, lun):
    fitness_coords = mlrose.TravellingSales(coords = ass)
    problem_fit = mlrose.TSPOpt(length = lun, fitness_fn = fitness_coords, maximize=False)
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)