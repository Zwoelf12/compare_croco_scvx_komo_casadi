import multiprocessing as mp
import itertools
import numpy as np
import tqdm
from pathlib import Path


class Multiprocess():
    def __init__(self, task, args, repetitions = 1, combination = None):
        """
        task : task that should be carried out
        args : list of lists of arguments that should be passed to task
        repetitions : how often should the set of arguments be applied
        combination : defines if all combinations should be used or just the in args defined
        """
        self.task = task 
        self.arg_combinations = build_arg_combinations(args, repetitions, combination)

    def run_processes(self):
        n_cpus = mp.cpu_count()-1
        n_cpus = 10
        print("running {} tasks on {} cpus in parallel".format(len(self.arg_combinations),n_cpus))
        with mp.Pool(n_cpus) as pool:
            asyn_results = pool.starmap_async(self.task, self.arg_combinations)
            results = asyn_results.get()
        return results

def build_arg_combinations(args , repetitions, combination):

    if combination == "all":
        arg_combinations = list(itertools.product(*args))
    else:
        arg_combinations = []
        for i in range(len(args[0])):
            args_i = []
            for a in args:
                args_i.append(a[i])
            arg_combinations.append(tuple(args_i))

    for i in range(repetitions-1):
        arg_combinations.extend(arg_combinations)

    return arg_combinations

