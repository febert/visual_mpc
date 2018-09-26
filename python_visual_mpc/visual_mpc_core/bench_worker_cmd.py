import argparse
import importlib
import importlib.machinery
import random
import numpy as np
import importlib.util

import sys
import argparse
import os
import importlib.machinery
import importlib.util
from python_visual_mpc.visual_mpc_core.infrastructure.sim import Sim
from python_visual_mpc.visual_mpc_core.benchmarks import perform_benchmark
import copy
import random
import numpy as np
import glob
import re
import os
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import combine_scores
import pickle

def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def bench_worker_cmd():

    parser = argparse.ArgumentParser(description='run parllel data collection')

    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('gpu_id', type=int, help='the starting gpu_id', default=0)
    parser.add_argument('start_index', type=int, help='', default=-1)
    parser.add_argument('end_index', type=int, help='', default=-1)

    args = parser.parse_args()
    hyperparams_file = args.experiment
    gpu_id = args.gpu_id

    loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    hyperparams = mod.config

    hyperparams['start_index'] = args.start_index
    hyperparams['end_index'] = args.end_index

    random.seed(None)
    np.random.seed(None)

    perform_benchmark(hyperparams, -1, gpu_id=gpu_id)

if __name__ == '__main__':
    bench_worker_cmd()
