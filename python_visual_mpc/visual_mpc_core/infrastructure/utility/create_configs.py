# creates a collection of random configurations for pushing
import argparse
import os
import python_visual_mpc
import imp
from python_visual_mpc.visual_mpc_core.infrastructure.sim import Sim

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pdb


def main():
    parser = argparse.ArgumentParser(description='create goal configs')
    parser.add_argument('experiment', type=str, help='experiment name')

    args = parser.parse_args()
    exp_name = args.experiment

    basepath = os.path.abspath(python_visual_mpc.__file__)
    basepath = '/'.join(str.split(basepath, '/')[:-2])
    data_coll_dir = basepath + '/pushing_data/' + exp_name
    hyperparams_file = data_coll_dir + '/hyperparams.py'

    hyperparams = imp.load_source('hyperparams', hyperparams_file).config

    c = Sim(hyperparams)
    c.run()

if __name__ == "__main__":
    main()