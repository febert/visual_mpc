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
    param_dir = args.experiment
    hyperparams = imp.load_source('hyperparams', param_dir).config

    c = Sim(hyperparams)
    c.run()

if __name__ == "__main__":
    main()