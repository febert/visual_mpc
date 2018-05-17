import glob
import pickle
import numpy as np
import copy
import re
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import read_scoes


def get_metric(sources):
    scores = []
    for source in sources:
        _, imp, score, _ = read_scoes(source)
        scores.append(score)

    return scores

def plot_results(files):
    scores = get_metric(files)
    abs_delta = np.abs(scores[0] - scores[1])
    # for i in range(scores[0].shape[0]):
        # print('{}: {}'.format(i, abs_delta[i]))
    order_matter_ind = np.where(abs_delta > 0.08)
    print(order_matter_ind)
    print('num oreder matter ', order_matter_ind[0].shape)

    # plt.plot(abs_delta)
    # plt.show()

if __name__ == '__main__':

    file = ['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/switchtask0','/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/switchtask1']
    plot_results(file)