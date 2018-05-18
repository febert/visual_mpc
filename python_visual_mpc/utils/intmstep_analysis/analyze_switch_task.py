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
    scores = np.stack(get_metric(files), axis=0)
    abs_delta = np.abs(scores[0] - scores[1])
    # for i in range(scores[0].shape[0]):
        # print('{}: {}'.format(i, abs_delta[i]))
    order_matter_ind = np.where(abs_delta > 0.08)
    min_scores = np.min(scores, axis=0)

    num_ex = min_scores.shape[0]
    print('minimumm of both orders: mean:{}, +- {}'.format(np.mean(min_scores), np.std(min_scores)/np.sqrt(num_ex)))

    print(order_matter_ind)
    print('num order matter ', order_matter_ind[0].shape)

    # plt.plot(abs_delta)
    # plt.show()

if __name__ == '__main__':

    file = ['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/switchtask0','/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/switchtask1']
    plot_results(file)