import glob
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
import re
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import read_scoes


def get_metric(sources, use_ind=None, only_take_first_n=None):
    scores = []
    for source in sources:
        _, imp, score, _ = read_scoes(source)
        scores.append(score)

    if use_ind is not None:
        scores = scores[0][use_ind] #assume only one array
    else:
        scores = np.concatenate(scores)

    if only_take_first_n:
        scores = scores[:only_take_first_n]

    cummulative_fraction = []
    n_total = scores.shape[0]
    print('ntotal:',n_total)
    thresholds = np.linspace(0., 0.2, 200)

    print('mean', np.mean(scores))
    print('std',np.std(scores)/np.sqrt(n_total))
    for thres in thresholds:
        occ = np.where(scores < thres)[0]
        cummulative_fraction.append(occ.shape[0] / n_total)

    return thresholds, cummulative_fraction

def plot_results(name, files, labels, use_ind=None, only_take_first_n=None):

    plt.figure()
    # markers = ['o', 'd']
    for file, label in zip(files, labels):
        print(label)
        thresholds, cummulative_fraction = get_metric(file, use_ind, only_take_first_n)
        plt.plot(thresholds, cummulative_fraction, label=label)

    plt.xlabel("threshold")
    plt.ylabel("fraction of runs less than threshold")
    # plt.ylim([0., 1.0])

    plt.legend()
    if use_ind is not None:
        suf = 'ordermatter'
    else: suf = ''
    plt.savefig('plots/{}scores_{}{}.pdf'.format(name, '-'.join(labels), suf))
    # plt.show()


def plot_order_matter():
    file = [['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/firsttry/100trials'],
            ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/firsttry']]

    inds = np.array([ 3,  4,  5,  8,  9, 17, 24, 26, 35, 36, 43, 45, 46, 59, 63, 66, 67,
               69, 70, 71, 72, 73, 74, 75, 79, 81, 83, 85, 95])

    labels = ['non scaffolding', 'ours']
    plot_results(file, labels, use_ind=inds)

if __name__ == '__main__':
    # plot_order_matter()
    # get results for all

    # #2 obj
    file = [['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/firsttry/100trials', '/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/firsttry'],
            ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/firsttry/48trials', '/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/firsttry']]
    labels = ['non-scaffolding','ours']
    plot_results('2obj_', file, labels)

    # 3obj
    # file = [['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/3obj'], ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/3obj']]
    # labels = ['non-scaffolding','ours']
    # plot_results('3obj',file ,labels, only_take_first_n=48)