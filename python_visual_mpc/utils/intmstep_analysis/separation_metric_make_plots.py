import glob
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
import re

def get_metric(file):
    metric = pickle.load(open(file, "rb"))['separation_metric']

    cummulative_fraction = []
    n_total = len(metric)
    thresholds = np.linspace(0., 10, 200)
    for thres in thresholds:
        occ = np.where(metric < thres)[0]
        cummulative_fraction.append(occ.shape[0]/n_total)

    return thresholds, cummulative_fraction

def plot_results(files, labels):
    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0, hspace=0)

    # markers = ['o', 'd']
    for file, label in zip(files, labels):
        thresholds, cummulative_fraction = get_metric(file)
        ax.plot(thresholds, cummulative_fraction, label=label)

    plt.xlabel("threshold")
    plt.ylabel("fraction of runs less than threshold")
    # plt.ylim([0., 1.0])

    plt.legend()
    plt.savefig('plots/separation_metric_cdf.pdf')
    # plt.show()

if __name__ == '__main__':
    file = ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/get_metric/scores_0to99.pkl',
            '/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/get_metric_fixed_time/scores_0to99.pkl']
    labels = ['genmin', 'fixed']
    plot_results(file, labels)