import glob
import pickle
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import re
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import sorted_nicely
from matplotlib.backends.backend_pdf import PdfPages

def read_scores(dir):
    files = glob.glob(dir + '/scores_*')
    files = sorted_nicely(files)

    scores = []
    imp = []
    for f in files:
        print('load', f)
        dict_ = pickle.load(open(f, "rb"))
        scores.append(dict_['scores'])
        imp.append(dict_['improvement'])
    return np.concatenate(scores), np.concatenate(imp)


def get_metric(folder, use_ind=None, only_take_first_n=None):

    scores, imp = read_scores(folder)

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

def plot_results(name, folders, labels, use_ind=None, only_take_first_n=None):

    plt.switch_backend('TkAgg')
    plt.figure(figsize=[5,4])
    # markers = ['o', 'd']
    for folder, label in zip(folders, labels):
        print(label)
        thresholds, cummulative_fraction = get_metric(folder, use_ind, only_take_first_n)
        plt.plot(thresholds, cummulative_fraction, label=label)

    plt.xlabel("threshold")
    # matplotlib.rcParams.update({'font.size': 20})
    # plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    matplotlib.rc('axes', titlesize=8)

    with PdfPages('plots/{}scores_{}.pdf'.format(name, '-'.join(labels))) as pdf:
        plt.ylim([0,1.0])
        plt.ylabel("fraction of runs less than threshold")
        plt.legend()
        pdf.savefig()
        # plt.savefig('plots/{}scores_{}.png'.format(name, '-'.join(labels)))
        # plt.show()

if __name__ == '__main__':
    # plot_order_matter()
    # get results for all

    # #2 obj
    folders = ['/mnt/sda1/experiments/cem_exp/benchmarks/pos_ctrl/reg_startgoal_threshterm_tradeoff/66581',
               '/mnt/sda1/experiments/cem_exp/benchmarks/pos_ctrl/predprop_threshterm/62538',
               '/mnt/sda1/experiments/cem_exp/benchmarks/pos_ctrl/gtruth_track_thresterm/62540']
    labels = ['ours', 'predprop','ground-truth-track']
    plot_results('2obj_', folders, labels)

    # 3obj
    # file = [['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/3obj'], ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/3obj']]
    # labels = ['non-scaffolding','ours']
    # plot_results('3obj',file ,labels, only_take_first_n=48)