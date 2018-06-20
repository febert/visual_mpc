import glob
import pickle
import numpy as np
import copy

"""
Copied to work with python 2.7
"""
import re
def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def read_scores(dir):
    files = glob.glob(dir + '/scores_*')
    files = sorted_nicely(files)

    scores = []
    imp = []
    for f in files:
        print('load', f)
        if 'txt' in f:
            with open(f, 'r') as file:
                lines = file.readlines()
                tkn_counter = 0
                while tkn_counter < 2:
                    if lines[0] == '----------------------\n':
                        tkn_counter += 1
                    lines = lines[1:]

                scores = []
                imp = []
                for l in lines:
                    improv, score = [float(i) for i in l.split(':')[1][1:].split(', ')]
                    scores.append(score)
                    imp.append(improv)
                return np.array(scores), np.array(imp)
        dict_ = pickle.load(open(f, "rb"))
        scores.append(dict_['scores'])
        imp.append(dict_['improvement'])
    return np.concatenate(scores), np.concatenate(imp)


def get_metric(folder, use_ind=None, only_take_first_n=None):

    scores, imp = read_scores(folder)

    cummulative_fraction = []
    n_total = scores.shape[0]
    print('ntotal:',n_total)
    thresholds = np.linspace(0., 0.5, 200)

    print('mean', np.mean(scores))
    print('std',np.std(scores)/np.sqrt(n_total))
    for thres in thresholds:
        occ = np.where(scores < thres)[0]
        cummulative_fraction.append(float(occ.shape[0]) / n_total)

    return thresholds, cummulative_fraction

def plot_results(name, folders, labels, use_ind=None, only_take_first_n=None):

    plt.switch_backend('TkAgg')
    plt.figure(figsize=[5,4])
    # markers = ['o', 'd']
    matplotlib.rcParams.update({'font.size': 15})

    for folder, label in zip(folders, labels):
        display_label = label.replace('-', ' ')
        print(display_label)
        thresholds, cummulative_fraction = get_metric(folder, use_ind, only_take_first_n)
        plt.subplot(1,1,1)
        plt.subplots_adjust(left=0.2, bottom=0.2, right=.9, top=.9, wspace=None, hspace=None)
        plt.plot(thresholds, cummulative_fraction, label=label)

    plt.xlabel("threshold")
    # plt.rc('axes', labelsize=20)  # fontsize of the x and y labels

    with PdfPages('plots/{}scores_{}.pdf'.format(name, '-'.join(labels))) as pdf:
        plt.ylim([0,1.0])
        plt.ylabel("fraction less than threshold")
        plt.legend()
        pdf.savefig()
        # plt.show()

if __name__ == '__main__':
    # plot_order_matter()
    # get results for all

    # #2 obj

    # folders = ['/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks_sawyer/weissgripper_predprop',
    #            '/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks_sawyer/weissgripper_dynrnn',
    #            '/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks_sawyer/weissgripper_regstartgoal_tradeoff']
    # labels = ['Prediction-Propagation','OpenCV-Tracking', 'GDN-Tracking-(ours)']

    folders = ['/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/grasping_sawyer/indep_views_reuseaction/exp_50',
               '/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/grasping_sawyer/indep_views_reuseaction_opencvtrack/exp_50']
    labels = ['OpenCV-Tracking', 'GDN-Tracking-(ours)']
    plot_results('2obj_', folders, labels)

    # 3obj
    # file = [['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/3obj'], ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/3obj']]
    # labels = ['non-scaffolding','ours']
    # plot_results('3obj',file ,labels, only_take_first_n=48)