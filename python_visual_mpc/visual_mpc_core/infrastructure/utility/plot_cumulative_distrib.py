import glob
import cPickle as pkl
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

def read_new_scores(dir):
    folders = glob.glob('{}/*'.format(dir))

    scores, improvements = np.zeros(len(folders)), np.zeros((len(folders)))

    for i, f in enumerate(folders):
        if 'perturb' not in f:
            agent_data = pkl.load(open('{}/traj_data/agent_data.pkl'.format(f), 'rb'))
            scores[i], improvements[i] = agent_data['stats']['final_dist'], agent_data['stats']['improvement']

    return scores, improvements

def get_metric(folder, use_ind=None, only_take_first_n=None):
    if 'old_mpc' in folder:
        scores, imp = read_scores(folder)
    else:
        scores, imp = read_new_scores(folder)


    cummulative_fraction = []
    n_total = scores.shape[0]
    print('ntotal:',n_total)
    thresholds = np.linspace(0.,50, 200)

    print('mean', np.mean(scores))
    print('std',np.std(scores)/np.sqrt(n_total))
    for thres in thresholds:
        occ = np.where(scores < thres)[0]
        cummulative_fraction.append(float(occ.shape[0]) / n_total)

    occ = np.where(scores < 17)[0]
    print float(occ.shape[0]) / n_total

    return thresholds, cummulative_fraction

def plot_results(name, folders, labels, use_ind=None, only_take_first_n=None):

    plt.switch_backend('Agg')
    plt.figure(figsize=[7.5,6])
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

    folders = ['/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/grasping_sawyer/indep_views_predprop/sudri/exp',
               '/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/grasping_sawyer/indep_views_reuseaction_opencvtrack/sudri/exp',
               '/home/sudeep/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/grasping_sawyer/indep_views_reuseaction_highres/sudri/bench']
    labels = ['Visual MPC + Predictor Propagation', 'Visual MPC + OpenCV-Tracking', 'Visual MPC + Registration Network (ours)']
    plot_results('bench_grasp', folders, labels)

    # 3obj
    # file = [['/mnt/sda1/experiments/cem_exp/benchmarks/multiobj_pushing/3obj'], ['/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/3obj']]
    # labels = ['non-scaffolding','ours']
    # plot_results('3obj',file ,labels, only_take_first_n=48)
