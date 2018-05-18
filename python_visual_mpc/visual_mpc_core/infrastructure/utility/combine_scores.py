import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import re

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def combine_scores(dir, only_first_n=None):
    improvement_l= []
    scores_l = []
    anglecost_l = []

    files = glob.glob(dir + '/scores_*')
    files = sorted_nicely(files)

    for f in files:
        print('load', f)
        dict_ = pickle.load(open(f, "rb"))
        scores_l.append(dict_['scores'])
        anglecost_l.append(dict_['anglecost'])
        improvement_l.append(dict_['improvement'])

    score = np.concatenate(scores_l, axis=0)
    anglecost = np.concatenate(anglecost_l, axis=0)
    improvement = np.concatenate(improvement_l, axis=0)

    if only_first_n is not None:
        improvement = improvement[:only_first_n]
        score = score[:only_first_n]
        anglecost = anglecost[:only_first_n]

    sorted_ind = copy.deepcopy(improvement).argsort()[::-1]

    mean_imp = np.mean(improvement)
    med_imp = np.median(improvement)
    mean_dist = np.mean(score)
    med_dist = np.median(score)

    make_stats(dir, score, 'score', bounds=[0., 0.5])
    make_stats(dir, improvement, 'improvement', bounds=[-0.5, 0.5])
    make_imp_score(score, improvement, dir)

    f = open(dir + '/results_all.txt', 'w')
    f.write('overall best pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average pos improvemnt: {0}\n'.format(mean_imp))
    f.write('median pos improvement {}'.format(med_imp))
    f.write('standard deviation of population {0}\n'.format(np.std(improvement)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(improvement)/np.sqrt(improvement.shape[0])))
    f.write('---\n')
    f.write('average pos score: {0}\n'.format(mean_dist))
    f.write('median pos score {}'.format(med_dist))
    f.write('standard deviation of population {0}\n'.format(np.std(score)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(score)/np.sqrt(score.shape[0])))
    f.write('---\n')
    f.write('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))
    f.write('---\n')
    f.write('average angle cost: {0}\n'.format(np.mean(anglecost)))
    f.write('----------------------\n')
    f.write('traj: improv, score, anglecost, rank\n')
    f.write('----------------------\n')
    for t in range(improvement.shape[0]):
        f.write('{}: {}, {}, {}, :{}\n'.format(t, improvement[t], score[t], anglecost[t], np.where(sorted_ind == t)[0][0]))

    f.close()

    print('writing {}'.format(dir))

def make_imp_score(score, imp, dir):
    plt.scatter(imp, score)
    plt.xlabel('improvement')
    plt.ylabel('final distance')
    plt.savefig(dir + '/imp_vs_dist.png')

def make_stats(dir, score, name, bounds):
    bin_edges = np.linspace(bounds[0], bounds[1], 11)
    binned_ind = np.digitize(score, bin_edges)
    occurrence, _ = np.histogram(score, bin_edges, density=False)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_mid = bin_edges + bin_width / 2
    plt.figure()
    plt.bar(bin_mid[:-1], occurrence, bin_width, facecolor='b', alpha=0.5)
    plt.title(name)
    plt.xlabel(name)
    plt.ylabel('occurences')
    plt.savefig(dir + '/' + name + '.png')
    plt.close()
    f = open(dir + '/{}_histo.txt'.format(name), 'w')
    for i in range(bin_edges.shape[0]-1):
        f.write('indices for bin {}, {} to {} : {} \n'.format(i, bin_edges[i], bin_edges[i+1], np.where(binned_ind == i+1)[0].tolist()))

if __name__ == '__main__':
    # n_worker = 4
    # n_traj = 10
    # dir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks/alexmodel/savp_register_gtruth_start/41256'
    # dir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks/pos_ctrl/updown_sact_boundact_register_gtruth/41272'
    dir = '/mnt/sda1/experiments/cem_exp/intmstep_benchmarks/3obj'

    # traj_per_worker = int(n_traj / np.float32(n_worker))
    # start_idx = [traj_per_worker * i for i in range(n_worker)]
    # end_idx = [traj_per_worker * (i + 1) - 1 for i in range(n_worker)]

    combine_scores(dir, only_first_n=48)