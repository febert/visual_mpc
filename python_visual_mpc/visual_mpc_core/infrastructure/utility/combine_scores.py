import glob
import pickle
from python_visual_mpc.visual_mpc_core.benchmarks import write_scores
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib
matplotlib.use('Agg')
import importlib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import re

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def combine_scores(conf, dir, only_first_n=None):
    improvement_l= []
    scores_l = []
    term_t_l = []

    files = glob.glob(dir + '/scores_*')
    files = sorted_nicely(files)

    for f in files:
        print('load', f)
        dict_ = pickle.load(open(f, "rb"))
        scores_l.append(dict_['scores'])
        improvement_l.append(dict_['improvement'])
        term_t_l.append(dict_['term_t'])

    score = np.concatenate(scores_l, axis=0)
    improvement = np.concatenate(improvement_l, axis=0)
    term_t = np.concatenate(term_t_l, axis=0)

    if only_first_n is not None:
        improvement = improvement[:only_first_n]
        score = score[:only_first_n]

    make_stats(dir, score, 'score', bounds=[0., 0.5])
    make_stats(dir, improvement, 'improvement', bounds=[-0.5, 0.5])
    make_imp_score(score, improvement, dir)

    write_scores(conf, dir + '/results_all.txt', improvement, score, term_t)
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
    dir = '/mnt/sda1/experiments/cem_exp/grasping_benchmarks/alexmodel_autograsp_b2000/62557'
    conf_dir = '/mnt/sda1/visual_mpc/experiments/cem_exp/grasping_benchmarks/alexmodel_autograsp_b2000'

    # traj_per_worker = int(n_traj / np.float32(n_worker))
    # start_idx = [traj_per_worker * i for i in range(n_worker)]
    # end_idx = [traj_per_worker * (i + 1) - 1 for i in range(n_worker)]

    loader = importlib.machinery.SourceFileLoader('mod_hyper', conf_dir + '/mod_hyper.py')
    spec = importlib.util.spec_from_loader(loader.name, loader)
    conf = importlib.util.module_from_spec(spec)
    loader.exec_module(conf)

    conf = conf.config

    combine_scores(conf, dir, only_first_n=50)