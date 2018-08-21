import glob
import pickle
from python_visual_mpc.visual_mpc_core.benchmarks import write_scores
import numpy as np
import copy
import importlib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
from collections import OrderedDict
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import pdb


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def combine_scores(conf, dir, only_first_n=None):
    files = glob.glob(dir + '/scores_*')
    files = sorted_nicely(files)
    if len(files) == 0:
        raise ValueError

    stats_lists = OrderedDict()

    for f in files:
        print('load', f)
        dict_ = pickle.load(open(f, "rb"))
        for key in dict_.keys():
            if key not in stats_lists:
                stats_lists[key] = []
            stats_lists[key].append(dict_[key])

    pdb.set_trace()
    stat_array = OrderedDict()
    for key in dict_.keys():
        stat_array[key] = np.concatenate(stats_lists[key], axis=0)

    improvement = stat_array['improvement']
    final_dist = stat_array['final_dist']
    if only_first_n is not None:
        improvement = improvement[:only_first_n]
        final_dist = final_dist[:only_first_n]

    make_stats(dir, final_dist, 'finaldist', bounds=[0., 0.5])
    make_stats(dir, improvement, 'improvement', bounds=[-0.5, 0.5])
    make_imp_score(final_dist, improvement, dir)

    write_scores(conf, dir + '/results_all.txt', stat_array)
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
    dir = '/mnt/sda1/experiments/cem_exp/benchmarks/pos_ctrl/reg_startgoal_threshterm_tradeoff/65936'
    conf_dir = '/mnt/sda1/visual_mpc/experiments/cem_exp/benchmarks/pos_ctrl/reg_startgoal_threshterm_tradeoff'


    loader = importlib.machinery.SourceFileLoader('mod_hyper', conf_dir + '/mod_hyper.py')
    spec = importlib.util.spec_from_loader(loader.name, loader)
    conf = importlib.util.module_from_spec(spec)
    loader.exec_module(conf)

    conf = conf.config

    combine_scores(conf, dir, only_first_n=50)
