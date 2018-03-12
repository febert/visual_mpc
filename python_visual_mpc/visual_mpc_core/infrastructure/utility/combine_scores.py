import glob
import cPickle
import numpy as np

def combine_scores(dir, exp_name):
    full_scores = []
    full_anglecost = []

    files = glob.glob(dir + '/scores_*')

    for f in files:
        print 'load', f
        dict_ = cPickle.load(open(f, "rb"))
        full_scores.append(dict_['scores'])
        full_anglecost.append(dict_['anglecost'])

    scores = np.concatenate(full_scores, axis=0)
    sorted_ind = scores.argsort()
    anglecost = np.concatenate(full_anglecost, axis=0)

    f = open(dir + '/results_all.txt', 'w')
    f.write('experiment name: ' + exp_name + '\n')
    f.write('overall best pos score: {0} of traj {1}\n'.format(scores[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst pos score: {0} of traj {1}\n'.format(scores[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average pos score: {0}\n'.format(np.mean(scores)))
    f.write('median pos score: {0}\n'.format(np.median(scores)))
    f.write('standard deviation of population {0}\n'.format(np.std(scores)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(scores) / np.sqrt(scores.shape[0])))
    f.write('---\n')
    f.write('average angle cost: {0}\n'.format(np.mean(anglecost)))
    f.write('----------------------\n')
    f.write('traj: score, anglecost, rank\n')
    f.write('----------------------\n')
    for t in range(scores.shape[0]):
        f.write('{0}: {1}, {2}, :{3}\n'.format(t, scores[t], anglecost[t], np.where(sorted_ind == t)[0][0]))
    f.close()


if __name__ == '__main__':
    n_worker = 4
    n_traj = 49
    dir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks/mj_pos'

    traj_per_worker = int(n_traj / np.float32(n_worker))
    start_idx = [traj_per_worker * i for i in range(n_worker)]
    end_idx = [traj_per_worker * (i + 1) - 1 for i in range(n_worker)]

    combine_scores(dir, 'name')