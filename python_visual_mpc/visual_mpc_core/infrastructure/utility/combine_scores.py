import glob
import pickle
import numpy as np

def combine_scores(dir, exp_name):
    improvement_l= []
    scores_l = []
    anglecost_l = []

    files = glob.glob(dir + '/scores_*')

    for f in files:
        print('load', f)
        dict_ = pickle.load(open(f, "rb"))
        scores_l.append(dict_['scores'])
        anglecost_l.append(dict_['anglecost'])
        improvement_l.append(dict_['improvement'])

    score = np.concatenate(scores_l, axis=0)
    anglecost = np.concatenate(anglecost_l, axis=0)
    improvement = np.concatenate(improvement_l, axis=0)
    sorted_ind = improvement.argsort()

    f = open(dir + '/results_all.txt', 'w')
    f.write('experiment name: ' + exp_name + '\n')
    f.write('overall best pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average pos improvemnt: {0}\n'.format(np.mean(improvement)))
    f.write('median pos improvement {}'.format(np.median(improvement)))
    f.write('standard deviation of population {0}\n'.format(np.std(improvement)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(improvement)/np.sqrt(improvement.shape[0])))
    f.write('---\n')
    f.write('average pos score: {0}\n'.format(np.mean(score)))
    f.write('median pos score {}'.format(np.median(score)))
    f.write('standard deviation of population {0}\n'.format(np.std(score)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(score)/np.sqrt(score.shape[0])))
    f.write('---\n')
    f.write('average angle cost: {0}\n'.format(np.mean(anglecost)))
    f.write('----------------------\n')
    f.write('traj: improv, score, anglecost, rank\n')
    f.write('----------------------\n')
    for t in range(improvement.shape[0]):
        f.write('{}: {}, {}, {}, :{}\n'.format(t, improvement[t], score[t], anglecost[t], np.where(sorted_ind == t)[0][0]))
    f.close()


if __name__ == '__main__':
    n_worker = 4
    n_traj = 49
    dir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks/shorttask/mj_plan/39178'

    traj_per_worker = int(n_traj / np.float32(n_worker))
    start_idx = [traj_per_worker * i for i in range(n_worker)]
    end_idx = [traj_per_worker * (i + 1) - 1 for i in range(n_worker)]

    combine_scores(dir, 'name')