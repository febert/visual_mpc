import argparse
from multiprocessing import Pool, Process, Manager
from python_visual_mpc.visual_mpc_core.agent.utils.traj_saver import record_worker
import cv2
import cPickle as pkl
import numpy as np
import glob
import random


def save_worker(save_conf):
    assigned_files, record_queue, T, target_width, seperate = save_conf
    target_dim = None
    ncam = None
    for traj in assigned_files:
        if target_dim == None:
            ncam = len(glob.glob('{}/images*/'.format(traj)))
            img = cv2.imread('{}/images0/im_0.jpg'.format(traj))
            old_dim = img.shape[:2]
            resize_ratio = target_width / float(old_dim[1])
            target_dim = (target_width, int(old_dim[0] * resize_ratio))
            print('resizing to {}'.format(target_dim[::-1]))

        agent_data = pkl.load(open('{}/agent_data.pkl'.format(traj), 'rb'))
        obs_dict = pkl.load(open('{}/obs_dict.pkl'.format(traj), 'rb'))
        policy_out = pkl.load(open('{}/policy_out.pkl'.format(traj), 'rb'))

        imgs = np.zeros((T, ncam, target_dim[1], target_dim[0], 3), dtype = np.uint8)

        for t in range(T):
            for n in range(ncam):
                img = cv2.imread('{}/images{}/im_{}.jpg'.format(traj, n, t))[:, :, ::-1]
                if '_mirror' in traj:
                    img = img[:, ::-1]
                imgs[t, n] = cv2.resize(img, target_dim, interpolation=cv2.INTER_AREA)
        obs_dict['images'] = imgs

        if seperate and not 'goal_reached' in agent_data:
            state = obs_dict['state']
            finger_sensor = obs_dict['finger_sensors']
            good_states = np.logical_and(state[:-1, 2] >= 0.9, state[:-1, -1] > 0)
            agent_data['goal_reached'] = np.sum(np.logical_and(finger_sensor[:-1, 0] > 0, good_states)) >= 2

        if 'stats' in agent_data:     #due to bug in genral_agent some robot trajs have a stats key that should be ignored
            assert agent_data['stats'] is None
            agent_data.pop('stats')

        record_queue.put((agent_data, obs_dict, policy_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str, help='target save path for record files')
    parser.add_argument('paths', type=str, help="Colon seperated list of paths to raw files")
    parser.add_argument('target_width', type=int, help='Target width to resize images')
    parser.add_argument('--T', type=int, help='agent trajectory time_sequence length', default=30)
    parser.add_argument('--offset', type=int, help='offset record counter (aka if records already exist)', default=0)
    parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
    parser.add_argument('--traj_per_file', type=int, help='number of trajectories per file', default=16)
    parser.add_argument('--seperate', dest='seperate_good', action='store_true', default=True)

    args = parser.parse_args()

    traj_files = []
    for s in args.paths.split(':'):
        for t_group in glob.glob('{}/traj_group*'.format(s)):
            traj_files = traj_files + glob.glob('{}/traj*'.format(t_group))
    random.shuffle(traj_files)

    print('Saving {} trajectories...'.format(len(traj_files)))

    m = Manager()
    record_queue = m.Queue()
    save_dir, T = args.save_dir, args.T
    seperate_good, traj_per_file = args.seperate_good, args.traj_per_file
    record_saver_proc = Process(target=record_worker, args=(
        record_queue, save_dir, T, seperate_good, traj_per_file, args.offset))
    record_saver_proc.start()

    if args.nworkers > 1:
        confs = []
        split = len(traj_files) // args.nworkers
        for w in range(args.nworkers):
            start, end = w * split, (w + 1) * split
            if w == args.nworkers - 1:
                end = len(traj_files)
            workers_files = traj_files[start:end]

            save_conf = (workers_files, record_queue, T, args.target_width, args.seperate_good)
            confs.append(save_conf)

        p = Pool(args.nworkers)
        p.map(save_worker, confs)
    else:
        save_worker((traj_files, record_queue, T, args.target_width, args.seperate_good))

    record_queue.put(None)
    record_saver_proc.join()
