import pickle as pkl
import glob
import argparse
import random
import numpy as np
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--T', type=int, default=30)
    parser.add_argument('--ncam', type=int, default=2)
    parser.add_argument('--nimages', type=int, default=100)
    parser.add_argument('--invert_bad', action='store_true', default=False)
    parser.add_argument('--no_deltas', action='store_true', default=False)
    parser.add_argument('--im_per_row', type=int, default=5)
    args = parser.parse_args()
    assert args.nimages % args.im_per_row == 0, "--nimages should be a multiple of --im_per_row"

    traj_names = glob.glob('{}/traj*'.format(args.input_dir))
    random.shuffle(traj_names)

    img_summaries = [[[] for _ in range(args.im_per_row)] for t in range(args.T)]
    summary_counter = 0
    delta_sums, rollout_fails, num_good = [], [], 0

    for t in traj_names:
        if not args.no_deltas:
            delta_sums.append(pkl.load(open('{}/obs_dict.pkl'.format(t), 'rb'))['control_delta'][1:])
        agent_data = pkl.load(open('{}/agent_data.pkl'.format(t), 'rb'))
        if agent_data.get('goal_reached', False):
            num_good += 1
            print('traj {} is good'.format(t))
        rollout_fails.append(agent_data.get('extra_resets', 0))

        if summary_counter < args.nimages:
            for i in range(args.T):
                frame_imgs = []
                for n in range(args.ncam):
                    img_t = cv2.imread('{}/images{}/im_{}.png'.format(t, n, i))[:, :, ::-1]
                    if args.invert_bad and not agent_data['goal_reached'] :
                        img_t = img_t[:, :, ::-1]
                    frame_imgs.append(img_t)               
                img_summaries[i][int(summary_counter % args.im_per_row)].append(np.concatenate(frame_imgs, axis=1))
            summary_counter += 1

    if not args.no_deltas:
        delta_sums = np.array(delta_sums)
        adim = delta_sums.shape[-1]
        print('mean deltas: {}'.format(np.sum(np.sum(delta_sums, axis=0), axis = 0) / (args.T * len(traj_names))))
        print('median delta: {}, max delta: {}'.format(np.median(delta_sums.reshape(-1, adim), axis = 0), np.amax(delta_sums.reshape(-1, adim),axis=0)))
        tmaxs = np.argmax(delta_sums[:, :, -1], axis = -1)
        traj_max = np.argmax(delta_sums[np.arange(len(traj_names)), tmaxs, -1])
        print('max degree dif at traj: {}, t: {}'.format(traj_names[traj_max], tmaxs[traj_max]))

    print(' perc good: {}, and avg num failed rollouts: {}'.format(num_good / float(len(traj_names)), np.mean(rollout_fails)))
    img_summaries = [np.concatenate([np.concatenate(row, axis=0) for row in frame_t], axis = 1) for frame_t in img_summaries]
    npy_to_gif(img_summaries, './summaries')  

