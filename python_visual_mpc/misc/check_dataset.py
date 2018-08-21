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
    args = parser.parse_args()
    assert args.nimages % 5 == 0, "nimages should be a multiple of 5"

    traj_names = glob.glob('{}/traj*'.format(args.input_dir))
    random.shuffle(traj_names)

    img_summaries = [[[] for _ in range(5)] for t in range(args.T)]
    summary_counter = 0
    delta_sums, rollout_fails, num_good = [], [], 0

    for t in traj_names:
        delta_sums.append(pkl.load(open('{}/obs_dict.pkl'.format(t), 'rb'))['control_delta'][1:])
        agent_data = pkl.load(open('{}/agent_data.pkl'.format(t), 'rb'))
        if agent_data['goal_reached']:
            num_good += 1
        rollout_fails.append(agent_data['extra_resets'])

        if summary_counter < args.nimages:
            for i in range(args.T):
                frame_imgs = []
                for n in range(args.ncam):
                    img_t = cv2.imread('{}/images{}/im_{}.png'.format(t, n, i))[:, :, ::-1]
                    if not agent_data['goal_reached'] and args.invert_bad:
                        img_t = img_t[:, :, ::-1]
                    frame_imgs.append(img_t)               
                img_summaries[i][int(summary_counter % 5)].append(np.concatenate(frame_imgs, axis=1))
            summary_counter += 1
    delta_sums = np.array(delta_sums)
    adim = delta_sums.shape[-1]
    print('mean deltas: {}'.format(np.sum(np.sum(delta_sums, axis=0), axis = 0) / (args.T * len(traj_names))))
    print('median delta: {}, max delta: {}'.format(np.median(delta_sums.reshape(-1, adim), axis = 0), np.amax(delta_sums.reshape(-1, adim),axis=0)))
    print(' perc good: {}, and avg num failed rollouts: {}'.format(num_good / float(len(traj_names)), np.mean(rollout_fails)))
    tmaxs = np.argmax(delta_sums[:, :, -1], axis = -1)
    traj_max = np.argmax(delta_sums[np.arange(len(traj_names)), tmaxs, -1])
    print('max degree dif at traj: {}, t: {}'.format(traj_names[traj_max], tmaxs[traj_max]))
    img_summaries = [np.concatenate([np.concatenate(row, axis=0) for row in frame_t], axis = 1) for frame_t in img_summaries]
    npy_to_gif(img_summaries, './summaries')  

