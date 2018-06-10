import argparse
import os
import sys
if sys.version_info[0] == 2:
    import imp
    import cPickle as pkl

else:
    import importlib.machinery
    import importlib.util
    import pickle as pkl

import glob
import numpy as np
import cv2

import moviepy.editor as mpy
def main():
    parser = argparse.ArgumentParser(description='run convert from directory to tf record')
    parser.add_argument('experiment', type=str, help='experiment hyperparameter path')
    parser.add_argument('output', type=str, help='gif output dir')
    parser.add_argument('-s', '--store', dest='store_gifs', action='store_true', default=False,
                        help='whether or not to save gifs')

    args = parser.parse_args()
    hyperparams_file = args.experiment
    out_dir = args.output

    data_coll_dir = '/'.join(hyperparams_file.split('/')[:-1])

    if sys.version_info[0] == 2:
        hyperparams = imp.load_source('hyperparams', args.experiment)
        hyperparams = hyperparams.config
        # python 2 means we're executing on sawyer. add dummy camera list
        hyperparams['agent']['cameras'] = ['main', 'left']
    else:
        loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        conf = importlib.util.module_from_spec(spec)
        loader.exec_module(conf)
        hyperparams = conf.config

    traj_per_file = hyperparams['traj_per_file']
    agent_config = hyperparams['agent']
    T = agent_config['T']

    data_dir = agent_config['data_save_dir']
    out_dir = data_coll_dir + '/' + out_dir
    img_height = agent_config['image_height']
    img_width = agent_config['image_width']

    print('loading from', data_dir)
    if args.store_gifs:
        print('saving to', out_dir)


    good_lift_ctr, total_ctr = 0, 0
    traj_group_dirs = glob.glob(data_dir + '/*')

    agent_config['goal_image'] = True

    if args.store_gifs and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for g in traj_group_dirs:
        trajs = glob.glob(g + '/*')
        for t in trajs:
            if not os.path.exists(t + '/state_action.pkl'):
                continue

            if 'cameras' in agent_config:
                valid = True
                for i in range(len(agent_config['cameras'])):
                    img_files = [t + '/images{}/im{}.png'.format(i, j) for j in range(T)]
                    if not all([os.path.exists(i) and os.path.isfile(i) for i in img_files]):
                        valid = False
                        print('traj {} missing /images{}'.format(t, i))
                        break
                if not valid:
                    continue
            else:
                if len(glob.glob(t + '/images/*.png')) != T:
                    continue

            try:
                state_action = pkl.load(open(t + '/state_action.pkl', 'rb'))
            except EOFError:
                continue

            if np.sum(np.isnan(state_action['target_qpos'])) > 0:
                print("FOUND NAN AT", t)  # error in mujoco environment sometimes manifest in NANs
            else:
                total_ctr += 1
                good_lift, _ = agent_config['file_to_record'](state_action, agent_config)

                if good_lift:
                    if args.store_gifs:
                        touch_sensors = state_action['finger_sensors']
                        touching = np.logical_and(state_action['states'][1:, -1] > 0,
                                       np.logical_and(touch_sensors[1:, 0] > 0, touch_sensors[1:, 1] > 0))

                        first_frame = np.argmax(touching)
                        next_proposals = state_action['finger_sensors'][first_frame + 1:, 0] > 0
                        next_proposals = [i + first_frame for i, val in enumerate(next_proposals) if val]

                        second_frame = np.argmax(state_action['states'][first_frame:, 2]) + first_frame
                        clip = []
                        for i in range(T):
                            if i == first_frame or i == second_frame:
                                clip.append(cv2.imread(t + '/images0/im_med{}.png'.format(i))[:, :, ::-1])
                            else:
                                clip.append(cv2.imread(t + '/images0/im_med{}.png'.format(i)))

                        clip = mpy.ImageSequenceClip(clip, fps = 5)
                        clip.write_gif('{}/good{}.gif'.format(out_dir, good_lift_ctr))
                    print('TRAJ {} IS GOOD'.format(t))
                    good_lift_ctr += 1
    print("Total num lifts {}, Good lift: {}%".format(good_lift_ctr, good_lift_ctr / float(total_ctr) * 100))



if __name__ == '__main__':
    main()
