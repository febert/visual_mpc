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
import copy

from python_visual_mpc.visual_mpc_core.infrastructure.utility.save_tf_record import save_tf_record

class DefaultTraj:
    def __init__(self):
        self.actions, self.X_Xdot_full, self.images  = None, None, None


def grasping_touch_file2record(state_action, agent_params):
    loaded_traj = DefaultTraj()

    loaded_traj.actions = state_action['actions']
    touch_sensors = state_action['finger_sensors']
    loaded_traj.X_Xdot_full = np.concatenate((state_action['target_qpos'][:-1, :], touch_sensors), axis = 1)

    good_lift = False

    valid_frames = np.logical_and(state_action['target_qpos'][1:, -1] > 0, np.logical_and(touch_sensors[:, 0] > 0, touch_sensors[:, 1] > 0))
    off_ground = state_action['target_qpos'][1:,2] >= 0
    object_poses = state_action['object_full_pose']

    if any(np.logical_and(valid_frames, off_ground)):
        obj_eq = object_poses[0, :, :2] == state_action['obj_start_end_pos']

        obj_eq = np.logical_and(obj_eq[:, 0], obj_eq[:, 1])
        obj_eq = np.argmax(obj_eq)
        obj_max =  np.amax(object_poses[:,obj_eq,2])
        if obj_max >=0:
            good_lift = True

    return good_lift, loaded_traj

def grasping_touch_nodesig_file2record(state_action, agent_params):
    loaded_traj = DefaultTraj()

    loaded_traj.actions = state_action['actions']
    touch_sensors = state_action['finger_sensors']
    loaded_traj.X_Xdot_full = np.concatenate((state_action['target_qpos'][:-1, :], touch_sensors), axis = 1)

    valid_frames = np.logical_and(state_action['target_qpos'][1:, -1] > 0, np.logical_and(touch_sensors[:, 0] > 0, touch_sensors[:, 1] > 0))
    off_ground = state_action['target_qpos'][1:,2] >= agent_params.get('good_lift_thresh', 0.)

    good_grasp = any(np.logical_and(valid_frames, off_ground))
    
    return good_grasp, loaded_traj


def grasping_sawyer_file2record(state_action, agent_params):
    loaded_traj = DefaultTraj()

    loaded_traj.actions = copy.deepcopy(state_action['actions'])

    if np.sum(np.abs(loaded_traj.actions)) <= 0.05:
        clip = agent_params['targetpos_clip']
        recover_actions = np.zeros((agent_params['T'] / 3, 4))
        rng_std = [0.08,0.08, 0.08, np.pi / 18]
        for i in range(4):
            for t in range(agent_params['T'] / 3):
                c_action = state_action['target_qpos'][3 * t + 1, i]
                if c_action == clip[1][i]:
                    val = np.random.normal() * rng_std[i]
                    while val < clip[1][i] - state_action['target_qpos'][3 * t, i]:
                        val = np.random.normal() * rng_std[i]
                    recover_actions[t, i] = val
                elif c_action == clip[0][i]:
                    val = np.random.normal() * rng_std[i]
                    while val > clip[0][i] - state_action['target_qpos'][3 * t, i]:
                        val = np.random.normal() * rng_std[i]
                    recover_actions[t, i] = val
                else:
                    recover_actions[t, i] = c_action - state_action['target_qpos'][3 * t, i]

        recover_actions = np.repeat(recover_actions, 3, axis = 0)
        loaded_traj.actions = copy.deepcopy(recover_actions)

    touch_sensors = state_action['finger_sensors']
    if 'autograsp' in agent_params:

        gripper_inputs = copy.deepcopy(touch_sensors[:, 0].reshape((-1, 1)))
        gripper_inputs[0, 0] = 0.       #grippers often have erroneous force readings at T = 0
        norm_states = copy.deepcopy(state_action['states'][:, :-1])
        for i in range(3):
            delta = agent_params['targetpos_clip'][1][i] - agent_params['targetpos_clip'][0][i]
            min = agent_params['targetpos_clip'][0][i]
            norm_states[:, i] -= min
            norm_states[:, i] /= delta

        loaded_traj.X_Xdot_full = np.concatenate((norm_states, gripper_inputs), axis=1)

    else:
        loaded_traj.X_Xdot_full = np.concatenate((state_action['states'], touch_sensors), axis=1)

    valid_frames = np.logical_and(state_action['states'][1:, -1] > 0,
                                  np.logical_and(touch_sensors[1:, 0] > 0, touch_sensors[1:, 1] > 0))
    off_ground = state_action['states'][1:, 2] >= agent_params.get('good_lift_thresh', 0.27)

    good_grasp = np.sum(np.logical_and(valid_frames, off_ground)) >= 2

    return good_grasp, loaded_traj

def pushing_touch_file2record(state_action, agent_params):
    loaded_traj = DefaultTraj()

    loaded_traj.actions = state_action['actions']
    touch_sensors = state_action['finger_sensors']
    loaded_traj.X_Xdot_full = np.concatenate((state_action['target_qpos'][:-1, :], touch_sensors), axis = 1)
    
    object_poses = state_action['object_full_pose']
    good_push = False

    if any(np.sum(np.sum(np.square(object_poses[1:,:,:2] - object_poses[1, :, :2].reshape((1, -1, 2))), axis = 1), axis = 1) > 0.01):
        good_push = True     
    return good_push, loaded_traj

def main():
    parser = argparse.ArgumentParser(description='run convert from directory to tf record')
    parser.add_argument('experiment', type=str, help='experiment hyperparameter path')
    parser.add_argument('output', type=str, help='new output dir')
    parser.add_argument('-g', action='store', dest='good_offset', type = int,
                    default = 0, help='Offset good records by g * traj_per_file')
    parser.add_argument('-b', action='store', dest='bad_offset', type = int,
                    default = 0, help='Offset bad records by b * traj_per_file')
    parser.add_argument('-i', action='store_true', dest='goal',
                        default=False, help='Store goal images')

    args = parser.parse_args()
    hyperparams_file = args.experiment
    out_dir = args.output

    data_coll_dir = '/'.join(hyperparams_file.split('/')[:-1])

    if sys.version_info[0] == 2:
        hyperparams = imp.load_source('hyperparams', args.experiment)
        hyperparams = hyperparams.config
        #python 2 means we're executing on sawyer. add dummy camera list
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

    extra_im = 0
    if args.goal:
        extra_im = 2
        agent_config['T'] += extra_im

    data_dir = agent_config['data_save_dir']
    out_dir = data_coll_dir + '/' + out_dir
    agent_config['data_save_dir'] = out_dir
    img_height = agent_config['image_height']
    img_width = agent_config['image_width']

    print('loading from', data_dir)
    print('saving to', out_dir)

    good_traj_list, bad_traj_list = [], []
    num_good_saved, num_bad_saved = args.good_offset, args.bad_offset
    print('GOOD OFFSET {}, BAD OFFSET {}'.format(num_good_saved, num_bad_saved))

    good_lift_ctr, total_ctr = 0, 0
    traj_group_dirs = glob.glob(data_dir+'/*')
    
    agent_config['goal_image'] = True 

    dirs_to_create = [agent_config['data_save_dir'] + d for d in ['/good/train', '/good/test', '/good/val', '/bad/train', '/bad/test', '/bad/val']]
    for d in dirs_to_create:
        if not os.path.exists(d):
            os.makedirs(d)
            print('Creating dir:', d)

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
                print("FOUND NAN AT", t)     #error in mujoco environment sometimes manifest in NANs
            else:
                total_ctr += 1
                good_lift, loaded_traj = agent_config['file_to_record'](state_action, agent_config)

                if 'cameras' in agent_config:
                    loaded_traj.images = np.zeros((T + extra_im, len(agent_config['cameras']), img_height, img_width, 3), dtype = np.uint8)
                else:
                    loaded_traj.images = np.zeros((T + extra_im, img_height, img_width, 3), dtype = np.uint8)

                if good_lift:
                    print(t)
                    good_lift_ctr += 1
                    good_traj_list.append(loaded_traj)

                    if args.goal:
                        touch_sensors = state_action['finger_sensors']
                        touching = np.logical_and(state_action['states'][1:, -1] > 0,
                                                  np.logical_and(touch_sensors[1:, 0] > 0, touch_sensors[1:, 1] > 0))

                        first_frame = np.argmax(touching)
                        next_proposals = state_action['finger_sensors'][first_frame + 1:, 0] > 0
                        next_proposals = [i + first_frame for i, val in enumerate(next_proposals) if val]
                        second_frame = np.argmax(state_action['states'][first_frame:, 2]) + first_frame

                        for cam in range(len(agent_config['cameras'])):
                            loaded_traj.images[-2, cam] = cv2.imread(t + '/images{}/im{}.png'.format(cam, first_frame))[:, :, ::-1]
                            loaded_traj.images[-1, cam] = cv2.imread(t + '/images{}/im{}.png'.format(cam, second_frame))[:,:, ::-1]
                        loaded_traj.actions = np.concatenate((loaded_traj.actions, np.zeros((2, 4))), axis = 0)
                        loaded_traj.X_Xdot_full = np.concatenate((loaded_traj.X_Xdot_full, np.zeros((2, 5))), axis = 0)



                else:
                    if args.goal:
                        continue       #bad trajectories have no target images
                    bad_traj_list.append(loaded_traj)

                for img in range(T):
                    if 'cameras' in agent_config:
                        for cam in range(len(agent_config['cameras'])):
                            loaded_traj.images[img, cam] = cv2.imread(t + '/images{}/im{}.png'.format(cam, img))[:, :, ::-1]
                    else:
                        loaded_traj.images[img] = cv2.imread(t + '/images/im{}.png'.format(img))[:, :, ::-1]

            if len(good_traj_list) % traj_per_file == 0 and len(good_traj_list) > 0:
                folder_prep = 'good/'
                if num_good_saved == 0:
                    folder_prep += 'test'
                elif np.random.rand() <= agent_config.get('train_val_split', 0.95):
                    folder_prep += 'train'
                else:
                    folder_prep += 'val'
                f_name = '{}/good_traj_{}_to_{}'.format(folder_prep, num_good_saved * traj_per_file, (num_good_saved + 1) * traj_per_file - 1)
                print('saving', f_name)
                save_tf_record(f_name, good_traj_list, agent_config)
                good_traj_list = []
                num_good_saved += 1
            elif len(bad_traj_list) % traj_per_file == 0 and len(bad_traj_list) > 0:
                folder_prep = 'bad/'
                if num_bad_saved == 0:
                    folder_prep += 'test'
                elif np.random.rand() <= agent_config.get('train_val_split', 0.95):
                    folder_prep += 'train'
                else:
                    folder_prep += 'val'
                f_name = '{}/bad_traj_{}_to_{}'.format(folder_prep, num_bad_saved * traj_per_file, (num_bad_saved + 1) * traj_per_file - 1)
                print('saving', f_name)
                save_tf_record(f_name, bad_traj_list, agent_config)
                bad_traj_list = []
                num_bad_saved += 1

    if  len(good_traj_list) > 0:
        f_name = 'good/train/good_traj_{0}_to_{1}'.format(num_good_saved * traj_per_file, (num_good_saved + 1) * traj_per_file - 1)
        save_tf_record(f_name, good_traj_list, agent_config)
        good_traj_list = []
    elif len(bad_traj_list) > 0:
        f_name = 'bad/train/bad_traj_{0}_to_{1}'.format(num_bad_saved * traj_per_file, (num_bad_saved + 1) * traj_per_file - 1)
        save_tf_record(f_name, bad_traj_list, agent_config)
        bad_traj_list = []
        
    print('perc good_lift', float(good_lift_ctr) / total_ctr)

if __name__ == '__main__':
    main()
