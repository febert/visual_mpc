import argparse
import os
import python_visual_mpc
import importlib.machinery
import importlib.util
import glob
import pickle as pkl
import numpy as np
import cv2
from python_visual_mpc.visual_mpc_core.infrastructure.utility.save_tf_record import save_tf_record
class LoadTraj:
    def __init__(self):
        self.actions, self.X_Xdot_full, self._sample_images, self.goal_image = None, None, None, None
def main():
    parser = argparse.ArgumentParser(description='run convert from directory to tf record')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('output', type=str, help='new output dir')

    args = parser.parse_args()
    exp_name = args.experiment
    out_dir = args.output

    basepath = os.path.abspath(python_visual_mpc.__file__)
    basepath = '/'.join(str.split(basepath, '/')[:-2])
    data_coll_dir = basepath + '/pushing_data/' + exp_name

    hyperparams_file = data_coll_dir + '/hyperparams.py'
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
    agent_config['data_save_dir'] = out_dir
    img_height = agent_config['image_height']
    img_width = agent_config['image_width']

    print('loading from', data_dir)
    print('saving to', out_dir)

    traj_list = []
    counter = 0
    num_saved = 0
    no_lift, total_ctr = 0, 0
    traj_group_dirs = glob.glob(data_dir+'/*')
    for g in traj_group_dirs:
        trajs = glob.glob(g + '/*')
        for t in trajs:
            state_action = pkl.load(open(t + '/state_action.pkl', 'rb'))
            if np.sum(np.isnan(state_action['target_qpos'])) > 0:
                print("FOUND NAN AT", t)
            else:
                loaded_traj = LoadTraj()
                goal_pos = state_action['obj_start_end_pos'][1]
                loaded_traj.actions = state_action['actions']
                loaded_traj.X_Xdot_full = state_action['target_qpos'][:T, :]
                loaded_traj._sample_images = np.zeros((T, img_height, img_width, 3), dtype = 'uint8')

                for i in range(T):
                    img = cv2.imread(t + '/images/im{}.png'.format(i))[:, :, ::-1]
                    loaded_traj._sample_images[i] = img

                    if np.sum(np.abs(goal_pos)) > 0 and all(goal_pos == loaded_traj.X_Xdot_full[i, :2]) and loaded_traj.X_Xdot_full[i, 2] ==-0.08 and loaded_traj.goal_image is None:
                        loaded_traj.goal_image = img
                if loaded_traj.goal_image is None:
                    print('NO GOAL PROBABLY DIDNT LIFT')
                    loaded_traj.goal_image = cv2.imread(t + '/images/im{}.png'.format(0))[:, :, ::-1]
                    no_lift += 1
                traj_list.append(loaded_traj)
                counter += 1
                total_ctr += 1
            if counter % traj_per_file == 0:
                f_name = 'traj_{0}_to_{1}'.format(num_saved * traj_per_file, (num_saved + 1) * traj_per_file - 1)
                save_tf_record(f_name, traj_list, agent_config)
                traj_list = []

                num_saved += 1
                counter = 0
    print('perc no_lift', no_lift / total_ctr)

if __name__ == '__main__':
    main()