import argparse
import os
import python_visual_mpc
import imp
import glob
import cPickle as pkl
import numpy as np
import cv2
from python_visual_mpc.visual_mpc_core.infrastructure.utility.save_tf_record import save_tf_record
class LoadTraj:
    def __init__(self):
        self.actions, self.X_Xdot_full, self._sample_images = None, None, None
def main():
    parser = argparse.ArgumentParser(description='run parllel data collection')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('output', type=str, help='new output dir')

    args = parser.parse_args()
    exp_name = args.experiment
    out_dir = args.output

    basepath = os.path.abspath(python_visual_mpc.__file__)
    basepath = '/'.join(str.split(basepath, '/')[:-2])
    data_coll_dir = basepath + '/pushing_data/' + exp_name
    hyperparams_file = data_coll_dir + '/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file).config

    traj_per_file = hyperparams['traj_per_file']
    agent_config = hyperparams['agent']
    T = agent_config['T']

    data_dir = agent_config['data_save_dir']
    out_dir = data_coll_dir + '/' + out_dir
    agent_config['data_save_dir'] = out_dir
    img_height = agent_config['image_height']
    img_width = agent_config['image_width']

    print 'loading from', data_dir
    print 'saving to', out_dir

    traj_list = []
    counter = 0
    num_saved = 0

    traj_group_dirs = glob.glob(data_dir+'/*')
    for g in traj_group_dirs:
        trajs = glob.glob(g + '/*')
        for t in trajs:
            state_action = pkl.load(open(t + '/state_action.pkl', 'r'))
            if np.sum(np.isnan(state_action['target_qpos'])) > 0:
                print "FOUND NAN AT", t
            else:
                loaded_traj = LoadTraj()
                loaded_traj.actions = state_action['actions']
                loaded_traj.X_Xdot_full = state_action['target_qpos'][:T, :]
                loaded_traj._sample_images = np.zeros((T, img_height, img_width, 3), dtype = 'uint8')
                for i in range(T):
                    img = cv2.imread(t + '/images/im{}.png'.format(i))[:, :, ::-1]
                    loaded_traj._sample_images[i] = img

                traj_list.append(loaded_traj)
                counter += 1

            if counter % traj_per_file == 0:
                f_name = 'traj_{0}_to_{1}'.format(num_saved * traj_per_file, (num_saved + 1) * traj_per_file - 1)
                save_tf_record(f_name, traj_list, agent_config)
                traj_list = []

                num_saved += 1
                counter = 0

if __name__ == '__main__':
    main()