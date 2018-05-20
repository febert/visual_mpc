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
        self.actions, self.X_Xdot_full, self.images, self.goal_image = None, None, None, None
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

    good_traj_list, bad_traj_list = [], []
    num_good_saved, num_bad_saved = 352, 273
    good_lift_ctr, total_ctr = 0, 0
    traj_group_dirs = glob.glob(data_dir+'/*')
    
    agent_config['goal_image'] = True 
    
    for g in traj_group_dirs:
        trajs = glob.glob(g + '/*')
        for t in trajs:
            if len(glob.glob(t + '/images/*.png')) != T or not os.path.exists(t + '/state_action.pkl'):
                print('TRAJ', t, 'is broken')
                continue
            try:
                state_action = pkl.load(open(t + '/state_action.pkl', 'rb'))
            except EOFError:
                print('TRAJ', t, 'is broken')
                continue
            if np.sum(np.isnan(state_action['target_qpos'])) > 0:
                print("FOUND NAN AT", t)
            else:
                loaded_traj = LoadTraj()
                goal_pos = state_action['obj_start_end_pos'][1]
                loaded_traj.actions = state_action['actions']
                loaded_traj.X_Xdot_full = state_action['target_qpos'][:T, :]# np.hstack((state_action['qpos'], state_action['qvel']))  
                loaded_traj.images = np.zeros((T, img_height, img_width, 3), dtype = 'uint8')

                good_lift = False
                total_ctr += 1
                if np.sum(np.abs(goal_pos)) > 0:
                    good_lift = True
                    good_lift_ctr += 1

                #object_poses = state_action['object_full_pose']
                #lift_mask = np.sum(object_poses[2:, :, 2] >= 0.12, axis = 0).astype(np.bool)
                #if np.any(lift_mask):
                #    if any(np.amax(object_poses[2:,lift_mask,2], axis = 1) - np.amin(object_poses[2:,lift_mask,2], axis=1) >= 0.05):
                #        good_lift = True
                #        good_lift_ctr += 1
                #        print('traj', t, 'is good!')
                
                for i in range(T):
                    img = cv2.imread(t + '/images/im{}.png'.format(i))[:, :, ::-1]
                    loaded_traj.images[i] = img

                    #img2 = cv2.imread(t + '/images1/im{}.png'.format(i))[:, :, ::-1]
                    #loaded_traj.images[i, 1] = img2

                    if good_lift and all(np.isclose(goal_pos, loaded_traj.X_Xdot_full[i, :2], atol=1e-3)) and loaded_traj.X_Xdot_full[i, 2] ==-0.08 and loaded_traj.goal_image is None:
                        loaded_traj.goal_image = img
    
                if loaded_traj.goal_image is None:
                    loaded_traj.goal_image = cv2.imread(t + '/images/im{}.png'.format(0))[:, :, ::-1]
                    
                    if good_lift:
                        print('NO DROP')
                        good_lift_ctr -=1
                        good_lift = False
                    
                if good_lift:
                    good_traj_list.append(loaded_traj)
                else:
                    bad_traj_list.append(loaded_traj)

            if len(good_traj_list) % traj_per_file == 0 and len(good_traj_list) > 0:
                f_name = 'good_traj_{0}_to_{1}'.format(num_good_saved * traj_per_file, (num_good_saved + 1) * traj_per_file - 1)
                save_tf_record(f_name, good_traj_list, agent_config)
                good_traj_list = []

                num_good_saved += 1
            elif len(bad_traj_list) % traj_per_file == 0 and len(bad_traj_list) > 0:
                f_name = 'bad_traj_{0}_to_{1}'.format(num_bad_saved * traj_per_file, (num_bad_saved + 1) * traj_per_file - 1)
                save_tf_record(f_name, bad_traj_list, agent_config)
                bad_traj_list = []

                num_bad_saved += 1

    if  len(good_traj_list) > 0:
        f_name = 'good_traj_{0}_to_{1}'.format(num_good_saved * traj_per_file, (num_good_saved + 1) * traj_per_file - 1)
        save_tf_record(f_name, good_traj_list, agent_config)
        good_traj_list = []

        num_good_saved += 1
    elif len(bad_traj_list) > 0:
        f_name = 'bad_traj_{0}_to_{1}'.format(num_bad_saved * traj_per_file, (num_bad_saved + 1) * traj_per_file - 1)
        save_tf_record(f_name, bad_traj_list, agent_config)
        bad_traj_list = []

        num_bad_saved += 1        
    print('perc good_lift', good_lift_ctr / total_ctr)

if __name__ == '__main__':
    main()
