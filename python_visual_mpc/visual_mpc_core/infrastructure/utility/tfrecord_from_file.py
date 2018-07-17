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
import random
import tensorflow as tf

import shutil

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tf_record(filename, trajectory_list):
    """
    saves data_files from one sample trajectory into one tf-record file
    """
    filename = filename + '.tfrecords'
    print(filename)
    writer = tf.python_io.TFRecordWriter(filename)
    feature = {}

    for traj in trajectory_list:
        for tind, feats in enumerate(traj):
            for k in feats:
                feature['{}/{}'.format(tind, k)] = feats[k]

    #
    #     feature[str(tind) + '/action']= _float_feature(traj.actions[tind,:].tolist())
    #
    #     arr = np.concatenate([traj.X_full[tind,:5], traj.touch_sensors[tind]], axis=0)
    #     feature[str(tind) + '/state'] = _float_feature(arr.tolist())
    #
    #     if 'cameras' in agentparams:
    #         for i in range(len(agentparams['cameras'])):
    #             image_raw = traj.images[tind, i].tostring()
    #             feature[str(tind) + '/image_view{}/encoded'.format(i)] = _bytes_feature(image_raw)
    #     else:
    #         if 'store_video_prediction' in agentparams:
    #             image_raw = traj.final_predicted_images[tind].tostring()
    #         else:
    #             image_raw = traj.images[tind].tostring()
    #         feature[str(tind) + '/image_view0/encoded'] = _bytes_feature(image_raw)
    #
    #     if hasattr(traj, 'Object_pose'):
    #         Object_pos_flat = traj.Object_pose[tind].flatten()
    #         feature['move/' + str(tind) + '/object_pos'] = _float_feature(Object_pos_flat.tolist())
    #
    #         if hasattr(traj, 'max_move_pose'):
    #             max_move_pose = traj.max_move_pose[tind].flatten()
    #             feature['move/' + str(tind) + '/max_move_pose'] = _float_feature(max_move_pose.tolist())
    #
    #     if hasattr(traj, 'gen_images'):
    #         feature[str(tind) + '/gen_images'] = _bytes_feature(traj.gen_images[tind].tostring())
    #         feature[str(tind) + '/gen_states'] = _float_feature(traj.gen_states[tind,:].tolist())
    #
    # if hasattr(traj, 'goal_image'):
    #     feature['/goal_image'] = _bytes_feature(traj.goal_image.tostring())
    #
    # if hasattr(traj, 'first_last_noarm'):
    #     feature['/first_last_noarm0'] = _bytes_feature(traj.first_last_noarm[0].tostring())
    #     feature['/first_last_noarm1'] = _bytes_feature(traj.first_last_noarm[1].tostring())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


class RecordSaver:
    def __init__(self, traj_per_file, save_dir, split = (0.90, 0.05, 0.05)):
        dirs_to_create = ['{}/{}'.format(save_dir, d) for d in ['train', 'test', 'val']]
        for d in dirs_to_create:
            print('Creating dir:', d)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        self._base_dir = save_dir
        self._train_val_test = split
        self._traj_buffers = [[] for _ in range(3)]
        self._save_counters = [0 for _ in range(3)]
        self._traj_per_file = traj_per_file

    def add_traj(self, traj):
        draw = None
        for i, buffer in enumerate(self._traj_buffers):
            if self._save_counters[i] == 0 and np.random.randint(0, 3) >= 1:
                draw = i
                continue

        if draw is None:
            draw = np.random.choice([0, 1, 2], 1, p=self._train_val_test)[0]

        self._traj_buffers[draw].append(traj)
        self._save()

    def flush(self):
        self._save(True)

    def __len__(self):
        return sum(self._save_counters)

    def _save(self, flush = False):
        for i, name in zip(range(3), ['train', 'test', 'val']):
            buffer = self._traj_buffers[i]
            if len(buffer) == 0:
                continue
            elif flush or len(buffer) % self._traj_per_file == 0:
                next_counter = self._save_counters[i] + len(buffer)
                folder = '{}/{}'.format(self._base_dir, name)
                file = '{}/traj_{}_to_{}'.format(folder, self._save_counters[i], next_counter - 1)
                save_tf_record(file, buffer)

                self._traj_buffers[i] = []
                self._save_counters[i] = next_counter



def check_lift(finger_sensors, gripper_z):
    max_touch = np.max(finger_sensors, axis = 1)
    print(max_touch.shape)
    return any(np.logical_and(finger_sensors > 0, gripper_z > 0.2))

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
    parser.add_argument('-s', action='store', dest='scale_img', type=int, default = 1, help='Scale cam images by this factor')

    args = parser.parse_args()
    hyperparams_file = args.experiment
    out_dir = args.output

    data_coll_dir = '/'.join(hyperparams_file.split('/')[:-1])

    if sys.version_info[0] == 2:
        hyperparams = imp.load_source('hyperparams', args.experiment)
        hyperparams = hyperparams.config
        #python 2 means we're executing on sawyer. add dummy camera list
        hyperparams['agent']['cameras'] = ['front_cam', 'left_cam']
    else:
        loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        conf = importlib.util.module_from_spec(spec)
        loader.exec_module(conf)
        hyperparams = conf.config

    traj_per_file = hyperparams['traj_per_file']
    agent_config = hyperparams['agent']
    T = agent_config['T']
    data_dir = agent_config['data_save_dir'] + '/train'
    out_dir = data_coll_dir + '/' + out_dir

    print('loading from', data_dir)
    print('saving to', out_dir)

    good_saver = RecordSaver(traj_per_file, '{}/good'.format(out_dir))
    bad_saver = RecordSaver(traj_per_file, '{}/bad'.format(out_dir))

    traj_group_dirs = glob.glob(data_dir+'/*')
    for g in traj_group_dirs:
        trajs = glob.glob(g + '/*')
        random.shuffle(trajs)
        
        for t in trajs:
            if not os.path.exists(t + '/obs_dict.pkl') or not os.path.exists(t + '/policy_out.pkl'):
                print('traj {} missing data'.format(t))
                continue

            try:
                obs_dict = pkl.load(open(t + '/obs_dict.pkl', 'rb'))
                policy_out = pkl.load(open(t + '/policy_out.pkl', 'rb'))
            except EOFError:
                print('traj {} missing data'.format(t))
                continue

            valid = True
            for i in range(len(agent_config['cameras'])):
                img_files = [t + '/images{}/im_{}.png'.format(i, j) for j in range(T)]
                if not all([os.path.exists(i) and os.path.isfile(i) for i in img_files]):
                    valid = False
                    print('traj {} missing /images{}'.format(t, i))
                    break
            if not valid:
                continue

            obs_dict.pop('term_t')

            loaded_traj = []
            policy_keys, obs_keys = list(policy_out[0].keys()), list(obs_dict.keys())
            good_lift = any(np.logical_and(np.max(obs_dict['finger_sensors'][:-1], 1) > 0, obs_dict['state'][:-1,2] > 0.2))

            for i in range(T):
                step_dict = {}
                for k in policy_out[i].keys():
                    step_dict['policy/{}'.format(k)] = float_feature(policy_out[i][k].flatten().tolist())
                for k in obs_keys:
                    step_dict['env/{}'.format(k)] = float_feature(obs_dict[k][i].flatten().tolist())

                for id, c in enumerate(['maincam', 'leftcam']):
                    img = cv2.imread(t + '/images{}/im_{}.png'.format(id, i))[:, :, ::-1].copy()
                    step_dict['image_view{}/encoded'.format(id)] = bytes_feature(img.tostring())

                loaded_traj.append(step_dict)
            if good_lift:
                good_saver.add_traj(loaded_traj)
            else:
                bad_saver.add_traj(loaded_traj)
    good_saver.flush()
    bad_saver.flush()
        
    print('perc good_lift', float(len(good_saver)) / (len(good_saver) + len(bad_saver)))

if __name__ == '__main__':
    main()
