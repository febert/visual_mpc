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
import moviepy.editor as mpy
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

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


class RecordSaver:
    def __init__(self, traj_per_file, save_dir, save_gif=False, split = (0.90, 0.05, 0.05)):
        dirs_to_create = ['train', 'test', 'val']
        if save_gif:
            dirs_to_create.append('gifs')
            self._gif_ctr = 0

        dirs_to_create = ['{}/{}'.format(save_dir, d) for d in dirs_to_create]
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
        self._keys = None

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

    def add_entry(self, key, shape):
        if self._keys is None:
            self._keys = {}
        self._keys[key] = shape

    def save_gif(self, clip):
        clip = mpy.ImageSequenceClip(clip, fps=10)
        clip.write_gif('{}/gifs/clip{}.gif'.format(self._base_dir, self._gif_ctr))
        self._gif_ctr += 1

    def save_manifest(self):
        if self._keys is None:
            raise ValueError
        with open('{}/manifest.txt'.format(self._base_dir), 'w') as f:
            f.write('# DATA MANIFEST\n')
            for key in self._keys:
                shape, shape_str = self._keys[key], ''
                for s in shape:
                    shape_str += ' {},'.format(s)

                f.write('{}, ({})\n'.format(key, shape_str[1:-1]))

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
                self._keys = save_tf_record(file, buffer)

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

    good_saver = RecordSaver(traj_per_file, '{}/good'.format(out_dir), save_gif=True)
    bad_saver = RecordSaver(traj_per_file, '{}/bad'.format(out_dir))

    created_manifest = False
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

            obs_z_thresh = np.logical_and(np.amax(obs_dict['object_poses_full'][:-1, :, 2], 1) > 0.15,
                                                  obs_dict['state'][:-1, 2] > 0.23)
            finger_sensors_thresh = np.max(obs_dict['finger_sensors'][:-1], 1) > 0
            good_lift = np.sum(np.logical_and(finger_sensors_thresh, obs_z_thresh)) >= 2
            front_cam = []
            for i in range(T):
                step_dict = {}
                if not created_manifest:
                    good_saver.add_entry('action', policy_out[i]['actions'].shape)
                    good_saver.add_entry('endeffector_pos', obs_dict['state'][i].shape)
                    img = cv2.imread(t + '/images{}/im_{}.png'.format(0, i))
                    good_saver.add_entry('image_view0/encoded', img.shape)
                    good_saver.add_entry('image_view1/encoded', img.shape)
                    created_manifest = True
                    good_saver.save_manifest()

                step_dict['action'] = float_feature(policy_out[i]['actions'].flatten().tolist())
                step_dict['endeffector_pos'] = float_feature(obs_dict['state'][i].flatten().tolist())
                # for k in policy_out[i].keys():
                #     step_dict['policy/{}'.format(k)] = float_feature(policy_out[i][k].flatten().tolist())
                # for k in obs_keys:
                #     step_dict['env/{}'.format(k)] = float_feature(obs_dict[k][i].flatten().tolist())

                for id, c in enumerate(['maincam', 'leftcam']):
                    img = cv2.imread(t + '/images{}/im_{}.png'.format(id, i))[:, :, ::-1].copy()
                    if id == 0 and good_lift:
                        front_cam.append(img)
                    step_dict['image_view{}/encoded'.format(id)] = bytes_feature(img.tostring())

                loaded_traj.append(step_dict)
            if good_lift:
                good_saver.add_traj(loaded_traj)
                good_saver.save_gif(front_cam)
            else:
                bad_saver.add_traj(loaded_traj)
    good_saver.flush()
    bad_saver.flush()
        
    print('perc good_lift', float(len(good_saver)) / (len(good_saver) + len(bad_saver)))

if __name__ == '__main__':
    main()
