import numpy as np
import tensorflow as tf
import random
import os
import cPickle
from collections import namedtuple
from python_visual_mpc.data_preparation.gather_data import make_traj_name_list, crop_and_rot
import ray
import re
import sys
import glob
import pdb
import itertools
import threading
import imp
import logging


class Trajectory(object):
    def __init__(self, conf):

        if 'total_num_img' in conf:
            total_num_img = conf['total_num_img']
        else: total_num_img = 96 #the actual number of images in the trajectory (for softmotion total_num_img=30)

        self.traj_per_group = 1000
        self.n_cam = len(conf['sourcetags'])  # number of cameras

        h = conf['target_res'][0]
        w = conf['target_res'][1]

        self.T = conf['sequence_length']
        self.images = np.zeros((self.T, self.n_cam, h, w, 3), dtype = np.float32)  # n_cam=0: main, n_cam=1: aux1

        action_dim = conf['adim']  # (for softmotion action_dim=4)
        state_dim = conf['sdim']  # (for softmotion action_dim=4)
        self.actions = np.zeros((self.T, action_dim), dtype = np.float32)
        self.endeffector_pos = np.zeros((self.T, state_dim), dtype = np.float32)
        self.joint_angles = np.zeros((self.T, 7), dtype = np.float32)


def reading_thread(conf, subset_traj, enqueue_op, sess, images_pl, states_pl, actions_pl):
    num_errors = 0
    conf = conf
    data_conf = conf['data_configuration']
    sourcetags = data_conf['sourcetags']
    print 'started process with PID:', os.getpid()

    def step_from_to(i_src, traj_dir_src, traj, pkldata):
        trajind = 0  # trajind is the index in the target trajectory
        end = data_conf['total_num_img']

        t_ev_nstep = data_conf['take_ev_nth_step']


        smp_range = end // t_ev_nstep  - conf['sequence_length']

        if 'shift_window' in conf:
            print 'performing shifting in time'
            start = np.random.random_integers(0, smp_range) * t_ev_nstep
        else:
            start = 0

        end = start + conf['sequence_length'] * t_ev_nstep

        all_actions = pkldata['actions']
        all_joint_angles = pkldata['jointangles']
        all_endeffector_pos = pkldata['endeffector_pos']

        for dataind in range(start, end, t_ev_nstep):  # dataind is the index in the source trajetory
            # get low dimensional data
            traj.actions[trajind] = all_actions[dataind]
            traj.joint_angles[trajind] = all_joint_angles[dataind]
            traj.endeffector_pos[trajind] = all_endeffector_pos[dataind]

            if 'imagename_no_zfill' in conf:
                im_filename = traj_dir_src + '/images/{0}_full_cropped_im{1}_*' \
                    .format(sourcetags[i_src], dataind)
            else:
                im_filename = traj_dir_src + '/images/{0}_full_cropped_im{1}_*' \
                    .format(sourcetags[i_src], str(dataind).zfill(2))
            # if dataind == 0:
            #     print 'processed from file {}'.format(im_filename)
            # if dataind == end - t_ev_nstep:
            #     print 'processed to file {}'.format(im_filename)

            file = glob.glob(im_filename)
            if len(file) > 1:
                raise ValueError
            file = file[0]

            im = crop_and_rot(data_conf, sourcetags, file, i_src)
            im = im.astype(np.float32)/255.

            traj.images[trajind, i_src] = im
            trajind += 1

        return traj

    for trajname in itertools.cycle(subset_traj):  # loop of traj0, traj1,..

        try:
            traj_index = re.match('.*?([0-9]+)$', trajname).group(1)
            traj = Trajectory(data_conf)

            traj_tailpath = '/'.join(str.split(trajname, '/')[-2:])   #string with only the group and trajectory
            traj_beginpath = '/'.join(str.split(trajname, '/')[:-3])   #string with only the group and trajectory

            pkl_file = trajname + '/joint_angles_traj{}.pkl'.format(traj_index)
            if not os.path.isfile(pkl_file):
                print 'no pkl file found in', trajname
                continue

            pkldata = cPickle.load(open(pkl_file, "rb"))

            for i_src, tag in enumerate(data_conf['sourcetags']):  # loop over cameras: main, aux1, ..
                traj_dir_src = traj_beginpath + tag + '/' + traj_tailpath
                step_from_to(i_src, traj_dir_src, traj, pkldata)

            sess.run(enqueue_op, feed_dict={images_pl: np.squeeze(traj.images),
                                            states_pl: traj.endeffector_pos,
                                            actions_pl: traj.actions})
        except KeyboardInterrupt:
            sys.exit()
        except:
            print "error occured"
            num_errors += 1

class OnlineReader(object):
    def __init__(self, conf, mode, sess):
        """
        :param conf:
        :param mode: training, validation, test
        """

        self.sess = sess
        self.conf = conf
        self.data_conf = conf['data_configuration']
        self.mode = mode

        self.images_pl = tf.placeholder(tf.float32, name='images', shape=(conf['sequence_length'], 64, 64, 3))
        adim = 5
        sdim = 4
        self.actions_pl = tf.placeholder(tf.float32, name='actions', shape=(conf['sequence_length'], adim))
        self.states_pl = tf.placeholder(tf.float32, name='states', shape=(conf['sequence_length'], sdim))

        if mode == 'train' or mode == 'val':
            self.num_threads = 1
        else: self.num_threads = 1

        if mode == 'test':
            self.shuffle = False
        else: self.shuffle = True

        self.q = tf.FIFOQueue(20, [tf.float32, tf.float32, tf.float32], shapes=[self.images_pl.get_shape().as_list(),
                                                                                 self.actions_pl.get_shape().as_list(),
                                                                                 self.states_pl.get_shape().as_list()])
        self.enqueue_op = self.q.enqueue([self.images_pl, self.actions_pl, self.states_pl])

        data_sets = self.search_data()
        combined_traj_list = self.combine_traj_lists(data_sets)
        self.start_threads(combined_traj_list)

    def get_batch_tensors(self):

        image_batch, action_batch, states_batch = self.q.dequeue_many(self.conf['batch_size'])
        return image_batch, action_batch, states_batch

    def search_data(self):

        print 'searching data'
        datasets = []
        for dir in self.data_conf['source_basedirs']:
            source_name = str.split(dir, '/')[-1]

            print 'preparing source_basedir', dir
            split_file = self.data_conf['current_dir'] + '/' + source_name + '_split.pkl'

            dataset_i = {}
            if os.path.isfile(split_file):
                print 'loading datasplit from ', split_file
                dataset_i = cPickle.load(open(split_file, "rb"))
            else:
                traj_list = make_traj_name_list(self.data_conf, [dir + self.data_conf['sourcetags'][0]],
                                                shuffle=True)

                #make train, val, test split
                test_traj = traj_list[:256]  # use first 256 for test
                traj_list = traj_list[256:]
                num_traj = len(traj_list)

                index = int(np.floor(self.conf['train_val_split'] * num_traj))

                train_traj = traj_list[:index]
                val_traj = traj_list[index:]

                dataset_i['source_basedir'] = dir
                dataset_i['train'] = train_traj
                dataset_i['val'] = val_traj
                dataset_i['test'] = test_traj

                cPickle.dump(dataset_i, open(split_file, 'wb'))

            datasets.append(dataset_i)

        return datasets

    def combine_traj_lists(self, datasets):
        combined = []
        for dset in datasets:
            # select whether to use train, val or test data
            dset = dset[self.mode]
            combined += dset
        if self.shuffle:
            random.shuffle(combined)
        return combined

    def start_threads(self, traj_list):
        """
            :param sourcedirs:
            :param tf_rec_dir:
            :param gif_dir:
            :param traj_name_list:
            :param crop_from_highres:
            :param start_end_grp: list with [startgrp, endgrp]
            :return:
            """
        # ray.init()

        itraj_start = 0
        n_traj = len(traj_list)

        traj_per_worker = int(n_traj / np.float32(self.num_threads))
        start_idx = [itraj_start + traj_per_worker * i for i in range(self.num_threads)]
        end_idx = [itraj_start + traj_per_worker * (i + 1) - 1 for i in range(self.num_threads)]

        for i in range(self.num_threads):
            print 'worker {} going from {} to {} '.format(i, start_idx[i], end_idx[i])
            subset_traj = traj_list[start_idx[i]:end_idx[i]]

            t = threading.Thread(target=reading_thread, args=(self.conf, subset_traj, self.enqueue_op, self.sess,
                                                 self.images_pl, self.states_pl, self.actions_pl))
            t.setDaemon(True)
            t.start()


def test_online_reader():

    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))

    conf['train_val_split'] = 0.95
    conf['sequence_length'] = 15  # 'sequence length, including context frames.'
    conf['batch_size'] = 10
    conf['context_frames'] = 2

    conf['img_height'] = 64
    conf['sdim'] = 3
    conf['adim'] = 4

    conf['current_dir'] = current_dir
    conf['shift_window'] = ''

    dataconf_file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/pushing_data/online_weiss/dataconf.py'
    data = imp.load_source('hyperparams', dataconf_file)
    conf['data_configuration'] = data.data_configuration
    conf['data_configuration']['sequence_length'] = conf['sequence_length']

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'testing the reader'

    sess = tf.InteractiveSession()
    r = OnlineReader(conf, 'train', sess=sess)
    image_batch, action_batch, endeff_pos_batch = r.get_batch_tensors()

    from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video

    for i_run in range(1):
        print 'run number ', i_run

        images, actions, endeff = sess.run([image_batch, action_batch, endeff_pos_batch])

        file_path = conf['data_configuration']['current_dir'] + '/preview'
        comp_single_video(file_path, images)

        # show some frames
        for b in range(conf['batch_size']):
            print 'actions'
            print actions[b]

            print 'endeff'
            print endeff[b]

            # print 'robot_pos'
            # print robot_pos
            #
            # print 'object_pos'
            # print object_pos

            # visualize_annotation(conf, images[b], robot_pos[b], object_pos[b])

            # images = np.squeeze(images)
            # img = np.uint8(255. * images[b, 0])
            # img = Image.fromarray(img, 'RGB')
            # # img.save(file_path,'PNG')
            # img.show()
            # print b

            pdb.set_trace()

if __name__ == '__main__':
    test_online_reader()