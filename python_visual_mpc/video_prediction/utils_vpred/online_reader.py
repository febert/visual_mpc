import numpy as np
import tensorflow as tf
import random
import os
import pickle
from collections import namedtuple
from python_visual_mpc.data_preparation.gather_data import make_traj_name_list, crop_and_rot
import re
import sys
import glob
import pdb
import itertools
import threading
import imp
import logging
import tarfile
import cv2
from collections import OrderedDict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
import time
import tarfile
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory

import copy

def get_start_end(conf):
    """
    get start and end time indices, which will be read from trajectory
    :param conf:
    :return:
    """
    if 'total_num_img' in conf:
        end = conf['total_num_img']
        t_ev_nstep = conf['take_ev_nth_step']
    else:
        t_ev_nstep = 1
        end = conf['sequence_length']

    smp_range = end // t_ev_nstep - conf['sequence_length']
    if 'shift_window' in conf:
        print('performing shifting in time')
        start = np.random.random_integers(0, smp_range) * t_ev_nstep
    else:
        start = 0
    end = start + conf['sequence_length'] * t_ev_nstep

    if 'take_ev_nth_step' in conf:
        take_ev_nth_step = conf['take_ev_nth_step']
    else: take_ev_nth_step = 1

    return start, end, take_ev_nth_step


def read_img(tag_dict, dataind, tar=None, trajname=None):
    """
    read a single image, either from tar-file or directly
    :param tar:  far file handle
    :param tag_dict: dictionary describing the tag
    :param dataind: the timestep in the data folder (may not be equal to the timestep used for the allocated array)
    :return:
    """
    if 'ncam' in tag_dict:
        images = []
        for icam in range(tag_dict['ncam']):
            images.append(read_single_img(dataind, tag_dict, tar, trajname, icam))
        img = np.stack(images, 0)
    else:
        img = read_single_img(dataind, tag_dict, tar, trajname)
    return img


def read_single_img(dataind, tag_dict, tar, trajname, icam=None):
    if tar != None:
        im_filename = 'traj/images/im{}.png'.format(dataind)
        img_stream = tar.extractfile(im_filename)
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    else:
        imfile = trajname + tag_dict['file'].format(dataind)
        if not os.path.exists(imfile):
            raise ValueError("file {} does not exist!".format(imfile))
        img = cv2.imread(imfile)
    imheight = tag_dict['shape'][0]  # get target im_sizes
    imwidth = tag_dict['shape'][1]
    if 'rowstart' in tag_dict:  # get target cropping if specified
        rowstart = tag_dict['rowstart']
        colstart = tag_dict['colstart']
    # setting used in wrist_rot
    if 'shrink_before_crop' in tag_dict:
        shrink_factor = tag_dict['shrink_before_crop']
        img = cv2.resize(img, (0, 0), fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_AREA)
        img = img[rowstart:rowstart + imheight, colstart:colstart + imwidth]
    # setting used in softmotion30_v1
    elif 'crop_before_shrink' in tag_dict:
        raw_image_height = img.shape[0]
        img = img[rowstart:rowstart + raw_image_height, colstart:colstart + raw_image_height]
        target_res = tag_dict['target_res']
        img = cv2.resize(img, target_res, interpolation=cv2.INTER_AREA)
    elif 'rowstart' in tag_dict:
        img = img[rowstart:rowstart + imheight, colstart:colstart + imwidth]

    img = img[:, :, ::-1]  # bgr => rgb
    img = img.astype(np.float32) / 255.
    return img


def reading_thread(conf, subset_traj, enqueue_op, sess, placeholders, use_tar):
    num_errors = 0
    print('started process with PID:', os.getpid())

    for trajname in itertools.cycle(subset_traj):  # loop of traj0, traj1,..
        nump_array_dict = read_trajectory(conf, trajname, use_tar=use_tar)

        # print 'reading ',trajname

        feed_dict = {}
        for tag_dict in conf['sourcetags']:
            tag_name = tag_dict['name']
            feed_dict[placeholders[tag_name]] = nump_array_dict[tag_name]

        t1 = time.time()
        sess.run(enqueue_op, feed_dict=feed_dict)

        # if traj_index % 10 == 0:
        #     print 't ful enqueu', time.time() - t0
        #     print 't enqueu run', time.time() - t1

        # except KeyboardInterrupt:
        #     sys.exit()
        # except:
        #     print "error occured"
        #     num_errors += 1


def read_trajectory(conf, trajname, use_tar = False):
    """
    the configuration file needs:
    source_basedirs key: a list of directories where to load the data from, data is concatenated (advantage: no renumbering needed when using multiple sources)
    sourcetags: a list of tags where each tag is a dict with
        # name: the name of the data field
        # shape: the target shape, will be cropped to match this shape
        # rowstart: starting row for cropping
        # rowend: end row for cropping
        # colstart: start column for cropping
        # shrink_before_crop: shrink image according to this ratio before cropping
        # brightness_threshold: if average pixel value lower discard video
        # not_per_timestep: if this key is there, load the data for this tag at once from pkl file

    :param conf:
    :param trajname: folder of trajectory to be loaded
    :param use_tar: whether to load from tar files
    """
    t0 = time.time()
    # try:
    traj_index = int(re.match('.*?([0-9]+)$', trajname).group(1))
    nump_array_dict = {}

    if use_tar:
        tar = tarfile.open(trajname + "/traj.tar")
        pkl_file_stream = tar.extractfile('traj/agent_data.pkl')
        pkldata = pickle.load(pkl_file_stream)
    else:
        tar = None
        pkldata = pickle.load(open(trajname + '/agent_data.pkl', 'rb'), encoding='latin1')

    for tag_dict in conf['sourcetags']:
        if 'not_per_timestep' not in tag_dict:
            numpy_arr = np.zeros([conf['sequence_length']] + tag_dict['shape'], dtype=np.float32)
            nump_array_dict[tag_dict['name']] = numpy_arr

    start, end, take_ev_nth_step = get_start_end(conf)

    # remove not_per_timestep tags
    filtered_source_tags = []
    for tag_dict in conf['sourcetags']:
        if 'not_per_timestep' in tag_dict:
            nump_array_dict[tag_dict['name']] = pkldata[tag_dict['name']]
        else:
            filtered_source_tags.append(tag_dict)

    trajind = 0
    for dataind in range(start, end, take_ev_nth_step):

        for tag_dict in filtered_source_tags:
            tag_name = tag_dict['name']

            if '.pkl' in tag_dict['file']:  # if it's data from Pickle file
                if 'pkl_names' in tag_dict:  # if a tag, e.g. the the state is split up into multiple tags
                    pklread0 = pkldata[tag_dict['pkl_names'][0]]
                    pklread1 = pkldata[tag_dict['pkl_names'][1]]
                    nump_array_dict[tag_name][trajind] = np.concatenate([pklread0[dataind], pklread1[dataind]],
                                                                        axis=0)
                else:
                    nump_array_dict[tag_name][trajind] = pkldata[tag_dict['name']][dataind]
            else:  # if it's image data
                nump_array_dict[tag_name][trajind] = read_img(tag_dict, dataind, trajname=trajname, tar=tar)
        trajind += 1

    if use_tar:
        tar.close() # important: close file
    return nump_array_dict



class OnlineReader(object):
    def __init__(self, conf, mode, sess, use_tar=False):
        """

        :param conf:
        :param mode:  'train': shuffle data or 'test': don't shuffle
        :param sess:
        """

        self.sess = sess
        self.conf = conf
        self.mode = mode

        self.use_tar = use_tar

        self.place_holders = OrderedDict()

        pl_shapes = []
        self.tag_names = []
        # loop through tags
        for tag_dict in conf['sourcetags']:
            if 'not_per_timestep' in tag_dict:
                pl_shapes.append(tag_dict['shape'])
            else:
                pl_shapes.append([conf['sequence_length']] + tag_dict['shape'])
            self.tag_names.append(tag_dict['name'])
            self.place_holders[tag_dict['name']] = tf.placeholder(tf.float32, name=tag_dict['name'], shape=pl_shapes[-1])
        if mode == 'train' or mode == 'val':
            self.num_threads = 10
        else: self.num_threads = 1

        if mode == 'test':
            self.shuffle = False
        else: self.shuffle = True

        tf_dtypes = [tf.float32]*len(pl_shapes)

        self.q = tf.FIFOQueue(1000, tf_dtypes, shapes=pl_shapes)
        self.enqueue_op = self.q.enqueue(list(self.place_holders.values()))

        auto_split = False  # automatically divide dataset into train, val, test and save the split to pkl-file
        if auto_split:
            data_sets = self.search_data()
            self.traj_list = self.combine_traj_lists(data_sets)
        else:
            self.traj_list = make_traj_name_list(conf, shuffle=self.shuffle)

        self.start_threads(self.traj_list)

    def get_batch_tensors(self):
        tensor_list = self.q.dequeue_many(self.conf['batch_size'])
        return tensor_list


    def search_data(self):
        """
        automatically divide dataset into train, val, test and save the split to pkl-file;
        if pkl-file already exists load the split
        :return: train, val, test datasets for every source
        """

        print('searching data')
        datasets = []
        for dir in self.conf['source_basedirs']:
            source_name = str.split(dir, '/')[-1]

            print('preparing source_basedir', dir)
            split_file = self.conf['current_dir'] + '/' + source_name + '_split.pkl'

            dataset_i = {}
            if os.path.isfile(split_file):
                print('loading datasplit from ', split_file)
                dataset_i = pickle.load(open(split_file, "rb"))
            else:
                traj_list = make_traj_name_list(self.conf, shuffle=True)

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

                pickle.dump(dataset_i, open(split_file, 'wb'))

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
            print('worker {} going from {} to {} '.format(i, start_idx[i], end_idx[i]))
            subset_traj = traj_list[start_idx[i]:end_idx[i]]

            t = threading.Thread(target=reading_thread, args=(self.conf, subset_traj,
                                                              self.enqueue_op, self.sess,
                                                              self.place_holders, self.use_tar
                                                                ))
            t.setDaemon(True)
            t.start()

def convert_to_tfrec(sourcedir, destdir):
    tag_images = {'name': 'images',
                  'file': '/images/im{}_cam{}.png',  # only tindex
                  'shape': [48, 64, 3],
                  'ncam':2
                  }

    tag_actions = {'name': 'states',
                   'file': '/state_action.pkl',  # only tindex
                   'pkl_names': ['qpos', 'qvel'],
                   'shape': [12],
                   }

    tag_states = {'name': 'actions',
                  'file': '/state_action.pkl',  # only tindex
                  'shape': [5],
                  }
    traj_conf = {
        'batch_size':64,
        'sequence_length':15,
        'ngroup': 1000,
        'image_height':48,
        'image_width':64,
        ''
        'sourcetags': [tag_images, tag_actions, tag_states],
        'source_basedirs': [sourcedir],
        # 'current_dir':os.environ['VMPC_DATA_DIR'] + '/datacol_appflow/',
        'data_save_dir':destdir,
    }

    traj_conf['T'] = traj_conf['sequence_length']
    trajlist = make_traj_name_list(traj_conf, shuffle=False)

    tag_images = {'name': 'images',
                  'file':'/images/im{}.png',   # only tindex
                  'shape':[48, 64, 3]}
    goal_img_conf = {
        'ngroup': 1000,
        'sequence_length':2,
        'sourcetags':[tag_images],
        'source_basedirs':[os.environ['VMPC_DATA_DIR'] + '/cartgripper_startgoal_masks6e4/train']}

    trajectory_list = []
    itr = 0
    for trajname in trajlist:
        traj = read_trajectory(traj_conf, trajname)
        goal_img_traj = read_trajectory(goal_img_conf,
                                        goal_img_conf['source_basedirs'][0] + '/' + '/'.join(str.split(trajname, '/')[-2:]))

        t = Trajectory(traj_conf)
        t._sample_images = (traj['images']*255.).astype(np.uint8)
        t.actions = traj['actions']
        t.X_Xdot_full = traj['states']
        t.goal_image = (goal_img_traj['images'][-1]*255.).astype(np.uint8)

        t = copy.deepcopy(t)
        trajectory_list.append(t)
        traj_per_file = 128
        print('traj_per_file', traj_per_file)
        if len(trajectory_list) == traj_per_file:
            filename = 'traj_{0}_to_{1}' \
                .format(itr - traj_per_file + 1, itr)
            save_tf_record(filename, trajectory_list, traj_conf)
            trajectory_list = []
        itr += 1

def test_online_reader():

    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])

    # hyperparams_file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/pushing_data/cartgripper_rgb/hyperparams.py'
    # hyperparams = imp.load_source('hyperparams', hyperparams_file)
    # conf = hyperparams.config
    #
    # conf['batch_size'] = 10
    #
    # print '-------------------------------------------------------------------'
    # print 'verify current settings!! '
    # for key in conf.keys():
    #     print key, ': ', conf[key]
    # print '-------------------------------------------------------------------'
    #
    # print 'testing the reader'
    tag_images = {'name': 'images',
                  'file': '/images/im{}_cam{}.png',  # only tindex
                  'shape': [48, 64, 3],
                  }

    tag_actions = {'name': 'states',
                  'file': '/state_action.pkl',  # only tindex
                  'pkl_names': ['qpos', 'qvel'],
                   'shape': [12],
                  }

    tag_states = {'name': 'actions',
                  'file': '/state_action.pkl',  # only tindex
                  'shape': [6],
                  }
    conf = {
        'batch_size':64,
        'sequence_length':30,
        'ngroup': 1000,
        'sourcetags': [tag_images, tag_actions, tag_states],
        'source_basedirs': [os.environ['VMPC_DATA_DIR'] + '/datacol_appflow/data/train'],
        # 'source_basedirs': [os.environ['VMPC_DATA_DIR'] + '/cartgripper_gtruth_flow/train'],
        'current_dir':os.environ['VMPC_DATA_DIR'] + '/datacol_appflow/',
        'ncam':2
    }

    sess = tf.InteractiveSession()
    r = OnlineReader(conf, 'test', sess=sess, use_tar=False)
    # image_batch, gtruth_flows_batch = r.get_batch_tensors()
    image_batch, action_batch, endeff_pos_batch = r.get_batch_tensors()

    from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video

    deltat = []
    end = time.time()
    for i_run in range(1000):
        # print 'run number ', i_run

        # images, gtruth_flows = sess.run([image_batch, gtruth_flows_batch])
        images, actions, endeff = sess.run([image_batch, action_batch, endeff_pos_batch])

        deltat.append(time.time() - end)
        if i_run % 10 == 0:
            print('tload{}'.format(time.time() - end))
            print('average time:', np.average(np.array(deltat)))
        end = time.time()

        # file_path = conf['current_dir'] + '/preview'
        # comp_single_video(file_path, images, num_exp=conf['batch_size'])
        # pdb.set_trace()

        # show some frames
        # for b in range(conf['batch_size']):
        #     print 'actions'
        #     print actions[b]
        #
        #     print 'endeff'
        #     print endeff[b]
        #
        #     plt.imshow(images[b,0])
        #     plt.show()

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

            # pdb.set_trace()

if __name__ == '__main__':
    # test_online_reader()
    convert_to_tfrec('/mnt/sda1/pushing_data/mj_pos_noreplan_fast/train', '/mnt/sda1/pushing_data/mj_pos_noreplan_fast_tfrec/train')
