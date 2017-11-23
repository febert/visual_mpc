import cPickle
import copy
import glob
import os
import random

import imutils  # pip install imutils
import numpy as np
import tensorflow as tf
from PIL import Image


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


import cv2
import ray
import create_gif
import argparse
import sys
import imp
import pdb


class More_than_one_image_except(Exception):
    def __init__(self, imagefile):
        self.image_file = imagefile
    def __str__(self):
      return self.image_file

class Trajectory(object):
    def __init__(self, conf):

        if 'total_num_img' in conf:
            self.total_num_img = conf['total_num_img']
        else: self.total_num_img = 96 #the actual number of images in the trajectory (for softmotion total_num_img=30)

        self.traj_per_group = 1000

        if 'take_ev_nth_step' in conf:
            self.take_ev_nth_step = conf['take_ev_nth_step']
        else: self.take_ev_nth_step = 2 #only use every n-step from the data (for softmotion take_ev_nth_step=1)
        split_seq_by = 1  #if greater than 1 split trajectory in n equal parts

        self.npictures = self.total_num_img/split_seq_by  #number of images after splitting (include images we use and not use)

        self.cameranames = ['main']
        self.n_cam = len(self.cameranames)  # number of cameras

        self.T = self.total_num_img / split_seq_by / self.take_ev_nth_step  # the number of timesteps in final trajectory

        h = conf['target_res'][0]
        w = conf['target_res'][1]
        self.images = np.zeros((self.T, self.n_cam, h, w, 3), dtype = np.uint8)  # n_cam=0: main, n_cam=1: aux1
        self.dimages = np.zeros((self.T, self.n_cam, h, w), dtype = np.uint8)
        self.dvalues = np.zeros((self.T, self.n_cam, h, w), dtype = np.float32)

        action_dim = conf['adim']  # (for softmotion action_dim=4)
        state_dim = conf['sdim']  # (for softmotion action_dim=4)
        self.actions = np.zeros((self.T, action_dim), dtype = np.float32)
        self.endeffector_pos = np.zeros((self.T, state_dim), dtype = np.float32)
        self.joint_angles = np.zeros((self.T, 7), dtype = np.float32)

@ray.remote
class TF_rec_converter(object):
    def __init__(self, conf,
                       gif_dir= None,
                       traj_name_list = None,
                       tf_start_ind = None,
                       crop_from_highres = False):
        """
        :param sourcedirs:  directory where the raw data lies
        :param tf_rec_dir:  director where to save tf records
        :param gif_dir:  where to save gif
        :param traj_name_list: a list of trajectory-folders
        :param tf_start_ind: the starting index to use in the tf-record filename
        :param crop_from_highres: whether to crop the image from the full resolution image
        """

        self.sourcedirs = conf['source_basedirs']
        self.tfrec_dir = conf['tf_rec_dir']
        self.gif_dir = gif_dir
        self.crop_from_highres = crop_from_highres
        self.tf_start_ind = tf_start_ind
        self.traj_name_list = traj_name_list
        self.conf = conf
        print 'started process with PID:', os.getpid()


    def gather(self):
        print

        donegif = False
        i_more_than_one_image = 0
        num_errors = 0

        nopkl_file = 0

        for dirs in self.sourcedirs:
            if not os.path.exists(dirs):
                raise ValueError('path {} does not exist!'.format(dirs))
        if not os.path.exists(self.tfrec_dir):
            raise ValueError('path {} does not exist!'.format(self.tfrec_dir))

        self.src_names = [str.split(n, '/')[-1] for n in self.sourcedirs]

        traj_list = []
        ntraj_gifmax = 8


        tf_start_ind = self.tf_start_ind
        for trajname in self.traj_name_list:  # loop of traj0, traj1,..

            # print 'processing {}, seq-part {}'.format(trajname, traj_tuple[1] )
            try:
                traj_index = re.match('.*?([0-9]+)$', trajname).group(1)
                self.traj = Trajectory(self.conf)

                traj_tailpath = '/'.join(str.split(trajname, '/')[-2:])
                traj_beginpath = '/'.join(str.split(trajname, '/')[:-3])

                #load actions:
                if 'pkl_source' in self.conf:
                    trajname_pkl = copy.deepcopy(trajname).split('/')
                    trajname_pkl[5] = self.conf['pkl_source']
                    trajname_pkl = '/'.join(trajname_pkl)
                    pkl_file = trajname_pkl + '/joint_angles_traj{}.pkl'.format(traj_index)
                else:
                    pkl_file = trajname + '/joint_angles_traj{}.pkl'.format(traj_index)
                if not os.path.isfile(pkl_file):
                    nopkl_file += 1
                    print 'no pkl file found, file no: ', nopkl_file
                    continue

                pkldata = cPickle.load(open(pkl_file, "rb"))
                self.all_actions = pkldata['actions']
                assert self.all_actions.shape[0] == Trajectory(self.conf).total_num_img
                self.all_joint_angles = pkldata['jointangles']
                assert self.all_joint_angles.shape[0] == Trajectory(self.conf).total_num_img
                self.all_endeffector_pos = pkldata['endeffector_pos']
                assert self.all_endeffector_pos.shape[0] == Trajectory(self.conf).total_num_img

                try:
                    for i_src, tag in enumerate(self.conf['sourcetags']):  # loop over cameras: main, aux1, ..
                        traj_dir_src = traj_beginpath + tag + '/' + traj_tailpath
                        self.step_from_to(i_src, traj_dir_src)
                except More_than_one_image_except as e:
                    print "more than one image in ", e.image_file
                    i_more_than_one_image += 1
                    continue

                traj_list.append(self.traj)
                maxlistlen = 128

                if maxlistlen == len(traj_list):

                    filename = 'traj_{0}_to_{1}' \
                        .format(tf_start_ind,tf_start_ind + maxlistlen-1)
                    self.save_tf_record(filename, traj_list)
                    tf_start_ind += maxlistlen
                    traj_list = []

                if self.gif_dir != None and not donegif:
                    if len(traj_list) == ntraj_gifmax:
                        create_gif.comp_video(traj_list, self.gif_dir + 'worker{}'.format(os.getpid()))
                        print 'created gif, exiting'
                        donegif = True

                print 'processed {} trajectories'.format(len(traj_list))
            except KeyboardInterrupt:
                sys.exit()
            except:
                print "error occured"
                num_errors += 1

        print 'done, {} more_than_one_image occurred:'.format(i_more_than_one_image)
        print 'done, {} errors occurred:'.format(num_errors)

        return 'done'


    def step_from_to(self, i_src, trajname):
        trajind = 0  # trajind is the index in the target trajectory
        end = Trajectory(self.conf).npictures
        for dataind in range(0, end, self.traj.take_ev_nth_step):  # dataind is the index in the source trajetory

            # get low dimensional data
            self.traj.actions[trajind] = self.all_actions[dataind]
            self.traj.joint_angles[trajind] = self.all_joint_angles[dataind]
            self.traj.endeffector_pos[trajind] = self.all_endeffector_pos[dataind]

            # getting color image:
            if self.crop_from_highres:
                if 'imagename_no_zfill' in self.conf:
                    im_filename = trajname + '/images/{0}_full_cropped_im{1}_*' \
                        .format(self.conf['sourcetags'][i_src], dataind)
                else:
                    im_filename = trajname + '/images/{0}_full_cropped_im{1}_*'\
                        .format(self.conf['sourcetags'][i_src], str(dataind).zfill(2))
            else:
                im_filename = self.traj_dir_src + '/images/{0}_cropped_im{1}_*.png'\
                    .format(trajname, dataind)

            if dataind == 0:
                print 'processed from file {}'.format(im_filename)
            if dataind == end - self.traj.take_ev_nth_step:
                print 'processed to file {}'.format(im_filename)

            file = glob.glob(im_filename)
            if len(file) > 1:
                raise More_than_one_image_except(im_filename)
            file = file[0]

            im = crop_and_rot(self.conf, file, i_src)

            self.traj.images[trajind, i_src] = im

            trajind += 1



    def save_tf_record(self, filename, trajectory_list):
        """
        saves data_files from one sample trajectory into one tf-record file
        """

        filename = os.path.join(self.tfrec_dir, filename + '.tfrecords')
        print('Writing', filename)

        if not os.path.exists(self.tfrec_dir):
            os.makedirs(self.tfrec_dir)

        writer = tf.python_io.TFRecordWriter(filename)

        feature = {}

        for tr in range(len(trajectory_list)):

            traj = trajectory_list[tr]
            sequence_length = traj.T

            for tstep in range(sequence_length):

                feature[str(tstep) + '/action']= _float_feature(traj.actions[tstep].tolist())
                feature[str(tstep) + '/endeffector_pos'] = _float_feature(traj.endeffector_pos[tstep].tolist())

                image_raw = traj.images[tstep, 0].tostring()  # for camera 0, i.e. main
                feature[str(tstep) + '/image_view0/encoded'] = _bytes_feature(image_raw)

                if Trajectory(self.conf).n_cam == 2:
                    image_raw = traj.images[tstep, 1].tostring()  # for camera 1, i.e. aux1
                    feature[str(tstep) + '/image_view1/encoded'] = _bytes_feature(image_raw)

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

def crop_and_rot(conf, file, i_src):
    img = cv2.imread(file)
    imheight = conf['target_res'][0]
    imwidth = conf['target_res'][1]

    rowstart = conf['rowstart']
    colstart = conf['colstart']
    # setting used in wrist_rot
    if 'shrink_before_crop' in conf:
        shrink_factor = conf['shrink_before_crop']
        img = cv2.resize(img, (0, 0), fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_AREA)
        img = img[rowstart:rowstart+imheight, colstart:colstart+imwidth]

    # setting used in softmotion30_v1
    elif 'crop_before_shrink' in conf:
        raw_image_height = conf['raw_image_height']
        img = img[rowstart:rowstart + raw_image_height, colstart:colstart + raw_image_height]
        # plt.imshow(img)
        # plt.show()
        target_res = conf['target_res']
        img = cv2.resize(img, target_res, interpolation=cv2.INTER_AREA)
    else:
        raise NotImplementedError

    # assert img.shape == (64,64,3)
    img = img[...,::-1]  #bgr => rgb

    if conf['sourcetags'][i_src] == 'aux1':
        img = imutils.rotate_bound(img, 180)

    # plt.imshow(img)
    # plt.show()
    return img

def get_maxtraj(sourcedirs):
    for dirs in sourcedirs:
        if not os.path.exists(dirs):
            raise ValueError('path {} does not exist!'.format(dirs))

    # go to first source and get group names:
    groupnames = glob.glob(os.path.join(sourcedirs[0], '*'))
    groupnames = [str.split(n, '/')[-1] for n in groupnames]
    gr_ind = []
    for grname in groupnames:
        try:
            gr_ind.append(int(re.match('.*?([0-9]+)$', grname).group(1)))
        except:
            continue
    max_gr = np.max(np.array(gr_ind))

    trajdir = sourcedirs[0] + "/traj_group{}".format(max_gr)
    trajname_l = glob.glob(trajdir +'/*')
    trajname_l = [str.split(n, '/')[-1] for n in trajname_l]
    traj_num = []
    for trname in trajname_l:
        traj_num.append(int(re.match('.*?([0-9]+)$', trname).group(1)))

    max_traj = np.max(np.array(traj_num))
    return max_traj


def start_parallel(conf, gif_dir, traj_name_list, n_workers, crop_from_highres= True, start_end_grp = None):
    """
    :param sourcedirs:
    :param tf_rec_dir:
    :param gif_dir:
    :param traj_name_list:
    :param crop_from_highres:
    :param start_end_grp: list with [startgrp, endgrp]
    :return:
    """
    ray.init()

    itraj_start = 0
    n_traj = len(traj_name_list)

    n_worker = n_workers

    traj_per_worker = int(n_traj / np.float32(n_worker))
    start_idx = [itraj_start+traj_per_worker * i for i in range(n_worker)]
    end_idx = [itraj_start+traj_per_worker * (i + 1) - 1 for i in range(n_worker)]

    workers = []
    for i in range(n_worker):
        print 'worker {} going from {} to {} '.format(i, start_idx[i], end_idx[i])
        subset_traj = traj_name_list[start_idx[i]:end_idx[i]]
        workers.append(TF_rec_converter.remote(conf, gif_dir,subset_traj, start_idx[i], crop_from_highres))

    id_list = []
    for worker in workers:
        # time.sleep(2)
        id_list.append(worker.gather.remote())

    # blocking call
    res = [ray.get(id) for id in id_list]


def make_traj_name_list(conf, start_end_grp = None, shuffle=True):
    combined_list = []
    for source_dir in conf['source_basedirs']:

        traj_per_gr = Trajectory(conf).traj_per_group
        max_traj = get_maxtraj([source_dir + conf['sourcetags'][0]])

        if start_end_grp != None:
            startgrp = start_end_grp[0]
            startidx = startgrp*Trajectory(conf).traj_per_group
            endgrp = start_end_grp[1]
            if max_traj < (endgrp+1)*traj_per_gr -1:
                endidx = max_traj
            else:
                endidx = (endgrp+1)*traj_per_gr -1
        else:
            endidx = max_traj
            startgrp = 0
            startidx = 0
            endgrp = endidx / traj_per_gr

        trajname_ind_l = []  # list of tuples (trajname, ind) where ind is 0,1,2 in range(self.split_seq_by)
        for gr in range(startgrp, endgrp + 1):  # loop over groups
            gr_dir_main = source_dir + conf['sourcetags'][0]+'/traj_group' + str(gr)

            if gr == startgrp:
                trajstart = startidx
            else:
                trajstart = gr * traj_per_gr
            if gr == endgrp:
                trajend = endidx
            else:
                trajend = (gr + 1) * traj_per_gr - 1

            for i_tra in range(trajstart, trajend + 1):
                trajdir = gr_dir_main + "/traj{}".format(i_tra)
                if not os.path.exists(trajdir):
                    print 'file {} not found!'.format(trajdir)
                    continue
                trajname_ind_l.append(trajdir)

        print 'source_basedir: {}, length: {}'.format(source_dir,len(trajname_ind_l))
        assert len(trajname_ind_l) == len(set(trajname_ind_l))  #check for duplicates
        combined_list += trajname_ind_l

    if shuffle:
        random.shuffle(combined_list)

    return combined_list


def main():
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('hyper', type=str, help='configuration file name')
    parser.add_argument('--start_gr', type=int, default=None, help='start group')
    parser.add_argument('--end_gr', type=int, default=None, help='end group')
    parser.add_argument('--no_parallel', type=bool, default=False, help='do not use parallel processing')
    parser.add_argument('--n_workers', type=int, default=5, help='number of workers')
    parser.add_argument('--no_shuffle', type=bool, default=False, help='whether to shuffle trajectories')
    args = parser.parse_args()

    conf_file = args.hyper
    if not os.path.exists(args.hyper):
        sys.exit("configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    #make sure the directory is empty
    assert glob.glob(conf['tf_rec_dir'] + '/*') == []

    gif_file = '/'.join(str.split(conf['tf_rec_dir'], '/')[:-1]) + '/preview_gather'

    if args.start_gr != None:
        start_end_grp = [args.start_gr,args.end_gr]
    else:
        start_end_grp = None

    parallel = not args.no_parallel

    shuffle = not args.no_shuffle
    print 'shuffle: ', shuffle
    traj_name_list = make_traj_name_list(conf, start_end_grp = start_end_grp, shuffle=shuffle)

    if parallel:
        start_parallel(conf, gif_file, traj_name_list, args.n_workers,
                       crop_from_highres=True, start_end_grp=start_end_grp)
    else:
        tfrec_converter = TF_rec_converter(conf,
                                           gif_file,
                                           traj_name_list,
                                           0,
                                           crop_from_highres=True)
        tfrec_converter.gather()

    make_train_test_split(conf)

def make_train_test_split(conf = None):
    if conf is None:
        parser = argparse.ArgumentParser(description='Run benchmarks')
        parser.add_argument('hyper', type=str, help='configuration file name')
        parser.add_argument('--start_gr', type=int, default=None, help='start group')
        parser.add_argument('--end_gr', type=int, default=None, help='end group')
        parser.add_argument('--no_parallel', type=bool, default=False, help='do not use parallel processing')
        parser.add_argument('--n_workers', type=int, default=5, help='number of workers')
        parser.add_argument('--no_shuffle', type=bool, default=False, help='whether to shuffle trajectories')
        args = parser.parse_args()

        conf_file = args.hyper
        if not os.path.exists(args.hyper):
            sys.exit("configuration not found")
        hyperparams = imp.load_source('hyperparams', conf_file)

        conf = hyperparams.configuration

    traindir = conf["tf_rec_dir"]
    testdir = '/'.join(conf["tf_rec_dir"].split('/')[:-1] + ['/test'])
    import shutil
    files = glob.glob(traindir + '/*')

    files = sorted_alphanumeric(files)
    shutil.move(files[0], testdir)

import re

def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

if __name__ == "__main__":
    # make_train_test_split()
    main()