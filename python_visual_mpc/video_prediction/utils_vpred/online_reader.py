import numpy as np
import tensorflow as tf
import random
import os
import cPickle
from collections import namedtuple
from python_visual_mpc.data_preparation.gather_data import make_traj_name_list, get_maxtraj, Trajectory, crop_and_rot
import ray
import re
import sys
import glob
import pdb
import itertools

import imp


@ray.remote
class Reading_thread(object):
    def __init__(self, conf, subset_traj, enqueue_op, sess, images_pl, states_pl, actions_pl):
        num_errors = 0
        self.conf = conf
        self.data_conf = conf['data_configuration']

        self.src_names = [str.split(n, '/')[-1] for n in self.sourcedirs]

        for trajname in itertools.cycle(subset_traj):  # loop of traj0, traj1,..

            # print 'processing {}, seq-part {}'.format(trajname, traj_tuple[1] )
            try:
                traj_index = re.match('.*?([0-9]+)$', trajname).group(1)
                self.traj = Trajectory(self.conf)

                traj_subpath = '/'.join(str.split(trajname, '/')[-2:])   #string with only the group and trajectory

                pkl_file = trajname + '/joint_angles_traj{}.pkl'.format(traj_index)
                if not os.path.isfile(pkl_file):
                    print 'no pkl file found in', trajname
                    continue

                pkldata = cPickle.load(open(pkl_file, "rb"))
                self.all_actions = pkldata['actions']
                self.all_joint_angles = pkldata['jointangles']
                self.all_endeffector_pos = pkldata['endeffector_pos']

                for i_src, tag in enumerate(conf['source_tags']):  # loop over cameras: main, aux1, ..
                    self.traj_dir_src = os.path.join(conf['source_basedir'], tag, traj_subpath)
                    self.step_from_to(i_src)

                sess.run(enqueue_op, feed_dict={images_pl: self.traj.images,
                                                states_pl: self.traj.endeffector_pos,
                                                actions_pl: self.traj.actions})
            except KeyboardInterrupt:
                sys.exit()
            except:
                print "error occured"
                num_errors += 1


    def step_from_to(self, i_src):
        trajind = 0  # trajind is the index in the target trajectory
        end = Trajectory(self.data_conf).npictures

        smp_range = end//self.traj.tspacing - self.conf['sequence_length']
        start = np.random.random_integers(0, smp_range)*self.traj.tspacing
        end = start + self.conf['sequence_length']*self.traj.tspacing
        pdb.set_trace()
        for dataind in range(start, end, self.traj.tspacing):  # dataind is the index in the source trajetory
            # get low dimensional data
            self.traj.actions[trajind] = self.all_actions[dataind]
            self.traj.joint_angles[trajind] = self.all_joint_angles[dataind]
            self.traj.endeffector_pos[trajind] = self.all_endeffector_pos[dataind]

            if 'imagename_no_zfill' in self.conf:
                im_filename = self.traj_dir_src + '/images/{0}_full_cropped_im{1}_*' \
                    .format(self.src_names[i_src], dataind)
            else:
                im_filename = self.traj_dir_src + '/images/{0}_full_cropped_im{1}_*' \
                    .format(self.src_names[i_src], str(dataind).zfill(2))
            if dataind == 0:
                print 'processed from file {}'.format(im_filename)
            if dataind == end - self.traj.tspacing:
                print 'processed to file {}'.format(im_filename)

            file = glob.glob(im_filename)
            if len(file) > 1:
                raise ValueError
            file = file[0]

            im = crop_and_rot(file, i_src)

            self.traj.images[trajind, i_src] = im
            trajind += 1


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
        self.states_pl = tf.placeholder(tf.float32, name='states', shape=(conf['context_frames'], sdim))

        if mode == 'training' or mode == 'validation':
            self.num_threads = 1
        else: self.num_threads = 1

        if mode == 'test':
            self.shuffle = False
        else: self.shuffle = True

        self.q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32], shapes=[self.images_pl.get_shape().as_list(),
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
            source_name = '/'.join(str.split(dir, '/')[-1])

            print 'preparing source_basedir', dir
            split_file = self.data_conf['current_dir'] + '/' + source_name + '_split.pkl'

            dataset = namedtuple('dataset', ['train', 'val', 'test'])
            if os.path.isfile(split_file):
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

                dataset_i = dataset(train_traj, val_traj, test_traj)

            datasets.append(dataset_i)

        return datasets


    def combine_traj_lists(self, datasets):
        combined = []
        for dset in datasets:
            dset = getattr(dset, self.mode)
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
        ray.init()

        itraj_start = 0
        n_traj = len(traj_list)

        traj_per_worker = int(n_traj / np.float32(self.num_threads))
        start_idx = [itraj_start + traj_per_worker * i for i in range(self.num_threads)]
        end_idx = [itraj_start + traj_per_worker * (i + 1) - 1 for i in range(self.num_threads)]

        workers = []
        for i in range(self.num_threads):
            print 'worker {} going from {} to {} '.format(i, start_idx[i], end_idx[i])
            subset_traj = traj_list[start_idx[i]:end_idx[i]]
            workers.append(Reading_thread.remote(self.conf, subset_traj, self.enqueue_op, self.sess,
                                                 self.images_pl, self.states_pl, self.actions_pl))
        id_list = []
        # for worker in workers:
        #     # time.sleep(2)
        #     id_list.append(worker.gather.remote())
        #
        # # blocking call
        # res = [ray.get(id) for id in id_list]


def test_online_reader():

    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))

    conf['train_val_split'] = 0.95
    conf['sequence_length'] = 15  # 'sequence length, including context frames.'
    conf['batch_size'] = 32
    conf['context_frames'] = 2

    conf['im_height'] = 64
    conf['sdim'] = 3
    conf['adim'] = 4

    conf['current_dir'] = current_dir

    dataconf_file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/pushing_data/online_weiss/dataconf.py'
    data = imp.load_source('hyperparams', dataconf_file)
    conf['data_configuration'] = data.data_configuration

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
        # images, actions, endeff, robot_pos, object_pos = sess.run([image_batch, action_batch, endeff_pos_batch, robot_pos_batch, object_pos_batch])

        file_path = '/'.join(str.split(DATA_DIR, '/')[:-1] + ['preview'])
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