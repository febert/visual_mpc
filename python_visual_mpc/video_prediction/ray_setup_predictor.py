# starting the client

import tensorflow as tf
import imp
import numpy as np
import pdb

from PIL import Image
import os

from datetime import datetime
import ray

@ray.remote(num_gpus=1)
class LocalServer(object):
    def __init__(self, netconf, policyparams, local_batch_size, use_ray= True):
        print 'making LocalServer'

        self.policyparams = policyparams
        self.local_batch_size = local_batch_size
        self.netconf = netconf
        if 'prediction_model' in netconf:
            Model = netconf['prediction_model']
        else:
            from video_prediction.sawyer.prediction_train_sawyer import Model

        if use_ray:
            print 'using CUDA_VISIBLE_DEVICES=', ray.get_gpu_ids()
            print 'ray_getgpou', ray.get_gpu_ids()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])
        else:
            print 'using CUDA_VISIBLE_DEVICES=', 0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

        from tensorflow.python.client import device_lib
        print device_lib.list_local_devices()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        # Make training session.
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,
                                                           allow_soft_placement=True,
                                                           log_device_placement=False))

        print '-------------------------------------------------------------------'
        print 'verify current settings!! '
        for key in netconf.keys():
            print key, ': ', netconf[key]
        print '-------------------------------------------------------------------'

        print 'Constructing multi gpu model for control...'

        if 'single_view' in netconf:
            numcam = 1
        else:
            numcam = 2
        self.start_images_pl = tf.placeholder(tf.float32, name='images',  # with zeros extension
                                        shape=(local_batch_size, netconf['sequence_length'], 64, 64, 3*numcam))
        self.actions_pl = tf.placeholder(tf.float32, name='actions',
                                        shape=(local_batch_size,netconf['sequence_length'], 4))
        self.start_states_pl = tf.placeholder(tf.float32, name='states',
                                              shape=(local_batch_size,netconf['context_frames'], 3))
        self.pix_distrib_1_pl = tf.placeholder(tf.float32, shape=(local_batch_size, netconf['context_frames'], 64, 64, numcam))
        self.pix_distrib_2_pl = tf.placeholder(tf.float32, shape=(local_batch_size, netconf['context_frames'], 64, 64, numcam))

        with tf.variable_scope('model', reuse=None):
            self.model = Model(netconf, self.start_images_pl, self.actions_pl, self.start_states_pl,
                               pix_distrib=self.pix_distrib_1_pl)

        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
        saver.restore(self.sess, netconf['pretrained_model'])
        print 'restore done. '


        # self.sess.run(tf.global_variables_initializer())
        # restore_vars = tf.get_default_graph().get_collection(name=tf.GraphKeys.VARIABLES, scope='model')
        # for var in restore_vars:
        #     print var.name, var.get_shape()
        # saver = tf.train.Saver(restore_vars, max_to_keep=0)
        # saver.restore(self.sess, netconf['pretrained_model'])


    def predict(self, last_frames=None, input_distrib=None, last_states=None, input_actions=None, goal_pix=None):

        input_distrib = np.expand_dims(input_distrib, axis=0)
        input_distrib = np.repeat(input_distrib, self.local_batch_size, 0)

        t_startiter = datetime.now()

        last_states = np.expand_dims(last_states, axis=0)
        input_state = np.repeat(last_states, self.local_batch_size, axis=0)
        img_channels = 3
        last_frames = np.expand_dims(last_frames, axis=0)
        last_frames = np.repeat(last_frames, self.local_batch_size, axis=0)

        app_zeros = np.zeros(shape=(self.local_batch_size, self.netconf['sequence_length'] -
                                    self.netconf['context_frames'], 64, 64, img_channels))
        last_frames = np.concatenate((last_frames, app_zeros), axis=1)
        input_images = last_frames.astype(np.float32) / 255.

        feed_dict = {}
        feed_dict[self.start_images_pl] = input_images
        feed_dict[self.start_states_pl] = input_state
        feed_dict[self.actions_pl] = input_actions

        feed_dict[self.pix_distrib_1_pl] = input_distrib

        distance_grid = self.get_distancegrid(goal_pix)
        gen_images, gen_distrib, gen_states = self.sess.run([self.model.gen_images,
                                                         self.model.gen_distrib1,
                                                         self.model.gen_states,
                                                        ],
                                                       feed_dict)

        scores = self.calc_scores(gen_distrib, distance_grid)

        print 'time for evaluating {0} actions: {1}'.format(
            self.local_batch_size,
            (datetime.now() - t_startiter).seconds + (datetime.now() - t_startiter).microseconds / 1e6)

        bestind = scores.argsort()[0]
        best_gen_distrib = gen_distrib[2][bestind].reshape(1, 64, 64, 1)

        return (best_gen_distrib, scores[bestind]), scores

    def calc_scores(self, gen_distrib, distance_grid):
        expected_distance = np.zeros(self.local_batch_size)
        if 'rew_all_steps' in self.policyparams:
            for tstep in range(self.netconf['sequence_length'] - 1):
                t_mult = 1
                if 'finalweight' in self.policyparams:
                    if tstep == self.netconf['sequence_length'] - 2:
                        t_mult = self.policyparams['finalweight']

                for b in range(self.local_batch_size):
                    gen = gen_distrib[tstep][b].squeeze() / np.sum(gen_distrib[tstep][b])
                    expected_distance[b] += np.sum(np.multiply(gen, distance_grid)) * t_mult
            scores = expected_distance
        else:
            for b in range(self.local_batch_size):
                gen = gen_distrib[-1][b].squeeze() / np.sum(gen_distrib[-1][b])
                expected_distance[b] = np.sum(np.multiply(gen, distance_grid))
            scores = expected_distance
        return scores

    def get_distancegrid(self, goal_pix):
        distance_grid = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                pos = np.array([i, j])
                distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

        print 'making distance grid with goal_pix', goal_pix
        return distance_grid


def setup_predictor(netconf, policyparams, ngpu, redis_address=''):
    if redis_address == '':
        ray.init(num_gpus=ngpu)
    else:
        ray.init(redis_address=redis_address)

    local_bsize = np.floor(netconf['batch_size']/ngpu).astype(np.int32)
    new_batch_size = local_bsize * ngpu
    workers = []

    startind = []
    endind = []

    start_counter = 0
    for i in range(ngpu):
        workers.append(LocalServer.remote(netconf, policyparams, local_bsize))

        startind.append(start_counter)
        endind.append(start_counter + local_bsize)
        print 'indices for gpu {0}: {1} to {2}'.format(i, startind[-1], endind[-1])
        start_counter += local_bsize

    def predictor_func(input_images=None,
                       input_one_hot_images1=None,
                       input_states=None,
                       input_actions=None,
                       goal_pix = None
                        ):

        result_list = []
        for i in range(ngpu):
            result = workers[i].predict.remote(
                                       input_images,
                                       input_one_hot_images1,
                                       input_states,
                                       input_actions[startind[i]:endind[i]],
                                       goal_pix
                                       )

            result_list.append(result)

        result_list = ray.get(result_list)

        scores_list = []
        best_gen_distrib_list = []

        for i in range(ngpu):
            best_gen_distrib, scores  = result_list[i]
            best_gen_distrib_list.append(best_gen_distrib)
            scores_list.append(scores)

        scores = np.concatenate(scores_list)

        best_gpuid = np.array([t[1] for t in best_gen_distrib_list]).argmin()
        single_best_gen_distrib = best_gen_distrib_list[best_gpuid][0].reshape((1,64,64,1))

        return single_best_gen_distrib, scores

    return predictor_func

if __name__ == '__main__':
    conffile = '/home/frederik/Documents/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop_1stimg_bckgd/conf.py'
    netconf = imp.load_source('mod_hyper', conffile).configuration
    predfunc = setup_predictor(netconf,None, 1)
    pdb.set_trace()