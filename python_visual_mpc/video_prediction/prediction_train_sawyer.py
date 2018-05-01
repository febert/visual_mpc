import os
import numpy as np
import tensorflow as tf
import imp
import sys
import pickle
import pdb

import imp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from .makegifs import comp_gif
import collections

from datetime import datetime
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 400

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

# How often to save a model checkpoint
SAVE_INTERVAL = 4000

from .utils_vpred.animate_tkinter import Visualizer_tkinter

from PIL import Image

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')
    flags.DEFINE_bool('diffmotions', False, 'visualize several different motions for a single scene')

sys.path.append("/docker_home/visual_mpc/python_visual_mpc")

## Helper functions
def peak_signal_to_noise_ratio(true, pred):
    """Image quality metric based on maximal signal power vs. power of the noise.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      peak signal to noise ratio (PSNR)
    """
    return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model(object):
    def __init__(self,
                 conf,
                 images=None,
                 actions=None,
                 states=None,
                 reuse_scope=None,
                 pix_distrib=None,
                 pix_distrib2=None,
                 build_loss = True):

        if 'prediction_model' in conf:
            Prediction_Model = conf['prediction_model']
        else:
            from .prediction_model_sawyer import Prediction_Model

        self.conf = conf

        if 'use_len' in conf:
            print('randomly shift videos for data augmentation')
            images, states, actions  = self.random_shift(images, states, actions)

        self.images_sel = images
        self.actions_sel = actions
        self.states_sel = states

        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        # Split into timesteps.
        if actions != None:
            actions = tf.split(axis=1, num_or_size_splits=actions.get_shape()[1], value=actions)
            actions = [tf.squeeze(act) for act in actions]
        if states != None:
            states = tf.split(axis=1, num_or_size_splits=states.get_shape()[1], value=states)
            states = [tf.squeeze(st) for st in states]
        images = tf.split(axis=1, num_or_size_splits=images.get_shape()[1], value=images)
        images = [tf.squeeze(img) for img in images]
        if pix_distrib != None:
            pix_distrib = tf.split(axis=1, num_or_size_splits=pix_distrib.get_shape()[1], value=pix_distrib)
            pix_distrib = [tf.squeeze(pix) for pix in pix_distrib]

        if pix_distrib2 != None:
            pix_distrib2 = tf.split(axis=1, num_or_size_splits=pix_distrib2.get_shape()[1], value=pix_distrib2)
            pix_distrib2= [tf.squeeze(pix) for pix in pix_distrib2]

        if reuse_scope is None:
            self.m = Prediction_Model(
                images,
                actions,
                states,
                iter_num=self.iter_num,
                pix_distributions1=pix_distrib,
                pix_distributions2=pix_distrib2,
                conf=conf)
            self.m.build()
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                self.m = Prediction_Model(
                    images,
                    actions,
                    states,
                    iter_num=self.iter_num,
                    pix_distributions1=pix_distrib,
                    pix_distributions2=pix_distrib2,
                    conf= conf)
                self.m.build()

        # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print i
        self.lr = tf.placeholder_with_default(conf['learning_rate'], (), 'learning_rate')

        self.gen_images = self.m.gen_images
        self.gen_distrib1 = self.m.gen_distrib1
        self.gen_distrib2 = self.m.gen_distrib2
        self.gen_states = self.m.gen_states
        self.gen_masks = self.m.gen_masks

        if build_loss:
            # L2 loss, PSNR for eval.
            true_fft_list, pred_fft_list = [], []
            loss, psnr_all = 0.0, 0.0

            self.fft_weights = tf.placeholder(tf.float32, [64, 64])

            for i, x, gx in zip(
                    list(range(len(self.m.gen_images))), images[conf['context_frames']:],
                    self.m.gen_images[conf['context_frames'] - 1:]):
                recon_cost_mse = mean_squared_error(x, gx)

                psnr_i = peak_signal_to_noise_ratio(x, gx)
                psnr_all += psnr_i
                summaries.append(tf.summary.scalar('recon_cost' + str(i), recon_cost_mse))
                summaries.append(tf.summary.scalar('psnr' + str(i), psnr_i))

                recon_cost = recon_cost_mse

                loss += recon_cost

            if ('ignore_state_action' not in conf) and ('ignore_state' not in conf):
                for i, state, gen_state in zip(
                        list(range(len(self.m.gen_states))), states[conf['context_frames']:],
                        self.m.gen_states[conf['context_frames'] - 1:]):
                    state_cost = mean_squared_error(state, gen_state) * 1e-4 * conf['use_state']
                    summaries.append(
                        tf.summary.scalar('state_cost' + str(i), state_cost))
                    loss += state_cost

            summaries.append(tf.summary.scalar('psnr_all', psnr_all))
            self.psnr_all = psnr_all

            self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])

            summaries.append(tf.summary.scalar('loss', loss))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            self.summ_op = tf.summary.merge(summaries)


    def random_shift(self, images, states, actions):

        print('shifting the video sequence randomly in time')
        tshift = 2
        uselen = self.conf['use_len']
        fulllength = self.conf['sequence_length']
        nshifts = (fulllength - uselen) / 2 + 1
        rand_ind = tf.random_uniform([1], 0, nshifts, dtype=tf.int64)
        self.rand_ind = rand_ind

        start = tf.concat(axis=0,values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(3, dtype=tf.int64)])
        images_sel = tf.slice(images, start, [-1, uselen, -1, -1, -1])
        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(1, dtype=tf.int64)])
        actions_sel = tf.slice(actions, start, [-1, uselen, -1])
        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(1, dtype=tf.int64)])
        states_sel = tf.slice(states, start, [-1, uselen, -1])

        return images_sel, states_sel, actions_sel


class Getdesig(object):
    def __init__(self,img,conf,img_namesuffix):
        self.suf = img_namesuffix
        self.conf = conf
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        self.coords = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        self.coords = np.array([event.ydata, event.xdata])
        self.ax.scatter(self.coords[1], self.coords[0], marker= "o", s=70, facecolors='b', edgecolors='b')
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.draw()
        plt.savefig(self.conf['output_dir']+'/img_desigpix'+self.suf)

def main(unused_argv, conf_script= None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print('using CUDA_VISIBLE_DEVICES=', FLAGS.device)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    if conf_script == None: conf_file = FLAGS.hyper
    else: conf_file = conf_script

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    inference = False
    if FLAGS.visualize:
        print('creating visualizations ...')
        conf['schedsamp_k'] = -1  # don't feed ground truth

        if 'test_data_dir' in conf:
            conf['data_dir'] = conf['test_data_dir']
        else: conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])

        conf['visualize'] = conf['output_dir'] + '/' + FLAGS.visualize
        conf['event_log_dir'] = '/tmp'
        conf.pop('use_len', None)
        conf['batch_size'] = 30

        conf['sequence_length'] = 14
        if FLAGS.diffmotions:
            inference = True
            conf['sequence_length'] = 15

    if 'sawyer' in conf:
        if conf['adim'] == 5:
            from read_tf_record_wristrot import build_tfrecord_input
        else:
            from read_tf_record_sawyer12 import build_tfrecord_input
    else:
        from read_tf_record import build_tfrecord_input

    print('Constructing models and inputs.')
    if FLAGS.diffmotions:
        if 'adim' in conf:
            adim = conf['adim']
        else: adim = 4

        if 'sdim' in conf:
            statedim = conf['sdim']
        else: statedim = 3

        actions_pl = tf.placeholder(tf.float32, name='actions',
                                    shape=(conf['batch_size'], conf['sequence_length'], adim))
        states_pl = tf.placeholder(tf.float32, name='states',
                                   shape=(conf['batch_size'], conf['sequence_length'], statedim))

        images_pl = tf.placeholder(tf.float32, name='images',
                                   shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
        val_images, _, val_states = build_tfrecord_input(conf, training=False)

        pix_distrib_pl = tf.placeholder(tf.float32, name='states',
                                        shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 1))

        with tf.variable_scope('model', reuse=None):
            val_model = Model(conf, images_pl, actions_pl, states_pl, pix_distrib=pix_distrib_pl,
                              inference=inference)
    else:
        with tf.variable_scope('model', reuse=None) as training_scope:
            images_aux1, actions, states = build_tfrecord_input(conf, training=True)
            images = images_aux1
            model = Model(conf, images, actions, states, inference=inference)

        with tf.variable_scope('val_model', reuse=None):
            val_images_aux1, val_actions, val_states = build_tfrecord_input(conf, training=False)
            val_images = val_images_aux1
            val_model = Model(conf, val_images, val_actions, val_states,
                              training_scope, inference=inference)

    print('Constructing saver.')
    # Make saver.

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # remove all states from group of variables which shall be saved and restored:
    vars_no_state = filter_vars(vars)
    saver = tf.train.Saver(vars_no_state, max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    if conf['visualize']:
        print('-------------------------------------------------------------------')
        print('verify current settings!! ')
        for key in list(conf.keys()):
            print(key, ': ', conf[key])
        print('-------------------------------------------------------------------')

        import re
        itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)

        saver.restore(sess, conf['visualize'])
        print('restore done.')

        feed_dict = {
                     val_model.iter_num: 0 }

        file_path = conf['output_dir']

        if FLAGS.diffmotions:

            b_exp, ind0 =28, 0

            img, state = sess.run([val_images, val_states])
            sel_img= img[b_exp,ind0:ind0+2]

            # fig1 = plt.figure(1)
            # plt.imshow(sel_img[0])
            # fig2 = plt.figure(2)
            # plt.imshow(sel_img[1])
            # plt.show()
            # pdb.set_trace()

            c = Getdesig(sel_img[0], conf, 'b{}'.format(b_exp))
            desig_pos_aux1 = c.coords.astype(np.int32)
            # desig_pos_aux1 = np.array([29, 37])
            print("selected designated position for aux1 [row,col]:", desig_pos_aux1)

            one_hot = create_one_hot(conf, desig_pos_aux1)

            feed_dict[pix_distrib_pl] = one_hot

            sel_state = np.stack([state[b_exp,ind0],state[b_exp,ind0+1]], axis=0)

            start_states = np.concatenate([sel_state,np.zeros((conf['sequence_length']-2, statedim))])
            start_states = np.expand_dims(start_states, axis=0)
            start_states = np.repeat(start_states, conf['batch_size'], axis=0)  # copy over batch
            feed_dict[states_pl] = start_states

            start_images = np.concatenate([sel_img,np.zeros((conf['sequence_length']-2, 64, 64, 3))])

            start_images = np.expand_dims(start_images, axis=0)
            start_images = np.repeat(start_images, conf['batch_size'], axis=0)  # copy over batch
            feed_dict[images_pl] = start_images

            actions = np.zeros([conf['batch_size'], conf['sequence_length'], adim])

            # step = .025
            step = .055
            n_angles = 8
            col_titles = []
            for b in range(n_angles):
                col_titles.append('move')
                for i in range(conf['sequence_length']):
                    actions[b,i][:2] = np.array([np.cos(b/float(n_angles)*2*np.pi)*step, np.sin(b/float(n_angles)*2*np.pi)*step])

            if adim == 5:
                b += 1
                actions[b, 0] = np.array([0, 0, 4, 0, 0])
                actions[b, 1] = np.array([0, 0, 4, 0, 0])
                col_titles.append('up/down')

                b += 1
                actions[b, 0] = np.array([0, 0, 0, 0, 4])
                actions[b, 1] = np.array([0, 0, 0, 0, 4])
                col_titles.append('close/open')

                delta_rot = 0.4
                b += 1
                for i in range(conf['sequence_length']):
                    actions[b, i] = np.array([0, 0, 0, delta_rot, 0])
                col_titles.append('rot +')

                b += 1
                for i in range(conf['sequence_length']):
                    actions[b, i] = np.array([0, 0, 0, -delta_rot, 0])
                col_titles.append('rot -')

                col_titles.append('noaction')

            elif adim == 4:
                b+=1
                actions[b, 0] = np.array([0, 0, 4, 0])
                actions[b, 1] = np.array([0, 0, 4, 0])


                b += 1
                actions[b, 0] = np.array([0, 0, 0, 4])
                actions[b, 1] = np.array([0, 0, 0, 4])

            feed_dict[actions_pl] = actions

            gen_images, gen_distrib, gen_masks, moved_parts, moved_images, moved_bckgd = sess.run([val_model.m.gen_images,
                                                            val_model.m.gen_distrib1,
                                                            val_model.m.gen_masks,
                                                            val_model.m.movd_parts_list,
                                                            val_model.m.moved_images,
                                                            val_model.m.moved_bckgd
                                                            ]
                                                           ,feed_dict)

            dict = collections.OrderedDict()
            dict['gen_images'] = gen_images
            dict['gen_masks'] = gen_masks
            dict['gen_distrib'] = gen_distrib
            dict['iternum'] = itr_vis

            dict['desig_pos'] = desig_pos_aux1
            # dict['moved_parts'] = moved_parts
            # dict['moved_images'] = moved_images
            # dict['moved_bckgd'] = moved_bckgd

            pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
            print('written files to:' + file_path)

            v = Visualizer_tkinter(dict, numex=b+1, append_masks=False,
                                   filepath=conf['output_dir'],
                                   suf='_diffmotions_b{}_l{}'.format(b_exp, conf['sequence_length']), col_titles=col_titles)
            v.build_figure()

        else:
            ground_truth, gen_images, gen_masks = sess.run([val_images,
                                                            val_model.m.gen_images,
                                                            val_model.m.gen_masks
                                                            ],
                                                            feed_dict)
            dict = collections.OrderedDict()
            dict['ground_truth'] = ground_truth
            dict['gen_images'] = gen_images
            dict['gen_masks'] = gen_masks
            dict['iternum'] = itr_vis

            pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
            print('written files to:' + file_path)

            v = Visualizer_tkinter(dict, numex=conf['batch_size'], append_masks=False, filepath=conf['output_dir'],
                                   col_titles=[str(i) for i in range(conf['batch_size'])])
            v.build_figure()
        return

    itr_0 =0
    if FLAGS.pretrained != None:
        conf['pretrained_model'] = FLAGS.pretrained

        saver.restore(sess, conf['pretrained_model'])
        # resume training at iteration step of the loaded model:
        import re
        itr_0 = re.match('.*?([0-9]+)$', conf['pretrained_model']).group(1)
        itr_0 = int(itr_0)
        print('resuming training at iteration:  ', itr_0)

    print('-------------------------------------------------------------------')
    print('verify current settings!! ')
    for key in list(conf.keys()):
        print(key, ': ', conf[key])
    print('-------------------------------------------------------------------')

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter = []
    # Run training.

    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()
        # Generate new batch of data_files.
        feed_dict = {model.iter_num: np.float32(itr)}

        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {val_model.iter_num: np.float32(itr)}
            [val_summary_str] = sess.run([val_model.summ_op], feed_dict)
            summary_writer.add_summary(val_summary_str, itr)


        if itr % SAVE_INTERVAL == 0: #and itr != 0:
            tf.logging.info('Saving model to' + conf['output_dir'])
            saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        t_iter.append((datetime.now() - t_startiter).seconds * 1e6 +  (datetime.now() - t_startiter).microseconds )

        if itr % 100 == 1:
            hours = (datetime.now() -starttime).seconds/3600
            tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                (datetime.now() - starttime).days,
                hours,
                (datetime.now() - starttime).seconds/60 - hours*60))
            avg_t_iter = np.sum(np.asarray(t_iter))/len(t_iter)
            tf.logging.info('time per iteration: {0}'.format(avg_t_iter/1e6))
            tf.logging.info('expected for complete training: {0}h '.format(avg_t_iter /1e6/3600 * conf['num_iterations']))

        if (itr) % SUMMARY_INTERVAL:
            summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, conf['output_dir'] + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush()


def create_one_hot(conf, desig_pix):
    one_hot = np.zeros((1, 1, 64, 64, 1), dtype=np.float32)
    # switch on pixels
    one_hot[0, 0, desig_pix[0], desig_pix[1]] = 1.
    one_hot = np.repeat(one_hot, conf['context_frames'], axis=1)
    app_zeros = np.zeros((1, conf['sequence_length']- conf['context_frames'], 64, 64, 1), dtype=np.float32)
    one_hot = np.concatenate([one_hot, app_zeros], axis=1)
    one_hot = np.repeat(one_hot, conf['batch_size'], axis=0)

    return one_hot


def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print('removed state variable from saving-list: ', v.name)
    return newlist

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
