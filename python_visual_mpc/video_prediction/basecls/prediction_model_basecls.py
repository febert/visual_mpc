import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.lstm_ops12 import basic_conv_lstm_cell
from python_visual_mpc.misc.zip_equal import zip_equal

from utils.transformations import dna_transformation, cdna_transformation
from utils.compute_motion_vecs import compute_motion_vector_dna, compute_motion_vector_cdna
from python_visual_mpc.video_prediction.basecls.utils.visualize import visualize_diffmotions, visualize, compute_metric
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
import cPickle
import collections
import pdb
from utils.visualize import visualize_diffmotions, visualize

class Base_Prediction_Model(object):

    def __init__(self,
                conf = None,
                trafo_pix = True,
                load_data = True,
                ):
        """
        :param conf:
        :param trafo_pix: whether to transform  distributions of designated pixels
        :param load_data:  whether to load data
        :param mode:  whether to build train- or val-model
        """
        self.trafo_pix = trafo_pix
        self.conf = conf
        self.cdna, self.stp, self.dna = False, False, False
        if self.conf['model'] == 'CDNA':
            self.cdna = True
        elif self.conf['model'] == 'DNA':
            self.dna = True
        elif self.conf['model'] == 'STP':
            self.stp = True
        if self.stp + self.cdna + self.dna != 1:
            raise ValueError("More than one option selected!")

        self.k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_masks = conf['num_masks']
        self.context_frames = conf['context_frames']
        self.batch_size = conf['batch_size']

        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self.sdim = conf['sdim']
        self.adim = conf['adim']

        if not load_data:
            self.actions_pl = tf.placeholder(tf.float32, name='actions',
                                        shape=(conf['batch_size'], conf['sequence_length'], self.adim))
            actions = self.actions_pl

            self.states_pl = tf.placeholder(tf.float32, name='states',
                                       shape=(conf['batch_size'], conf['sequence_length'], self.sdim))
            states = self.states_pl

            self.images_pl = tf.placeholder(tf.float32, name='images',
                                       shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
            images = self.images_pl

            self.pix_distrib_pl = tf.placeholder(tf.float32, name='states',
                                            shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 1))
            pix_distrib1  = self.pix_distrib_pl

        else:
            if 'adim' in conf:
                from python_visual_mpc.video_prediction.read_tf_record_wristrot import \
                    build_tfrecord_input as build_tfrecord_fn
            else:
                from python_visual_mpc.video_prediction.read_tf_record_sawyer12 import \
                    build_tfrecord_input as build_tfrecord_fn
            train_images, train_actions, train_states = build_tfrecord_fn(conf, training=True)
            val_images, val_actions, val_states = build_tfrecord_fn(conf, training=False)

            images, actions, states = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                             lambda: [train_images, train_actions, train_states],
                                             lambda: [val_images, val_actions, val_states])

        self.color_channels = 3

        self.lstm_func = basic_conv_lstm_cell

        # Generated robot states and images.
        self.gen_states = []
        self.gen_images = []
        self.gen_masks = []

        self.moved_images = []
        self.moved_bckgd = []

        self.moved_pix_distrib1 = []
        self.moved_pix_distrib2 = []

        self.states = states
        self.gen_distrib1 = []
        self.gen_distrib2 = []

        self.trafos = []
        self.movd_parts_list = []
        self.pred_kerns = []

        self.prediction_flow = []

        if 'use_len' in conf:
            print 'randomly shift videos for data augmentation'
            images, states, actions  = self.random_shift(images, states, actions)

        self.iter_num = tf.placeholder(tf.float32, [])

        # Split into timesteps.
        actions = tf.split(axis=1, num_or_size_splits=actions.get_shape()[1], value=actions)
        actions = [tf.squeeze(act) for act in actions]
        states = tf.split(axis=1, num_or_size_splits=states.get_shape()[1], value=states)
        states = [tf.squeeze(st) for st in states]
        images = tf.split(axis=1, num_or_size_splits=images.get_shape()[1], value=images)
        images = [tf.squeeze(img) for img in images]

        if trafo_pix:
            pix_distrib1 = tf.split(axis=1, num_or_size_splits=pix_distrib1.get_shape()[1], value=pix_distrib1)
            self.pix_distrib1= [tf.squeeze(pix) for pix in pix_distrib1]

        self.actions = actions
        self.images = images
        self.states = states

        self.build()

    def random_shift(self, images, states, actions):
        print 'shifting the video sequence randomly in time'
        tshift = 2
        uselen = self.conf['use_len']
        fulllength = self.conf['sequence_length']
        nshifts = (fulllength - uselen) / 2 + 1
        rand_ind = tf.random_uniform([1], 0, nshifts, dtype=tf.int64)
        self.rand_ind = rand_ind

        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(3, dtype=tf.int64)])
        images_sel = tf.slice(images, start, [-1, uselen, -1, -1, -1])
        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(1, dtype=tf.int64)])
        actions_sel = tf.slice(actions, start, [-1, uselen, -1])
        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(1, dtype=tf.int64)])
        states_sel = tf.slice(states, start, [-1, uselen, -1])

        return images_sel, states_sel, actions_sel


    def build(self):
        """
        Build the network
        :return:
        """

        if 'kern_size' in self.conf.keys():
            KERN_SIZE = self.conf['kern_size']
        else: KERN_SIZE = 5

        batch_size, img_height, img_width, self.color_channels = self.images[0].get_shape()[0:4]

        if self.states != None:
            current_state = self.states[0]
        else:
            current_state = None


        if self.k == -1:
            feedself = True
        else:
            # Scheduled sampling:
            # Calculate number of ground-truth frames to pass in.
            self.num_ground_truth = tf.to_int32(
                tf.round(tf.to_float(batch_size) * (self.k / (self.k + tf.exp(self.iter_num / self.k)))))
            feedself = False

        # LSTM state sizes and states.

        if 'lstm_size' in self.conf:
            self.lstm_size = self.conf['lstm_size']
            print 'using lstm size', self.lstm_size
        else:
            self.lstm_size = np.int32(np.array([16, 32, 64, 32, 16]))


        self.lstm_state1, self.lstm_state2, self.lstm_state3, self.lstm_state4 = None, None, None, None
        self.lstm_state5, self.lstm_state6, self.lstm_state7 = None, None, None

        t = -1
        for image, action in zip(self.images[:-1], self.actions[:-1]):
            t +=1
            print t
            # Reuse variables after the first timestep.

            self.reuse = bool(self.gen_images)
            self.prev_image, self.prev_pix_distrib1, self.prev_pix_distrib2 = self.get_input_image(
                                                                        feedself,
                                                                        image,
                                                                        t)

            current_state = self.build_network_core(action, current_state, self.prev_image)
        self.build_loss()

    def apply_trafo_predict(self, enc6, hidden5):
        """
        Apply the transformatios (DNA, CDNA) and combine them to from the output-image
        :param KERN_SIZE:
        :param current_state:
        :param enc6:
        :param hidden5:
        :param prev_image:
        :param prev_pix_distrib1:
        :param prev_pix_distrib2:
        :param state_action:
        :return:
        """
        if self.dna:
            dna_kernel, tf_distrib_ndesig1, tf_distrib_ndesig2, tf_l = self.apply_dna(enc6,
                          self.prev_image, self.prev_pix_distrib1, self.prev_pix_distrib2)

        if self.cdna:
            cdna_kerns, tf_distrib_ndesig1, tf_distrib_ndesig2, tf_l = self.apply_cdna(
                enc6, hidden5, self.prev_image, self.prev_pix_distrib1, self.prev_pix_distrib2)

        if '1stimg_bckgd' in self.conf:
            background = self.images[0]
            print 'using background from first image..'
        else:
            background = self.prev_image

        output, mask_list = self.fuse_trafos(enc6, background, tf_l, scope='convt7_cam2')
        self.gen_images.append(output)
        self.gen_masks.append(mask_list)

        if self.trafo_pix:
            pix_distrib_output = self.fuse_pix_distrib(mask_list,
                                                       self.pix_distrib1,
                                                       self.prev_pix_distrib1,
                                                       tf_distrib_ndesig1)
            self.gen_distrib1.append(pix_distrib_output)

            if 'ndesig' in self.conf:
                pix_distrib_output = self.fuse_pix_distrib(mask_list,
                                                           self.pix_distrib2,
                                                           self.prev_pix_distrib2,
                                                           tf_distrib_ndesig2)
                self.gen_distrib2.append(pix_distrib_output)

        if 'visual_flowvec' in self.conf:
            if self.cdna:
                motion_vecs = compute_motion_vector_cdna(self.conf, cdna_kerns)

            if self.dna:
                motion_vecs = compute_motion_vector_dna(self.conf, dna_kernel)

            output = tf.zeros([self.conf['batch_size'], 64, 64, 2])
            for vec, mask in zip(motion_vecs, mask_list[1:]):
                if self.conf['model'] == 'CDNA':
                    vec = tf.reshape(vec, [self.conf['batch_size'], 1, 1, 2])
                    vec = tf.tile(vec, [1, 64, 64, 1])
                output += vec * mask
            flow_vectors = output

            self.prediction_flow.append(flow_vectors)


    def apply_cdna(self, enc6, hidden5, prev_image, prev_pix_distrib1, prev_pix_distrib2):
        if 'gen_pix' in self.conf:
            enc7 = slim.layers.conv2d_transpose(
                enc6, self.color_channels, 1, stride=1, scope='convt4')
            transformed_l = [tf.nn.sigmoid(enc7)]
            self.extra_masks = 2
        else:
            transformed_l = []
            self.extra_masks = 1
        if 'mov_bckgd' in self.conf:
            self.extra_masks = self.num_masks
        cdna_input = tf.reshape(hidden5, [int(self.batch_size), -1])
        new_transformed, cdna_kerns = cdna_transformation(self.conf, prev_image,
                                                               cdna_input,
                                                               reuse_sc=self.reuse)

        self.pred_kerns.append(cdna_kerns)

        transformed_l += new_transformed
        self.moved_images.append(transformed_l)
        if self.trafo_pix:
            tf_distrib_ndesig1, _ = cdna_transformation(self.conf, prev_pix_distrib1,
                                                                 cdna_input,
                                                                 reuse_sc=True)
            self.moved_pix_distrib1.append(tf_distrib_ndesig1)

            if 'ndesig' in self.conf:
                tf_distrib_ndesig2, _ = cdna_transformation(
                                        self.conf,
                                        prev_pix_distrib2,
                                        cdna_input,
                                        reuse_sc=True)

                self.moved_pix_distrib2.append(tf_distrib_ndesig2)
            else:
                tf_distrib_ndesig2 = None
        else:
            tf_distrib_ndesig1 = None
            tf_distrib_ndesig2 = None

        return cdna_kerns, tf_distrib_ndesig1, tf_distrib_ndesig2, transformed_l

    def apply_dna(self, enc6, prev_image, prev_pix_distrib1, prev_pix_distrib2):
        # Using largest hidden state for predicting untied conv kernels.
        KERN_SIZE = self.conf['kern_size']

        trafo_input = slim.layers.conv2d_transpose(
            enc6, KERN_SIZE ** 2, 1, stride=1, scope='convt4_cam2')
        transformed_l, dna_kernel  = dna_transformation(self.conf, prev_image, trafo_input)

        self.pred_kerns.append(dna_kernel)

        if self.trafo_pix:
            transf_distrib_ndesig1, _ = dna_transformation(self.conf, prev_pix_distrib1, trafo_input)
            if 'ndesig' in self.conf:
                transf_distrib_ndesig2, _ = dna_transformation(self.conf, prev_pix_distrib2, trafo_input)
            else:
                transf_distrib_ndesig2 = None
        else:
            transf_distrib_ndesig1 = None
            transf_distrib_ndesig2 = None
        self.extra_masks = 1
        return dna_kernel, transf_distrib_ndesig1, transf_distrib_ndesig2, transformed_l

    def build_network_core(self, action, current_state, input_image):
        lstm_func = basic_conv_lstm_cell


        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=self.reuse):

            enc0 = slim.layers.conv2d(  # 32x32x32
                input_image,
                32, [5, 5],
                stride=2,
                scope='scale1_conv1',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm1'})
            hidden1, self.lstm_state1 = self.lstm_func(  # 32x32x16
                enc0, self.lstm_state1, self.lstm_size[0], scope='state1')
            hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')

            enc1 = slim.layers.conv2d(  # 16x16x16
                hidden1, hidden1.get_shape()[3], [3, 3], stride=2, scope='conv2')
            hidden3, self.lstm_state3 = self.lstm_func(  # 16x16x32
                enc1, self.lstm_state3, self.lstm_size[1], scope='state3')
            hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')

            enc2 = slim.layers.conv2d(  # 8x8x32
                hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv3')

            if not 'ignore_state_action' in self.conf:
                # Pass in state and action.
                state_action = tf.concat(axis=1, values=[action, current_state])

                smear = tf.reshape(state_action,[int(self.batch_size), 1, 1, int(state_action.get_shape()[1])])
                smear = tf.tile(
                    smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])

                enc2 = tf.concat(axis=3, values=[enc2, smear])
            else:
                print 'ignoring states and actions'
            enc3 = slim.layers.conv2d(  # 8x8x32
                enc2, hidden3.get_shape()[3], [1, 1], stride=1, scope='conv4')
            hidden5, self.lstm_state5 = self.lstm_func(  # 8x8x64
                enc3, self.lstm_state5, self.lstm_size[2], scope='state5')
            hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')

            enc4 = slim.layers.conv2d_transpose(  # 16x16x64
                hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')
            hidden6, self.lstm_state6 = self.lstm_func(  # 16x16x32
                enc4, self.lstm_state6, self.lstm_size[3], scope='state6')
            hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')
            if 'noskip' not in self.conf:
                # Skip connection.
                hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16

            enc5 = slim.layers.conv2d_transpose(  # 32x32x32
                hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
            hidden7, self.lstm_state7 = self.lstm_func(  # 32x32x16
                enc5, self.lstm_state7, self.lstm_size[4], scope='state7')
            hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')
            if not 'noskip' in self.conf:
                # Skip connection.
                hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

            enc6 = slim.layers.conv2d_transpose(  # 64x64x16
                hidden7,
                hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm9'})

            if current_state != None:
                current_state = slim.layers.fully_connected(
                    state_action,
                    int(current_state.get_shape()[1]),
                    scope='state_pred',
                    activation_fn=None)
            self.gen_states.append(current_state)

            self.apply_trafo_predict(enc6, hidden5)

            return current_state


    def get_input_image(self, feedself, image, t):
        done_warm_start = len(self.gen_images) > self.context_frames - 1

        prev_pix_distrib1, prev_pix_distrib2 = None, None

        if feedself and done_warm_start:
            print 'feeding self'
            # Feed in generated image.
            prev_image = self.gen_images[-1]  # 64x64x6
            if self.trafo_pix:
                prev_pix_distrib1 = self.gen_distrib1[-1]
                if 'ndesig' in self.conf:
                    prev_pix_distrib2 = self.gen_distrib2[-1]
        elif done_warm_start:
            print 'doing sched sampling'
            # Scheduled sampling
            prev_image = scheduled_sample(image, self.gen_images[-1], self.batch_size,
                                          self.num_ground_truth)
        else:
            # Always feed in ground_truth
            print 'feeding gtruth'
            prev_image = image
            if self.trafo_pix:
                prev_pix_distrib1 = self.pix_distrib1[t]
                if 'ndesig' in self.conf:
                    prev_pix_distrib2 = self.pix_distrib2[t]
                if len(prev_pix_distrib1.get_shape()) == 3:
                    prev_pix_distrib1 = tf.expand_dims(prev_pix_distrib1, -1)
                    if 'ndesig' in self.conf:
                        prev_pix_distrib2 = tf.expand_dims(prev_pix_distrib2, -1)
                    else:
                        prev_pix_distrib2 = None
            else:
                prev_pix_distrib1 = None
                prev_pix_distrib2 = None

        return prev_image, prev_pix_distrib1, prev_pix_distrib2

    def mean_squared_error(self, true, pred):
        """L2 distance between tensors true and pred.

        Args:
          true: the ground truth image.
          pred: the predicted image.
        Returns:
          mean squared error between ground truth and predicted image.
        """
        return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


    def build_loss(self):
        summaries = []

        self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())

        if not self.trafo_pix:
            # L2 loss, PSNR for eval.
            true_fft_list, pred_fft_list = [], []
            loss, psnr_all = 0.0, 0.0

            for i, x, gx in zip(
                    range(len(self.gen_images)), self.images[self.conf['context_frames']:],
                    self.gen_images[self.conf['context_frames'] - 1:]):
                recon_cost_mse = self.mean_squared_error(x, gx)

                summaries.append(tf.summary.scalar('recon_cost' + str(i), recon_cost_mse))

                recon_cost = recon_cost_mse

                loss += recon_cost

            if ('ignore_state_action' not in self.conf) and ('ignore_state' not in self.conf):
                for i, state, gen_state in zip(
                        range(len(self.gen_states)), self.states[self.conf['context_frames']:],
                        self.gen_states[self.conf['context_frames'] - 1:]):
                    state_cost = self.mean_squared_error(state, gen_state) * 1e-4 * self.conf['use_state']
                    summaries.append(
                        tf.summary.scalar('state_cost' + str(i), state_cost))
                    loss += state_cost

            self.loss = loss = loss / np.float32(len(self.images) - self.conf['context_frames'])
            summaries.append(tf.summary.scalar('loss', loss))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            self.summ_op = tf.summary.merge(summaries)


    def fuse_trafos(self, enc6, background_image, transformed, scope):
        extra_masks = self.extra_masks

        masks = slim.layers.conv2d_transpose(
            enc6, (self.conf['num_masks']+ extra_masks), 1, stride=1, scope=scope)

        img_height = 64
        img_width = 64
        num_masks = self.conf['num_masks']

        if self.conf['model']=='DNA':
            if num_masks != 1:
                raise ValueError('Only one mask is supported for DNA model.')

        # the total number of masks is num_masks +extra_masks because of background and generated pixels!
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, num_masks +extra_masks])),
            [int(self.batch_size), int(img_height), int(img_width), num_masks +extra_masks])
        mask_list = tf.split(axis=3, num_or_size_splits=num_masks +extra_masks, value=masks)

        output = mask_list[0] * background_image
        for layer, mask in zip_equal(transformed, mask_list[1:]):
            output += layer * mask

        return output, mask_list

    def fuse_pix_distrib(self, mask_list, pix_distributions, prev_pix_distrib,
                         transf_distrib):

        if '1stimg_bckgd' in self.conf:
            background_pix = pix_distributions[0]
            background_pix = tf.expand_dims(background_pix, -1)
            print 'using pix_distrib-background from first image..'
        else:
            background_pix = prev_pix_distrib
        pix_distrib_output = mask_list[0] * background_pix
        for i in range(self.num_masks):
            pix_distrib_output += transf_distrib[i] * mask_list[i + self.extra_masks]
        return pix_distrib_output

    def fuse_pix_movebckgd(self, mask_list, transf_pix, transf_backgd_pix):
        pix_distrib = transf_backgd_pix + transf_pix

        pix_distrib_output = 0
        for pix, mask in zip_equal(pix_distrib, mask_list):
            pix_distrib_output += pix* mask
        return pix_distrib_output


    def visualize(self, sess):
        visualize(sess, self.conf, self)

    def visualize_diffmotions(self, sess):
        visualize_diffmotions(sess, self.conf, self)

    def compute_metric(self, sess, create_images):
        compute_metric(sess, self.conf, self, create_images)

def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data_files points.

    Args:
      ground_truth_x: tensor of ground-truth data_files points.
      generated_x: tensor of generated data_files points.
      batch_size: batch size
      num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)
    return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])
