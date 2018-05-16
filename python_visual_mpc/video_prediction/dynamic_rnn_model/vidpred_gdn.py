import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm

from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
from python_visual_mpc.video_prediction.dynamic_rnn_model.lstm_ops import BasicConv2DLSTMCell
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten, resample_layer, warp_pts_layer
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import sigmoid_kl_with_logits
from python_visual_mpc.video_prediction.dynamic_rnn_model.utils import preprocess, deprocess
from python_visual_mpc.video_prediction.basecls.utils.visualize import visualize_diffmotions, visualize, compute_metric
from python_visual_mpc.video_prediction.basecls.utils.compute_motion_vecs import compute_motion_vector_cdna, compute_motion_vector_dna
from python_visual_mpc.video_prediction.read_tf_records2 import build_tfrecord_input as build_tfrecord_fn

from .dynamic_base_model import mean_squared_error
import pdb
# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


from .dynamic_base_model import DNACell
from .dynamic_base_model import Dynamic_Base_Model
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet


from python_visual_mpc.video_prediction.utils_vpred.video_summary import make_video_summaries


class VidPred_GDN_Model():
    def __init__(self,
                 conf = None,
                 images=None,
                 actions=None,
                 states=None,
                 pix_distrib=None,
                 pix_distrib2=None,
                 trafo_pix = True,
                 load_data = True,
                 build_loss = True,
                 ):

        self.iter_num = tf.placeholder(tf.float32, [], name='iternum')

        if 'ndesig' in conf:
            ndesig = conf['ndesig']
        else:
            ndesig = 1
            conf['ndesig'] = 1

        if 'img_height' in conf:
            self.img_height = conf['img_height']
        else: self.img_height = 64
        if 'img_width' in conf:
            self.img_width = conf['img_width']
        else: self.img_width = 64

        if states is not None and states.get_shape().as_list()[1] != conf['sequence_length']:  # append zeros if states is shorter than sequence length
            states = tf.concat(
                [states, tf.zeros([conf['batch_size'], conf['sequence_length'] - conf['context_frames'], conf['sdim']])],
                axis=1)

        self.trafo_pix = trafo_pix
        if pix_distrib is not None:
            assert trafo_pix == True
            pix_distrib = tf.concat([pix_distrib, tf.zeros([conf['batch_size'], conf['sequence_length'] - conf['context_frames'],ndesig, self.img_height, self.img_width, 1])], axis=1)
            pix_distrib = pix_distrib

        use_state = True

        vgf_dim = 32

        self.trafo_pix = trafo_pix
        self.conf = conf

        k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_transformed_images = conf['num_transformed_images']
        self.context_frames = conf['context_frames']
        self.batch_size = conf['batch_size']

        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        print('base model uses traincond', self.train_cond)

        self.sdim = conf['sdim']
        self.adim = conf['adim']

        if images is None:
            pix_distrib = None
            if not load_data:
                self.actions_pl = tf.placeholder(tf.float32, name='actions',
                                                 shape=(conf['batch_size'], conf['sequence_length'], self.adim))
                actions = self.actions_pl

                self.states_pl = tf.placeholder(tf.float32, name='states',
                                                shape=(conf['batch_size'], conf['sequence_length'], self.sdim))
                states = self.states_pl

                self.images_pl = tf.placeholder(tf.float32, name='images',
                                                shape=(conf['batch_size'], conf['sequence_length'], self.img_height, self.img_width, 3))
                images = self.images_pl

                self.pix_distrib_pl = tf.placeholder(tf.float32, name='states',
                                                     shape=(conf['batch_size'], conf['sequence_length'], ndesig, self.img_height, self.img_width, 1))
                pix_distrib = self.pix_distrib_pl

            else:
                dict = build_tfrecord_fn(conf, training=True)
                train_images, train_actions, train_states = dict['images'], dict['actions'], dict['endeffector_pos']
                dict = build_tfrecord_fn(conf, training=False)
                val_images, val_actions, val_states = dict['images'], dict['actions'], dict['endeffector_pos']

                images, actions, states = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                                  lambda: [train_images, train_actions, train_states],
                                                  lambda: [val_images, val_actions, val_states])

            if 'use_len' in conf:
                print('randomly shift videos for data augmentation')
                images, states, actions  = self.random_shift(images, states, actions)

        ## start interface

        # Split into timesteps.
        actions = tf.split(axis=1, num_or_size_splits=actions.get_shape()[1], value=actions)
        actions = [tf.squeeze(act) for act in actions]
        states = tf.split(axis=1, num_or_size_splits=states.get_shape()[1], value=states)
        states = [tf.squeeze(st) for st in states]
        images = tf.split(axis=1, num_or_size_splits=images.get_shape()[1], value=images)
        images = [tf.squeeze(img) for img in images]
        if pix_distrib is not None:
            pix_distrib = tf.split(axis=1, num_or_size_splits=pix_distrib.get_shape()[1], value=pix_distrib)
            pix_distrib = [tf.reshape(pix, [self.batch_size, ndesig, self.img_height, self.img_width, 1]) for pix in pix_distrib]

        self.actions = actions
        self.images = images
        self.states = states

        image_shape = images[0].get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        images_length = len(images)
        sequence_length = images_length - 1
        _, action_dim = actions[0].get_shape().as_list()
        _, state_dim = states[0].get_shape().as_list()

        if k == -1:
            feedself = True,
            num_ground_truth = None
        else:
            # Scheduled sampling:
            # Calculate number of ground-truth frames to pass in.
            k = int(k)
            num_ground_truth_float = tf.to_float(batch_size) * (k / (k + tf.exp(self.iter_num / k)))
            num_ground_truth_floor = tf.floor(num_ground_truth_float)
            ceil_prob = num_ground_truth_float - num_ground_truth_floor
            floor_prob = 1 - ceil_prob
            log_probs = tf.log([floor_prob, ceil_prob])
            num_ground_truth = tf.to_int32(num_ground_truth_floor) + \
                tf.to_int32(tf.squeeze(tf.multinomial(log_probs[None, :], num_samples=1)))
            # Make it exactly zero if num_ground_truth_float is less than 1 / sequence_length.
            # This is to ensure that eventually after enough iterations, the model is
            # autoregressive (as opposed to autoregressive with very high probability).
            num_ground_truth = tf.cond(num_ground_truth_float < (1.0 / sequence_length),
                                       lambda: tf.constant(0, dtype=tf.int32),
                                       lambda: num_ground_truth)
            feedself = False

        first_pix_distrib = None if pix_distrib is None else pix_distrib[0]

        cell = DNACell(conf,
                       [height, width, color_channels],
                       state_dim,
                       first_image=images[0],
                       first_pix_distrib=first_pix_distrib,
                       num_ground_truth=num_ground_truth,
                       lstm_skip_connection=False,
                       feedself=feedself,
                       use_state=use_state,
                       vgf_dim=vgf_dim)

        inputs = [tf.stack(images[:sequence_length]), tf.stack(actions[:sequence_length]), tf.stack(states[:sequence_length])]
        if pix_distrib is not None:
            inputs.append(tf.stack(pix_distrib[:sequence_length]))
        if pix_distrib2 is not None:
            inputs.append(tf.stack(pix_distrib2[:sequence_length]))

        if 'float16' in conf:
            use_dtype = tf.float16
        else: use_dtype = tf.float32
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=[sequence_length] * batch_size, dtype=use_dtype,
                                       swap_memory=True, time_major=True)

        (gen_images, gen_states, gen_masks, gen_transformed_images), other_outputs = outputs[:5], outputs[5:]

        self.gen_images = tf.unstack(gen_images, axis=0)
        self.gen_states = tf.unstack(gen_states, axis=0)
        self.gen_masks = list(zip(*[tf.unstack(gen_mask, axis=0) for gen_mask in gen_masks]))
        self.gen_transformed_images = list(
            zip(*[tf.unstack(gen_transformed_image, axis=0) for gen_transformed_image in gen_transformed_images]))
        other_outputs = list(other_outputs)

        # making video summaries
        self.train_video_summaries = make_video_summaries(conf['sequence_length'], [self.images, self.gen_images], 'train_images')
        self.val_video_summaries = make_video_summaries(conf['sequence_length'], [self.images, self.gen_images], 'val_images')

        if 'compute_flow_map' in self.conf:
            gen_flow_map = other_outputs.pop(0)
            self.gen_flow_map = tf.unstack(gen_flow_map, axis=0)

        if pix_distrib is not None:
            self.gen_distrib = other_outputs.pop(0)
            self.gen_distrib = tf.unstack(self.gen_distrib, axis=0)
            self.gen_transformed_pixdistribs = other_outputs.pop(0)
            self.gen_transformed_pixdistribs = list(zip(
                *[tf.unstack(gen_transformed_pixdistrib, axis=0) for gen_transformed_pixdistrib in
                  self.gen_transformed_pixdistribs]))

        # put in gdn here!

        gen_images_stopped = [tf.stop_gradient(el) for el in self.gen_images]
        self.gdn = GoalDistanceNet(conf, images = self.images, pred_images = gen_images_stopped,
                                   load_data = False, iter_num=self.iter_num)

        assert not other_outputs

        if build_loss:
            self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
            loss = self.build_loss()

            self.gdn.add_pair_loss()
            loss += self.gdn.loss

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

            self.train_summ_op = tf.summary.merge([self.train_summ_op, self.gdn.train_summ_op])
            self.val_summ_op = tf.summary.merge([self.val_summ_op, self.gdn.val_summ_op])

    def build_loss(self):
        train_summaries = []
        val_summaries = []
        # L2 loss, PSNR for eval.
        loss, psnr_all = 0.0, 0.0

        for i, x, gx in zip(
                list(range(len(self.gen_images))), self.images[self.conf['context_frames']:],
                self.gen_images[self.conf['context_frames'] - 1:]):
            recon_cost_mse = mean_squared_error(x, gx)
            # train_summaries.append(tf.summary.scalar('recon_cost' + str(i), recon_cost_mse))
            # val_summaries.append(tf.summary.scalar('val_recon_cost' + str(i), recon_cost_mse))
            recon_cost = recon_cost_mse

            loss += recon_cost

        if ('ignore_state_action' not in self.conf) and ('ignore_state' not in self.conf):
            for i, state, gen_state in zip(
                    list(range(len(self.gen_states))), self.states[self.conf['context_frames']:],
                    self.gen_states[self.conf['context_frames'] - 1:]):
                state_cost = mean_squared_error(state, gen_state) * 1e-4 * self.conf['use_state']
                # train_summaries.append(tf.summary.scalar('state_cost' + str(i), state_cost))
                # val_summaries.append(tf.summary.scalar('val_state_cost' + str(i), state_cost))
                loss += state_cost

        self.loss = loss = loss / np.float32(len(self.images) - self.conf['context_frames'])
        train_summaries.append(tf.summary.scalar('loss', loss))
        val_summaries.append(tf.summary.scalar('val_loss', loss))

        self.train_summ_op = tf.summary.merge(train_summaries)
        self.val_summ_op = tf.summary.merge(val_summaries)

        return loss


    def visualize(self, sess):
        visualize(sess, self.conf, self)

    def visualize_diffmotions(self, sess):
        visualize_diffmotions(sess, self.conf, self)

    def compute_metric(self, sess, create_images):
        compute_metric(sess, self.conf, self, create_images)

    def random_shift(self, images, states, actions):
        print('shifting the video sequence randomly in time')
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


