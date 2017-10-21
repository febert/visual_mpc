import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm

from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
from python_visual_mpc.video_prediction.dynamic_rnn_model.lstm_ops import BasicConv2DLSTMCell
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import sigmoid_kl_with_logits
from python_visual_mpc.video_prediction.dynamic_rnn_model.utils import preprocess, deprocess
from python_visual_mpc.video_prediction.basecls.utils.visualize import visualize_diffmotions, visualize
from python_visual_mpc.video_prediction.basecls.utils.compute_motion_vecs import compute_motion_vector_cdna, compute_motion_vector_dna

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

class DNACell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 conf,
                 image_shape,
                 state_dim,
                 first_image,
                 num_ground_truth,
                 dilation_rate,
                 lstm_skip_connection,
                 feedself,
                 use_state,
                 vgf_dim,
                 reuse=None,
                 dependent_mask = True,
                 trafo_pix=False,
                 first_pix_distrib=None):

        super(DNACell, self).__init__(_reuse=reuse)

        self.image_shape = image_shape
        self.state_dim = state_dim
        self.first_image = first_image
        self.frist_pix_distrib = first_pix_distrib


        self.num_ground_truth = num_ground_truth
        self.kernel_size = [conf['kern_size'], conf['kern_size']]
        self.dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
        self.lstm_skip_connection = lstm_skip_connection
        self.feedself = feedself
        self.use_state = use_state
        self.num_transformed_images = conf['num_masks']

        self.model = conf['model']

        self.context_frames = conf['context_frames']
        self.conf = conf
        self.trafo_pix = trafo_pix

        if '1stimg_bckgd' in conf:
            self.first_image_background = True
        else: self.first_image_background = False

        if 'previmg_bckgd' in conf:
            self.prev_image_background = True
        else: self.prev_image_background = False

        if 'gen_img' in conf:
            self.generate_scratch_image = True
        else: self.generate_scratch_image = False

        self.dependent_mask = dependent_mask
        self.vgf_dim = vgf_dim

        self.layer_normalization = conf['normalization']
        if conf['normalization'] == 'ln':
            self.normalizer_fn = layer_norm
        elif conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        elif conf['normalization'] == 'none':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % self.layer_normalization)

        # Compute output_size
        height, width, _ = self.image_shape
        num_masks = int(bool(self.first_image_background)) + \
                    int(bool(self.prev_image_background)) + \
                    int(bool(self.generate_scratch_image)) + \
                    self.num_transformed_images

        output_size = [
            tf.TensorShape(self.image_shape),  # gen_image
            tf.TensorShape([self.state_dim]),  # gen_state
            [tf.TensorShape([height, width, 1])] * num_masks,  # masks
            [tf.TensorShape(self.image_shape)] * num_masks,  # transformed_images
        ]

        if 'visual_flowvec' in self.conf:
            output_size.append(tf.TensorShape([height, width, 2]))

        if trafo_pix:
            output_size.append(tf.TensorShape([height, width, 1]))

        self._output_size = tuple(output_size)

        # Compute state_size
        if self.lstm_skip_connection:
            lstm_filters_multiplier = 2
        else:
            lstm_filters_multiplier = 1
        lstm_cell_sizes = [
            tf.TensorShape([height / 2, width / 2, self.vgf_dim]),
            tf.TensorShape([height / 4, width / 4, self.vgf_dim * 2]),
            tf.TensorShape([height / 8, width / 8, self.vgf_dim * 4]),
            tf.TensorShape([height / 4, width / 4, self.vgf_dim * 2]),
            tf.TensorShape([height / 2, width / 2, self.vgf_dim]),
        ]
        lstm_state_sizes = [
            tf.TensorShape([height / 2, width / 2, lstm_filters_multiplier * self.vgf_dim]),
            tf.TensorShape([height / 4, width / 4, lstm_filters_multiplier * self.vgf_dim * 2]),
            tf.TensorShape([height / 8, width / 8, lstm_filters_multiplier * self.vgf_dim * 4]),
            tf.TensorShape([height / 4, width / 4, lstm_filters_multiplier * self.vgf_dim * 2]),
            tf.TensorShape([height / 2, width / 2, lstm_filters_multiplier * self.vgf_dim]),
        ]
        lstm_state_size = [tf.nn.rnn_cell.LSTMStateTuple(lstm_cell_size, lstm_state_size)
                           for lstm_cell_size, lstm_state_size in zip(lstm_cell_sizes, lstm_state_sizes)]
        lstm_state_size = tuple(lstm_state_size)
        state_size = [
            tf.TensorShape(self.image_shape),
            tf.TensorShape([self.state_dim]),
            tf.TensorShape([]),
            lstm_state_size,
        ]

        if trafo_pix:
            state_size.append(tf.TensorShape([height, width, 1]))

        self._state_size = tuple(state_size)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def _lstm_func(self, inputs, state, filters):
        inputs_shape = inputs.get_shape().as_list()
        batch_size= inputs_shape[0]
        input_shape = inputs_shape[1:]
        lstm_cell = BasicConv2DLSTMCell(input_shape, filters, kernel_size=(5, 5),
                                        normalizer_fn=None if self.layer_normalization == 'none' else self.normalizer_fn,
                                        separate_norms=self.layer_normalization == 'ln',
                                        skip_connection=self.lstm_skip_connection, reuse=tf.get_variable_scope().reuse)
        # if state is None:
        #     state = lstm_cell.zero_state(batch_size, tf.float32)
        return lstm_cell(inputs, state)

    def call(self, inputs, states):
        image, action, state = inputs
        gen_image, gen_state, time, lstm_states = states
        lstm_state0, lstm_state1, lstm_state2, lstm_state3, lstm_state4 = lstm_states

        image_shape = image.get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        _, state_dim = state.get_shape().as_list()
        kernel_size = self.kernel_size
        dilation_rate = self.dilation_rate
        num_transformed_images = self.num_transformed_images
        vgf_dim = self.vgf_dim

        done_warm_start = time > self.context_frames - 1
        if self.feedself:
            image = tf.cond(tf.reduce_all(done_warm_start),
                            lambda: gen_image,  # feed in generated image
                            lambda: image)  # feed in ground_truth
        else:
            image = tf.cond(tf.reduce_all(done_warm_start),
                            lambda: scheduled_sample(image, gen_image, batch_size, self.num_ground_truth),  # schedule sampling
                            lambda: image)  # feed in ground_truth
        state = tf.cond(tf.reduce_all(time == 0),
                        lambda: state,  # feed in ground_truth state only for first time step
                        lambda: gen_state)  # feed in predicted state
        state_action = tf.concat([action, state], axis=-1)

        with tf.variable_scope('h0'):
            h0 = conv_pool2d(image, vgf_dim, kernel_size=(5, 5), strides=(2, 2))
            h0 = self.normalizer_fn(h0)
            h0 = tf.nn.relu(h0)

        with tf.variable_scope('lstm_h0'):
            lstm_h0, lstm_state0 = self._lstm_func(h0, lstm_state0, vgf_dim)

        with tf.variable_scope('h1'):
            h1 = conv_pool2d(lstm_h0, vgf_dim * 2, kernel_size=(3, 3), strides=(2, 2))
            h1 = self.normalizer_fn(h1)
            h1 = tf.nn.relu(h1)

        with tf.variable_scope('lstm_h1'):
            lstm_h1, lstm_state1 = self._lstm_func(h1, lstm_state1, vgf_dim * 2)

        with tf.variable_scope('h2'):
            h2 = conv_pool2d(lstm_h1, vgf_dim * 4, kernel_size=(3, 3), strides=(2, 2))
            h2 = self.normalizer_fn(h2)
            h2 = tf.nn.relu(h2)

        # Pass in state and action.
        if self.use_state:
            with tf.variable_scope('state_action_h2'):
                state_action_smear = tf.tile(state_action[:, None, None, :],
                                             [1, h2.get_shape().as_list()[1], h2.get_shape().as_list()[2], 1])
                state_action_h2 = tf.concat([h2, state_action_smear], axis=-1)
                state_action_h2 = conv2d(state_action_h2, vgf_dim * 4, kernel_size=(1, 1), strides=(1, 1))
                # TODO: consider adding normalizer and relu here
        else:
            state_action_h2 = h2

        with tf.variable_scope('lstm_h2'):
            lstm_h2, lstm_state2 = self._lstm_func(state_action_h2, lstm_state2, vgf_dim * 4)

        with tf.variable_scope('h3'):
            h3 = upsample_conv2d(lstm_h2, vgf_dim * 2, kernel_size=(3, 3), strides=(2, 2))
            h3 = self.normalizer_fn(h3)
            h3 = tf.nn.relu(h3)

        with tf.variable_scope('lstm_h3'):
            lstm_h3, lstm_state3 = self._lstm_func(h3, lstm_state3, vgf_dim * 2)

        with tf.variable_scope('h4'):
            h4 = upsample_conv2d(tf.concat([lstm_h3, h1], axis=-1), vgf_dim, kernel_size=(3, 3), strides=(2, 2))
            h4 = self.normalizer_fn(h4)
            h4 = tf.nn.relu(h4)

        with tf.variable_scope('lstm_h4'):
            lstm_h4, lstm_state4 = self._lstm_func(h4, lstm_state4, vgf_dim)

        with tf.variable_scope('h5'):
            h5 = upsample_conv2d(tf.concat([lstm_h4, h0], axis=-1), vgf_dim, kernel_size=(3, 3), strides=(2, 2))
            h5 = self.normalizer_fn(h5)
            h5 = tf.nn.relu(h5)

        if self.model == 'dna':
            with tf.variable_scope('h6_dna_kernel'):
                h6_dna_kernel = conv2d(h5, vgf_dim, kernel_size=(3, 3), strides=(1, 1))
                h6_dna_kernel = self.normalizer_fn(h6_dna_kernel)
                h6_dna_kernel = tf.nn.relu(h6_dna_kernel)

        if self.generate_scratch_image:
            with tf.variable_scope('h6_scratch'):
                h6_scratch = conv2d(h5, vgf_dim, kernel_size=(3, 3), strides=(1, 1))
                h6_scratch = self.normalizer_fn(h6_scratch)
                h6_scratch = tf.nn.relu(h6_scratch)

        with tf.variable_scope('h6_masks'):
            h6_masks = conv2d(h5, vgf_dim, kernel_size=(3, 3), strides=(1, 1))
            h6_masks = self.normalizer_fn(h6_masks)
            h6_masks = tf.nn.relu(h6_masks)

        if self.model == 'DNA':
            # Using largest hidden state for predicting untied conv kernels.
            with tf.variable_scope('dna_kernels'):
                if dilation_rate == [1, 1]:
                    kernels = conv2d(h6_dna_kernel, kernel_size[0] * kernel_size[1] * num_transformed_images,
                                     kernel_size=(3, 3), strides=(1, 1))
                else:
                    kernels = conv_pool2d(h6_dna_kernel, kernel_size[0] * kernel_size[1] * num_transformed_images,
                                          kernel_size=(3, 3), strides=dilation_rate)
                kernels = tf.reshape(kernels, [batch_size, height // dilation_rate[0], width // dilation_rate[1],
                                               kernel_size[0], kernel_size[1], num_transformed_images])
            kernel_spatial_axes = [3, 4]
        elif self.model == 'CDNA':
            with tf.variable_scope('cdna_kernels'):
                kernels = dense(flatten(lstm_h2), kernel_size[0] * kernel_size[1] * num_transformed_images)
                kernels = tf.reshape(kernels, [batch_size, kernel_size[0], kernel_size[1], num_transformed_images])
            kernel_spatial_axes = [1, 2]
        else:
            raise ValueError('Invalid model %s' % self.model)

        if isinstance(kernels, (list, tuple)):
            normalized_kernels = []
            for kernel in kernels:
                kernel = tf.nn.relu(kernel - RELU_SHIFT) + RELU_SHIFT
                kernel /= tf.reduce_sum(kernel, axis=kernel_spatial_axes, keep_dims=True)
                normalized_kernels.append(kernel)
            kernels = normalized_kernels
        else:
            kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
            kernels /= tf.reduce_sum(kernels, axis=kernel_spatial_axes, keep_dims=True)

        transformed_images = []
        if self.first_image_background:
            transformed_images.append(self.first_image)
        if self.prev_image_background:
            transformed_images.append(image)
        if self.generate_scratch_image:
            # Using largest hidden state for predicting a new image layer.
            # This allows the network to also generate one image from scratch,
            # which is useful when regions of the image become unoccluded.
            with tf.variable_scope('scratch_image'):
                scratch_image = conv2d(h6_scratch, image_shape[-1], kernel_size=(3, 3), strides=(1, 1))
                scratch_image = tf.nn.sigmoid(scratch_image)
                transformed_images.append(scratch_image)

        if self.model == 'DNA':
            with tf.variable_scope('transformed'):
                transformed_images += apply_dna_kernels(image, kernels, dilation_rate=dilation_rate)

        elif self.model == 'CDNA':
            with tf.variable_scope('transformed'):
                transformed_images += apply_cdna_kernels(image, kernels, dilation_rate=dilation_rate)
        else:
            raise ValueError('Invalid model %s' % self.model)

        with tf.variable_scope('masks'):
            if self.dependent_mask:
                mask_inputs = tf.concat([h6_masks] + transformed_images, axis=-1)
            else:
                mask_inputs = h6_masks
            masks = conv2d(mask_inputs, len(transformed_images), kernel_size=(3, 3), strides=(1, 1))
            masks = tf.nn.softmax(masks)
            masks = tf.split(masks, len(transformed_images), axis=-1)

        if 'visual_flowvec' in self.conf:
            if self.model == 'CDNA':
                kernels = tf.transpose(kernels, [1,2,0,3])
                motion_vecs = compute_motion_vector_cdna(self.conf, kernels)

            if self.model == 'DNA':
                motion_vecs = compute_motion_vector_dna(self.conf, kernels)

            output = tf.zeros([self.conf['batch_size'], 64, 64, 2])
            for vec, mask in zip(motion_vecs, masks[-num_transformed_images:]):
                if self.conf['model'] == 'CDNA':
                    vec = tf.reshape(vec, [self.conf['batch_size'], 1, 1, 2])
                    vec = tf.tile(vec, [1, 64, 64, 1])
                output += vec * mask
            flow_vectors = output

        assert len(transformed_images) == len(masks)
        gen_image = tf.add_n([transformed_image * mask for transformed_image, mask in zip(transformed_images, masks)])

        with tf.variable_scope('state_pred'):
            gen_state = dense(state_action, state_dim)

        if 'visual_flowvec' in self.conf:
            outputs = gen_image, gen_state, masks, transformed_images, flow_vectors
        else:
            outputs = gen_image, gen_state, masks, transformed_images

        new_lstm_states = lstm_state0, lstm_state1, lstm_state2, lstm_state3, lstm_state4
        new_states = gen_image, gen_state, time + 1, new_lstm_states
        return outputs, new_states


class Base_Prediction_Model(object):
    def __init__(self,
                conf = None,
                trafo_pix = True,
                load_data = True,
                ):

        self.iter_num = tf.placeholder(tf.float32, [])
        kernel_size = (5, 5)
        dilation_rate = (1, 1)
        use_state = True

        vgf_dim = 32
        rnn_type = None

        self.trafo_pix = trafo_pix
        self.conf = conf

        k = conf['schedsamp_k']
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
            pix_distrib1 = self.pix_distrib_pl

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

        if 'use_len' in conf:
            print 'randomly shift videos for data augmentation'
            images, states, actions  = self.random_shift(images, states, actions)

        ## start interface

        # Split into timesteps.
        actions = tf.split(axis=1, num_or_size_splits=actions.get_shape()[1], value=actions)
        actions = [tf.squeeze(act) for act in actions]
        states = tf.split(axis=1, num_or_size_splits=states.get_shape()[1], value=states)
        states = [tf.squeeze(st) for st in states]
        images = tf.split(axis=1, num_or_size_splits=images.get_shape()[1], value=images)
        images = [tf.squeeze(img) for img in images]


        self.actions = actions
        self.images = images
        self.states = states

        if trafo_pix:
            pix_distrib1 = tf.split(axis=1, num_or_size_splits=pix_distrib1.get_shape()[1], value=pix_distrib1)
            self.pix_distrib1 = [tf.squeeze(pix) for pix in pix_distrib1]

        dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
        rnn_type = rnn_type or 'dynamic'
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

        cell = DNACell(conf,
                       [height, width, color_channels],
                       state_dim,
                       first_image=images[0],
                       num_ground_truth=num_ground_truth,
                       dilation_rate=dilation_rate,
                       lstm_skip_connection=False,
                       feedself=feedself,
                       use_state=use_state,
                       vgf_dim=vgf_dim)

        if rnn_type == 'dynamic':
            inputs = [tf.stack(images[:sequence_length]), tf.stack(actions[:sequence_length]), tf.stack(states[:sequence_length])]
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=[sequence_length] * batch_size, dtype=tf.float32,
                                           swap_memory=True, time_major=True)

            if 'visual_flowvec' in self.conf:
                gen_images, gen_states, gen_masks, gen_transformed_images, flow_vectors = outputs[:5]
                self.prediction_flow = tf.unstack(flow_vectors, axis=0)
            else:
                gen_images, gen_states, gen_masks, gen_transformed_images = outputs[:4]
            self.gen_images = tf.unstack(gen_images, axis=0)
            self.gen_states = tf.unstack(gen_states, axis=0)
            self.gen_masks = list(zip(*[tf.unstack(gen_mask, axis=0) for gen_mask in gen_masks]))
            self.gen_transformed_images = list(zip(*[tf.unstack(gen_transformed_image, axis=0) for gen_transformed_image in gen_transformed_images]))
        elif rnn_type == 'static':
            # Slower to compile than dynamic_rnn yet runtime performance is very similar.
            # Raises OOM error when images are too large.
            inputs = list(zip(images[:sequence_length], actions[:sequence_length], states[:sequence_length]))
            outputs, _ = tf.nn.static_rnn(cell, inputs, dtype=tf.float32, sequence_length=[sequence_length] * batch_size)
            self.gen_images, self.gen_states, self.gen_masks, self.gen_transformed_images = list(zip(*outputs))[:4]
        else:
            raise ValueError('Invalid rnn type %s' % rnn_type)

        summaries = []

        self.global_step = tf.Variable(0, trainable=False)
        if self.conf['learning_rate'] == 'scheduled' and not self.visualize:
            print('using scheduled learning rate')
            self.lr = tf.train.piecewise_constant(self.global_step, self.conf['lr_boundaries'], self.conf['lr_values'])
        else:
            self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())

        self.gen_transformed_images = tf.unstack(self.gen_transformed_images, axis=0)
        self.gen_images = tf.unstack(self.gen_images, axis=0)
        self.gen_states = tf.unstack(self.gen_states, axis=0)

        if not self.trafo_pix:
            # L2 loss, PSNR for eval.
            true_fft_list, pred_fft_list = [], []
            loss, psnr_all = 0.0, 0.0

            for i, x, gx in zip(
                    range(len(self.gen_images)), self.images[self.conf['context_frames']:],
                    self.gen_images[self.conf['context_frames'] - 1:]):
                recon_cost_mse = mean_squared_error(x, gx)
                summaries.append(tf.summary.scalar('recon_cost' + str(i), recon_cost_mse))
                recon_cost = recon_cost_mse

                loss += recon_cost

            if ('ignore_state_action' not in self.conf) and ('ignore_state' not in self.conf):
                for i, state, gen_state in zip(
                        range(len(self.gen_states)), self.states[self.conf['context_frames']:],
                        self.gen_states[self.conf['context_frames'] - 1:]):
                    state_cost = mean_squared_error(state, gen_state) * 1e-4 * self.conf['use_state']
                    summaries.append(
                        tf.summary.scalar('state_cost' + str(i), state_cost))
                    loss += state_cost

            self.loss = loss = loss / np.float32(len(self.images) - self.conf['context_frames'])
            summaries.append(tf.summary.scalar('loss', loss))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, self.global_step)
            self.summ_op = tf.summary.merge(summaries)

    def visualize(self, sess):
        visualize(sess, self.model)

    def visualize_diffmotions(self, sess):
        visualize_diffmotions(sess, self.model)

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


def apply_dna_kernels_non_dilated(image, kernels):
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, height, width, kernel_size, num_transformed_images = kernels.get_shape().as_list()
    # Flatten the spatial dimensions.
    kernels_reshaped = tf.reshape(kernels, [batch_size, height, width,
                                            kernel_size[0] * kernel_size[1], num_transformed_images])
    image_padded = pad2d(image, kernel_size, padding='SAME', mode='SYMMETRIC')
    # Combine channel and batch dimensions into the first dimension.
    image_transposed = tf.transpose(image_padded, [3, 0, 1, 2])
    image_reshaped = flatten(image_transposed, 0, 1)[..., None]
    patches_reshaped = tf.extract_image_patches(image_reshaped, ksizes=[1] + kernel_size + [1],
                                                strides=[1] * 4, rates=[1] * 4, padding='VALID')
    # Separate channel and batch dimensions.
    patches = tf.reshape(patches_reshaped, [color_channels, batch_size, height, width, kernel_size[0] * kernel_size[1]])
    # Reduce along the spatial dimensions of the kernel.
    outputs = tf.reduce_sum(patches[..., None] * kernels_reshaped[None, ...], axis=-2)
    # Swap channel and transformation dimensions.
    outputs = tf.transpose(outputs, [4, 1, 2, 3, 0])
    outputs = tf.unstack(outputs, axis=0)
    return outputs


def apply_dna_kernels_dilated(image, kernels, dilation_rate=(1, 1)):
    dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, kernel_height, kernel_width, kernel_size, num_transformed_images = kernels.get_shape().as_list()
    # Flatten the spatial dimensions.
    kernels_reshaped = tf.reshape(kernels, [batch_size, kernel_height, kernel_width,
                                            kernel_size[0] * kernel_size[1], num_transformed_images])
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # for dilation = [2, 2], this is equivalent to this:
    # small_images = [image[:, 0::2, 0::2, :], image[:, 0::2, 1::2, :], image[:, 1::2, 0::2, :], image[:, 1::2, 1::2, :]]
    small_images = tf.space_to_batch_nd(image_padded, dilation_rate, paddings=[[0, 0]] * 2)
    small_images = tf.reshape(small_images, [dilation_rate[0] * dilation_rate[1], batch_size,
                                             image_padded.get_shape().as_list()[1] // dilation_rate[0],
                                             image_padded.get_shape().as_list()[2] // dilation_rate[1],
                                             color_channels])
    small_images = tf.unstack(small_images, axis=0)
    small_outputs = []
    for small_image in small_images:
        # Combine channel and batch dimensions into the first dimension.
        image_transposed = tf.transpose(small_image, [3, 0, 1, 2])
        image_reshaped = flatten(image_transposed, 0, 1)[..., None]
        patches_reshaped = tf.extract_image_patches(image_reshaped, ksizes=[1] + kernel_size + [1],
                                                    strides=[1] * 4, rates=[1] * 4, padding='VALID')
        # Separate channel and batch dimensions.
        patches = tf.reshape(patches_reshaped, [color_channels, batch_size,
                                                height // dilation_rate[0], width // dilation_rate[1],
                                                kernel_size[0] * kernel_size[1]])
        # Reduce along the spatial dimensions of the kernel.
        outputs = tf.reduce_sum(patches[..., None] * kernels_reshaped[None, ...], axis=-2)
        # Swap channel and transformation dimensions.
        outputs = tf.transpose(outputs, [4, 1, 2, 3, 0])
        outputs = tf.unstack(outputs, axis=0)
        small_outputs.append(outputs)
    small_outputs = list(zip(*small_outputs))
    small_outputs = [tf.reshape(small_output, [dilation_rate[0] * dilation_rate[1] * batch_size,
                                               height // dilation_rate[0], width // dilation_rate[1], color_channels])
                     for small_output in small_outputs]
    outputs = [tf.batch_to_space_nd(small_output, dilation_rate, crops=[[0, 0]] * 2) for small_output in small_outputs]
    return outputs


def apply_dna_kernels(image, kernels, dilation_rate=(1, 1)):
    dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
    if dilation_rate == [1, 1]:
        outputs = apply_dna_kernels_non_dilated(image, kernels)
    else:
        outputs = apply_dna_kernels_dilated(image, kernels, dilation_rate=dilation_rate)
    return outputs


def apply_cdna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, kernel_size_r, kernel_size_c, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_size_r, kernel_size_c]

    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    kernels = tf.transpose(kernels, [1, 2, 0, 3])
    kernels = tf.reshape(kernels, [kernel_size[0], kernel_size[1], batch_size, num_transformed_images])
    # Swap the batch and channel dimensions.
    image_transposed = tf.transpose(image_padded, [3, 1, 2, 0])
    # Transform image.
    outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
    # Transpose the dimensions to where they belong.
    outputs = tf.reshape(outputs, [color_channels, height, width, batch_size, num_transformed_images])
    outputs = tf.transpose(outputs, [4, 3, 1, 2, 0])
    outputs = tf.unstack(outputs, axis=0)
    return outputs


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data points.

    Args:
        ground_truth_x: tensor of ground-truth data points.
        generated_x: tensor of generated data points.
        batch_size: batch size
        num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
        New batch with num_ground_truth sampled from ground_truth_x and the rest
        from generated_x.
    """
    ground_truth_mask = tf.concat([tf.zeros([num_ground_truth], dtype=tf.int32),
                                   tf.ones([batch_size - num_ground_truth], dtype=tf.int32)], axis=0)
    ground_truth_mask = tf.reshape(ground_truth_mask, [batch_size])
    ground_truth_mask = tf.random_shuffle(ground_truth_mask)

    ground_truth_partitioned = tf.dynamic_partition(ground_truth_x, ground_truth_mask, 2)
    generated_partitioned = tf.dynamic_partition(generated_x, ground_truth_mask, 2)
    stitch_indices = tf.dynamic_partition(tf.range(batch_size), ground_truth_mask, 2)
    stitch_data = [ground_truth_partitioned[0], generated_partitioned[1]]
    outputs = tf.dynamic_stitch(stitch_indices, stitch_data)
    outputs = tf.reshape(outputs, [int(batch_size)] + outputs.get_shape().as_list()[1:])
    return outputs

def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))



