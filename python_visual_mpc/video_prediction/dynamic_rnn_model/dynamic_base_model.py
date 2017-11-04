import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm

from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
from python_visual_mpc.video_prediction.dynamic_rnn_model.lstm_ops import BasicConv2DLSTMCell
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import sigmoid_kl_with_logits
from python_visual_mpc.video_prediction.dynamic_rnn_model.utils import preprocess, deprocess
from python_visual_mpc.video_prediction.basecls.utils.visualize import visualize_diffmotions, visualize, compute_metric
from python_visual_mpc.video_prediction.basecls.utils.compute_motion_vecs import compute_motion_vector_cdna, compute_motion_vector_dna
import pdb
# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

class DNACell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 conf,
                 image_shape,
                 state_dim,
                 first_image,
                 first_pix_distrib1,
                 first_pix_distrib2,
                 num_ground_truth,
                 lstm_skip_connection,
                 feedself,
                 use_state,
                 vgf_dim,
                 reuse=None,
                 dependent_mask = True,
                 trafo_pix=False,
                 ):

        super(DNACell, self).__init__(_reuse=reuse)

        self.image_shape = image_shape
        self.state_dim = state_dim
        self.first_image = first_image
        self.first_pix_distrib1 = first_pix_distrib1

        if 'ndesig' in conf:
            self.first_pix_distrib2 = first_pix_distrib2

        self.num_ground_truth = num_ground_truth
        self.kernel_size = [conf['kern_size'], conf['kern_size']]
        if 'dilation_rate' in conf:
            dilation_rate = conf['dilation_rate']
        else: dilation_rate = (1,1)
        self.dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2

        self.lstm_skip_connection = lstm_skip_connection
        self.feedself = feedself
        self.use_state = use_state
        self.num_transformed_images = conf['num_transformed_images']

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

        # output_size
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
            tf.TensorShape([height, width, 2]),  # flow_map
        ]
        if self.first_pix_distrib1 is not None:
            output_size.append(tf.TensorShape([height, width, 1]))  # pix_distrib1
            output_size.append([tf.TensorShape([height, width, 1])] * num_masks)  # transformed_pix_distribs1
            if 'ndesig' in conf:
                output_size.append(tf.TensorShape([height, width, 1]))  # pix_distrib2
                output_size.append([tf.TensorShape([height, width, 1])] * num_masks)  # transformed_pix_distribs2

        self._output_size = tuple(output_size)

        # state_size
        if self.lstm_skip_connection:
            lstm_filters_multiplier = 2
        else:
            lstm_filters_multiplier = 1
        lstm_cell_sizes = [
            tf.TensorShape([height // 2, width // 2, self.vgf_dim]),
            tf.TensorShape([height // 4, width // 4, self.vgf_dim * 2]),
            tf.TensorShape([height // 8, width // 8, self.vgf_dim * 4]),
            tf.TensorShape([height // 4, width // 4, self.vgf_dim * 2]),
            tf.TensorShape([height // 2, width // 2, self.vgf_dim]),
        ]
        lstm_state_sizes = [
            tf.TensorShape([height // 2, width // 2, lstm_filters_multiplier * self.vgf_dim]),
            tf.TensorShape([height // 4, width // 4, lstm_filters_multiplier * self.vgf_dim * 2]),
            tf.TensorShape([height // 8, width // 8, lstm_filters_multiplier * self.vgf_dim * 4]),
            tf.TensorShape([height // 4, width // 4, lstm_filters_multiplier * self.vgf_dim * 2]),
            tf.TensorShape([height // 2, width // 2, lstm_filters_multiplier * self.vgf_dim]),
        ]
        state_size = [
            tuple([tf.nn.rnn_cell.LSTMStateTuple(lstm_cell_size, lstm_state_size)
                   for lstm_cell_size, lstm_state_size in zip(lstm_cell_sizes, lstm_state_sizes)]),  # lstm_states
            tf.TensorShape([]),  # time
            tf.TensorShape(self.image_shape),  # gen_image
            tf.TensorShape([self.state_dim]),  # gen_state
        ]
        if self.first_pix_distrib1 is not None:
            state_size.append(tf.TensorShape([height, width, 1]))  # gen_pix_distrib1
            if 'ndesig' in conf:
                state_size.append(tf.TensorShape([height, width, 1]))  # gen_pix_distrib1
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
        # inputs
        (image, action, state), other_inputs = inputs[:3], inputs[3:]
        if other_inputs:
            if 'ndesig' in self.conf:
                pix_distrib1, pix_distrib2 = other_inputs
            else: pix_distrib1 = other_inputs

        # states
        (lstm_states, time, gen_image, gen_state), other_states = states[:4], states[4:]
        lstm_state0, lstm_state1, lstm_state2, lstm_state3, lstm_state4 = lstm_states
        if other_states:
            if 'ndesig' in self.conf:
                gen_pix_distrib1, gen_pix_distrib2 = other_states
            else: gen_pix_distrib1 = other_states

        image_shape = image.get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        assert height == width
        scale_size = height
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
            if self.first_pix_distrib1 is not None:
                pix_distrib1 = tf.cond(tf.reduce_all(done_warm_start),
                                      lambda: gen_pix_distrib1,  # feed in generated pixel distribution
                                      lambda: pix_distrib1)  # feed in ground_truth
                if 'ndesig' in self.conf:
                    pix_distrib2 = tf.cond(tf.reduce_all(done_warm_start),
                                           lambda: gen_pix_distrib2,  # feed in generated pixel distribution
                                           lambda: pix_distrib2)  # feed in ground_truth
        else:
            image = tf.cond(tf.reduce_all(done_warm_start),
                            lambda: scheduled_sample(image, gen_image, batch_size, self.num_ground_truth),
                            # schedule sampling
                            lambda: image)  # feed in ground_truth
            if self.first_pix_distrib1 is not None:
                raise NotImplementedError
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

        if self.model == 'dna':
            # Using largest hidden state for predicting untied conv kernels.
            with tf.variable_scope('dna_kernels'):
                kernels = conv2d(h6_dna_kernel, kernel_size[0] * kernel_size[1] * num_transformed_images,
                                 kernel_size=(3, 3), strides=(1, 1))
                kernels = tf.reshape(kernels, [batch_size, height, width,
                                               kernel_size[0], kernel_size[1], num_transformed_images])
            kernel_spatial_axes = [3, 4]
        elif self.model == 'cdna':
            with tf.variable_scope('cdna_kernels'):
                kernels = dense(flatten(lstm_h2), kernel_size[0] * kernel_size[1] * num_transformed_images)
                kernels = tf.reshape(kernels, [batch_size, kernel_size[0], kernel_size[1], num_transformed_images])
            kernel_spatial_axes = [1, 2]
        else:
            raise ValueError('Invalid model %s' % self.model)

        with tf.name_scope('kernel_normalization'):
            kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
            kernels /= tf.reduce_sum(kernels, axis=kernel_spatial_axes, keep_dims=True)

        transformed_images = []
        with tf.name_scope('transformed_images'):
            transformed_images += apply_kernels(image, kernels, dilation_rate=dilation_rate)
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

        if self.first_pix_distrib1 is not None:
            transformed_pix_distribs1 = []
            with tf.name_scope('transformed_pix_distrib'):
                transformed_pix_distribs1 += apply_kernels(pix_distrib1, kernels, dilation_rate=dilation_rate)
            if self.first_image_background:
                transformed_pix_distribs1.append(self.first_pix_distrib1)
            if self.prev_image_background:
                transformed_pix_distribs1.append(pix_distrib1)
            if self.generate_scratch_image:
                transformed_pix_distribs1.append(tf.zeros_like(pix_distrib1))

            if 'ndesig' in self.conf:
                transformed_pix_distribs2 = []
                with tf.name_scope('transformed_pix_distrib'):
                    transformed_pix_distribs2 += apply_kernels(pix_distrib2, kernels, dilation_rate=dilation_rate)
                if self.first_image_background:
                    transformed_pix_distribs2.append(self.first_pix_distrib2)
                if self.prev_image_background:
                    transformed_pix_distribs2.append(pix_distrib2)
                if self.generate_scratch_image:
                    transformed_pix_distribs2.append(tf.zeros_like(pix_distrib2))

        with tf.variable_scope('masks'):
            if self.dependent_mask:
                mask_inputs = tf.concat([h6_masks] + transformed_images, axis=-1)
            else:
                mask_inputs = h6_masks
            masks = conv2d(mask_inputs, len(transformed_images), kernel_size=(3, 3), strides=(1, 1))
            masks = tf.nn.softmax(masks)
            masks = tf.split(masks, len(transformed_images), axis=-1)

        with tf.name_scope('gen_image'):
            assert len(transformed_images) == len(masks)
            gen_image = tf.add_n([transformed_image * mask
                                  for transformed_image, mask in zip(transformed_images, masks)])

        with tf.name_scope('flow_map'):
            flow_map = compute_flow_map(kernels, masks[:num_transformed_images])

        if self.first_pix_distrib1 is not None:
            with tf.name_scope('gen_pix_distrib'):
                assert len(transformed_pix_distribs1) <= len(masks) <= len(
                    transformed_pix_distribs1) + 1  # there might be an extra mask because of the scratch image
                gen_pix_distrib1 = tf.add_n([transformed_pix_distrib * mask
                                            for transformed_pix_distrib, mask in zip(transformed_pix_distribs1, masks)])
            if 'ndesig' in self.conf:
                assert len(transformed_pix_distribs2) <= len(masks) <= len(
                    transformed_pix_distribs2) + 1  # there might be an extra mask because of the scratch image
                gen_pix_distrib1 = tf.add_n([transformed_pix_distrib * mask
                                             for transformed_pix_distrib, mask in
                                             zip(transformed_pix_distribs2, masks)])

        with tf.variable_scope('state_pred'):
            gen_state = dense(state_action, state_dim)

        # outputs
        outputs = [gen_image, gen_state, masks, transformed_images, flow_map]
        if self.first_pix_distrib1 is not None:
            outputs.append(gen_pix_distrib1)
            outputs.append(transformed_pix_distribs1)
            if 'ndesig' in self.conf:
                outputs.append(gen_pix_distrib1)
                outputs.append(transformed_pix_distribs1)

        outputs = tuple(outputs)
        # states
        new_lstm_states = lstm_state0, lstm_state1, lstm_state2, lstm_state3, lstm_state4
        new_states = [new_lstm_states, time + 1, gen_image, gen_state]
        if self.first_pix_distrib1 is not None:
            new_states.append(gen_pix_distrib1)
            if 'ndesig' in self.conf:
                new_states.append(gen_pix_distrib2)
        new_states = tuple(new_states)
        return outputs, new_states


class Dynamic_Base_Model(object):
    def __init__(self,
                 conf = None,
                 images=None,
                 actions=None,
                 states=None,
                 pix_distrib=None,
                 pix_distrib2=None,
                 trafo_pix = True,
                 load_data = True,
                 inference = True,
                 ):

        self.iter_num = tf.placeholder(tf.float32, [])

        self.trafo_pix = trafo_pix
        if pix_distrib is not None:
            assert trafo_pix == True
            states = tf.concat([states, tf.zeros(
                [conf['batch_size'], conf['sequence_length'] - conf['context_frames'], conf['sdim']])], axis=1)
            pix_distrib = tf.concat([pix_distrib, tf.zeros(
                [conf['batch_size'], conf['sequence_length'] - conf['context_frames'], 64, 64, 1])], axis=1)
            if 'ndesig' in conf:
                pix_distrib2 = tf.concat([pix_distrib2, tf.zeros(
                    [conf['batch_size'], conf['sequence_length'] - conf['context_frames'], 64, 64, 1])], axis=1)

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
        print 'base model uses traincond', self.train_cond

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
                                                shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
                images = self.images_pl

                self.pix_distrib1_pl = tf.placeholder(tf.float32, name='states',
                                                      shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 1))
                pix_distrib = self.pix_distrib1_pl

                if 'ndesig' in conf:
                    self.pix_distrib2_pl = tf.placeholder(tf.float32, name='states',
                                                          shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 1))
                    pix_distrib2 = self.pix_distrib2_pl

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
        if pix_distrib is not None:
            pix_distrib = tf.split(axis=1, num_or_size_splits=pix_distrib.get_shape()[1], value=pix_distrib)
            pix_distrib = [tf.reshape(pix, [self.batch_size, 64, 64, 1]) for pix in pix_distrib]
        if pix_distrib2 is not None:
            pix_distrib2 = tf.split(axis=1, num_or_size_splits=pix_distrib2.get_shape()[1], value=pix_distrib2)
            pix_distrib2 = [tf.reshape(pix, [self.batch_size, 64,64, 1]) for pix in pix_distrib2]

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


        first_pix_distrib1 = None if pix_distrib is None else pix_distrib[0]
        first_pix_distrib2 = None if pix_distrib2 is None else pix_distrib2[0]


        cell = DNACell(conf,
                       [height, width, color_channels],
                       state_dim,
                       first_image=images[0],
                       first_pix_distrib1=first_pix_distrib1,
                       first_pix_distrib2=first_pix_distrib2,
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

        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=[sequence_length] * batch_size, dtype=tf.float32,
                                       swap_memory=True, time_major=True)

        (gen_images, gen_states, gen_masks, gen_transformed_images, gen_flow_map), other_outputs = outputs[:5], outputs[5:]
        self.gen_images = tf.unstack(gen_images, axis=0)
        self.gen_states = tf.unstack(gen_states, axis=0)
        self.gen_masks = list(zip(*[tf.unstack(gen_mask, axis=0) for gen_mask in gen_masks]))
        self.gen_transformed_images = list(
            zip(*[tf.unstack(gen_transformed_image, axis=0) for gen_transformed_image in gen_transformed_images]))
        self.gen_flow_map = tf.unstack(gen_flow_map, axis=0)
        other_outputs = list(other_outputs)

        if pix_distrib is not None:
            self.gen_distrib1 = other_outputs.pop(0)
            self.gen_distrib1 = tf.unstack(self.gen_distrib1, axis=0)
            self.gen_transformed_pixdistribs1 = other_outputs.pop(0)
            self.gen_transformed_pixdistribs1 = list(zip(
                *[tf.unstack(gen_transformed_pixdistrib, axis=0) for gen_transformed_pixdistrib in
                  self.gen_transformed_pixdistribs1]))
            if 'ndesig' in conf:
                self.gen_distrib2 = other_outputs.pop(0)
                self.gen_distrib2 = tf.unstack(self.gen_distrib2, axis=0)
                self.gen_transformed_pixdistribs2 = other_outputs.pop(0)
                self.gen_transformed_pixdistribs2 = list(zip(
                    *[tf.unstack(gen_transformed_pixdistrib, axis=0) for gen_transformed_pixdistrib in
                      self.gen_transformed_pixdistribs2]))
        assert not other_outputs

        self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
        if not inference:
            loss, summaries = self.build_loss()
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            self.summ_op = tf.summary.merge(summaries)

    def build_loss(self):

        summaries = []


        # L2 loss, PSNR for eval.
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

        return loss, summaries


    def visualize(self, sess):
        visualize(sess, self.conf, self)

    def visualize_diffmotions(self, sess):
        visualize_diffmotions(sess, self.conf, self)

    def compute_metric(self, sess, create_images):
        compute_metric(sess, self.conf, self, create_images)

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



def apply_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D or 6-D tensor of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    if len(kernels.get_shape()) == 4:
        outputs = apply_cdna_kernels(image, kernels, dilation_rate=dilation_rate)
    elif len(kernels.get_shape()) == 6:
        outputs = apply_dna_kernels(image, kernels, dilation_rate=dilation_rate)
    else:
        raise ValueError
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

def compute_flow_map(kernels, masks=None):
    """
    Args:
        kernels: A 4-D or 6-D tensor of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.
        masks: A 4-D tensor of shape
            `[batch, in_height, in_width, num_transformed_images]` or
            a list of `num_transformed_images` 3-D tensors, each of shape
            `[batch, in_height, in_width]`.

    Returns:
        A 4-D tensors of shape
            `[batch, in_height, in_width, 2]`.

    Coordinate convention: x axis goes from left to right and y axis goes
    from top to bottom.
    The flow indicates the relative location of where the pixel came from,
    e.g. a negative x indicates that the pixel moved to the right from the
    last frame to the current one, and a positive y indicates that the pixel
    moved up.
    """
    if masks is not None and isinstance(masks, (tuple, list)):
        masks = tf.concat(masks, axis=-1)
    if len(kernels.get_shape()) == 4:
        batch_size, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
        if masks is not None:
            _, height, width, _ = masks.get_shape().as_list()
        else:
            raise ValueError('Unable to infer the height and width of the image')
    elif len(kernels.get_shape()) == 6:
        batch_size, height, width, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    else:
        raise ValueError
    kernel_size = [kernel_height, kernel_width]

    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    range_x = (kernel_size[1] - 1) / 2.
    range_y = (kernel_size[0] - 1) / 2.
    x = tf.linspace(-range_x, range_x, kernel_size[1])
    y = tf.linspace(-range_y, range_y, kernel_size[0])
    xv, yv = tf.meshgrid(x, y)
    if len(kernels.get_shape()) == 4:
        # expand over batch and transformation dimensions
        xv_expanded = xv[None, :, :, None]
        yv_expanded = yv[None, :, :, None]
    elif len(kernels.get_shape()) == 6:
        # expand over batch, spatial and transformation dimensions
        xv_expanded = xv[None, None, None, :, :, None]
        yv_expanded = yv[None, None, None, :, :, None]
    else:
        raise ValueError
    vec_x = tf.reduce_sum(xv_expanded * kernels, axis=(-3, -2))
    vec_y = tf.reduce_sum(yv_expanded * kernels, axis=(-3, -2))
    if len(kernels.get_shape()) == 4:
        # replicate flows over the spatial dimensions
        vec_x = tf.tile(vec_x[:, None, None, :], [1, height, width, 1])
        vec_y = tf.tile(vec_y[:, None, None, :], [1, height, width, 1])
    flow_map = tf.stack([vec_x, vec_y], axis=-2)
    if masks is not None:
        flow_map = tf.reduce_sum(flow_map * masks[..., None, :], axis=-1)
    else:  # assume masks are uniform, so average the flows along the transformation dimension
        flow_map = tf.reduce_mean(flow_map, axis=-1)
    return flow_map



