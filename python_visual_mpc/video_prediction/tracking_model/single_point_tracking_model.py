import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.lstm_ops12 import basic_conv_lstm_cell
from python_visual_mpc.misc.zip_equal import zip_equal

import collections
import pickle
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
import pdb
from tensorflow.contrib.layers import layer_norm
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
from python_visual_mpc.video_prediction.basecls.utils.transformations import dna_transformation, cdna_transformation
from python_visual_mpc.video_prediction.basecls.prediction_model_basecls import Base_Prediction_Model
from python_visual_mpc.flow.trafo_based_flow.correction import compute_motion_vector_cdna, compute_motion_vector_dna

from python_visual_mpc.flow.trafo_based_flow.correction import Trafo_Flow
from python_visual_mpc.flow.descriptor_based_flow.descriptor_flow_model import Descriptor_Flow
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten

from python_visual_mpc.video_prediction.dynamic_rnn_model.dynamic_base_model import Dynamic_Base_Model


from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
class Single_Point_Tracking_Model(Dynamic_Base_Model):
    def __init__(self,
                conf = None,
                trafo_pix = True,
                load_data = True,
                build_loss = True):

        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self.conf = conf

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

            print('single point uses traincond', self.train_cond)
            images, actions, states = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                              lambda: [train_images, train_actions, train_states],
                                              lambda: [val_images, val_actions, val_states])

            if 'use_len' in conf:
                images, states, actions  = self.random_shift(images, states, actions)

        # sample random points to track
        self.init_points = self.sample_initpoints()
        self.start_pix_distrib = make_initial_pixdistrib(self.conf, self.init_points)

        Dynamic_Base_Model.__init__(self,
                                    conf = conf,
                                    images=images,
                                    actions=actions,
                                    states=states,
                                    pix_distrib=self.start_pix_distrib,
                                    trafo_pix = True,
                                    load_data = load_data,
                                    build_loss = False  #don't build the loss now.
                                    )

        # sample initpoints at locations where movement is large
        # self.sample_initpoints(self.gen_flow_map[0])

        self.layer_normalization = conf['normalization']
        if conf['normalization'] == 'ln':
            self.normalizer_fn = layer_norm
        elif conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        elif conf['normalization'] == 'none':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % self.layer_normalization)

        with tf.variable_scope('tracker'):
            self.build_tracker()

        if build_loss:
            self.build_loss()
            self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.summ_op = tf.summary.merge(self.summaries)

    def sample_initpoints(self, flow_map=None):
        """
        :param flow_map at first time step
        :return: batch of sample cooridnates (one for each batch example)
        """
        batch_size = self.conf['batch_size']

        if flow_map is not None:
            flow_magnitudes = tf.norm(flow_map)
            flow_magnitudes /= tf.reduce_sum(flow_magnitudes, axis=[1,2])
            flow_magnitudes = tf.reshape(flow_magnitudes, [batch_size, -1])
            log_prob = tf.log(flow_magnitudes)
        else:
            log_prob = tf.constant(np.ones([batch_size, 64**2])/64**2)

        coords = tf.multinomial(log_prob, 1)
        coords = unravel_ind(coords, [64, 64])
        return coords

    def build_tracker(self):
        """
            Build the tracking network
            :return:
        """
        self.descp = []
        self.descp.append(self.build_descriptor(self.images[0], self.start_pix_distrib[:,0]))

        self.target_descp = select_img_val(self.descp[0], self.init_points)
        self.target_pixel_val = select_img_val(self.images[0], self.init_points)

        self.track_distrib = []

        for t in range(self.conf['sequence_length']-1):
            self.descp.append(self.build_descriptor(self.images[t+1], self.gen_distrib[t + 1], reuse=True))
            self.track_distrib.append(self.get_distrib(self.descp[t], self.target_descp))

    def get_distrib(self, descp_field, target_descp):
        if self.conf['metric'] == 'inverse_euclidean':
            dist_fields = tf.reduce_sum(tf.square(descp_field-target_descp), 3)
            inverse_dist_fields = tf.div(1., dist_fields + 1e-5)
            #normed_dist_fields should correspond DNA-like trafo kernels
            track_distrib = inverse_dist_fields / (tf.reduce_sum(inverse_dist_fields, [3,4], keep_dims=True) + 1e-6)
            print('using inverse_euclidean')
        elif self.conf['metric'] == 'cosine':
            cos_dist = tf.reduce_sum(descp_field*target_descp, axis=3)/(tf.norm(target_descp, axis=3)+1e-5)/(tf.norm(descp_field, axis=3) +1e-5)
            cos_dist = tf.reshape(cos_dist, [self.conf['batch_size'],64**2])
            track_distrib = tf.nn.softmax(cos_dist*self.conf['softmax_temp'], 2)

            print('using cosine distance')
        else: raise NotImplementedError
        return tf.reshape(track_distrib, [self.conf['batch_size'], 64, 64, 1])


    def build_descriptor(self, image, gen_distrib1, reuse=None):
        with tf.variable_scope('', reuse=reuse):
            base_dim = 32

            gen_distrib1 = tf.tile(gen_distrib1, [1,1,1,3])
            inp = tf.concat([image, gen_distrib1], axis=3)

            with tf.variable_scope('h0'):
                h0 = conv_pool2d(inp, base_dim, kernel_size=(5, 5), strides=(2, 2))
                h0 = self.normalizer_fn(h0)
                h0 = tf.nn.relu(h0)
            with tf.variable_scope('h1'):
                h1 = conv_pool2d(h0, base_dim * 2, kernel_size = (3, 3), strides = (2, 2))
                h1 = self.normalizer_fn(h1)
                h1 = tf.nn.relu(h1)
            with tf.variable_scope('h2'):
                h2 = conv_pool2d(h1, base_dim * 4, kernel_size=(3, 3), strides=(2, 2))
                h2 = self.normalizer_fn(h2)
                h2 = tf.nn.relu(h2)
            with tf.variable_scope('h3'):
                h3 = upsample_conv2d(h2, base_dim * 2, kernel_size=(3, 3), strides=(2, 2))
                h3 = self.normalizer_fn(h3)
                h3 = tf.nn.relu(h3)
            with tf.variable_scope('h4'):
                h4 = upsample_conv2d(tf.concat([h3, h1], axis=-1), base_dim, kernel_size=(3, 3), strides=(2, 2))
                h4 = self.normalizer_fn(h4)
                h4 = tf.nn.relu(h4)
            with tf.variable_scope('h5'):
                h5 = upsample_conv2d(tf.concat([h4, h0], axis=-1), base_dim, kernel_size=(3, 3), strides=(2, 2))
                h5 = self.normalizer_fn(h5)
                h5 = tf.nn.relu(h5)
            with tf.variable_scope('h6_masks'):
                descp = conv2d(h5, self.conf['descp_size'], kernel_size=(3, 3), strides=(1, 1))
                descp = self.normalizer_fn(descp)
                descp = tf.nn.relu(descp)
        return descp


    def build_loss(self):

        loss = 0
        summaries = []
        for t in range(self.conf['sequence_length']-1):
            gen_dist = self.gen_distrib[t] / tf.reshape(tf.reduce_sum(self.gen_distrib[t], axis=[1, 2, 3]),
                                                        [16,1,1,1])
            if self.conf['prob_dist_measure'] == 'kl':
                pdist_loss = compute_kl(self.track_distrib[t], gen_dist)
            elif self.conf['prob_dist_measure'] == 'diff':
                pdist_loss = compute_kl(self.track_distrib[t], gen_dist)

            summaries.append(tf.summary.scalar('prob_dist_cost' + str(t), pdist_loss))
            loss += pdist_loss

        # descriptor coherence loss
        for t in range(self.conf['sequence_length'] - 1):
            track_coherence_loss = tf.reduce_mean(
                tf.norm(self.target_descp - self.descp[t], axis=3)*self.track_distrib[t])

            summaries.append(tf.summary.scalar('track_coherence_loss' + str(t), pdist_loss))
            loss += track_coherence_loss

        # per pixel difference loss
        for t in range(self.conf['sequence_length'] - 1):
            pixel_diff_loss = tf.reduce_mean(
                tf.norm(self.target_pixel_val - self.images[t], axis=3) * self.track_distrib[t])

            summaries.append(tf.summary.scalar('pixel_difference_loss' + str(t), pixel_diff_loss))
            loss += pixel_diff_loss

        loss = loss / np.float32(len(self.images) - self.conf['context_frames'])
        base_loss, base_summaries = Dynamic_Base_Model.build_loss(self)

        self.loss = loss = base_loss + loss
        self.summaries = base_summaries + summaries

    def visualize(self, sess, images, actions, states):
        print('-------------------------------------------------------------------')
        print('verify current settings!! ')
        for key in list(self.conf.keys()):
            print(key, ': ', self.conf[key])
        print('-------------------------------------------------------------------')

        import re
        itr_vis = re.match('.*?([0-9]+)$', self.conf['visualize']).group(1)

        self.saver.restore(sess, self.conf['visualize'])
        print('restore done.')

        feed_dict = {self.lr: 0.0,
                     self.iter_num: 0,
                     self.train_cond: 1}

        file_path = self.conf['output_dir']

        ground_truth, gen_images, gen_masks, pred_flow, track_flow = self.sess.run([self.images,
                                                                               self.gen_images,
                                                                               self.gen_masks,
                                                                               self.prediction_flow,
                                                                               self.tracking_flow01
                                                                               ],
                                                                              feed_dict)
        dict = collections.OrderedDict()
        dict['iternum'] = itr_vis
        dict['ground_truth'] = ground_truth
        dict['gen_images'] = gen_images
        dict['prediction_flow'] = pred_flow
        dict['tracking_flow'] = track_flow
        # dict['gen_masks'] = gen_masks

        pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
        print('written files to:' + file_path)

        v = Visualizer_tkinter(dict, numex=10, append_masks=True, filepath=self.conf['output_dir'])
        v.build_figure()


def unravel_ind(argmax, shape):
    output_list = []
    output_list.append(argmax / (shape[1]))
    output_list.append(argmax % shape[1])
    return tf.cast(tf.concat(output_list, 1), dtype=tf.int32)

def make_initial_pixdistrib(conf, desig_pix):

    flat_ind = []
    for b in range(conf['batch_size']):
        r = desig_pix[b,0]
        c = desig_pix[b,1]
        flat_ind.append(r * 64+ c)
    flat_ind = tf.stack(flat_ind, axis=0)
    one_hot = tf.one_hot(flat_ind, depth=64 ** 2, axis=-1)
    one_hot = tf.reshape(one_hot, [conf['batch_size'], 64, 64, 1])

    return tf.stack([one_hot, one_hot], axis=1)

def select_img_val(descp, pos):
    """
    :param descp: shape [batch_size, 64,64, len_descp]
    :return:
    """
    batch_size = int(descp.get_shape())
    indices = []
    for b in range(batch_size):
        ind = tf.concat([tf.constant([b]), pos[b][0], pos[b][1]], axis=0)
        indices.append(ind)
    indices = tf.stack(indices, axis=0)
    target_descp = tf.gather_nd(descp, indices)

    return target_descp

def compute_kl(track_distrib, pred_distrib):
    """
    :param track_distrib: p
    :param pred_distrib: q

    D_kl = sum_r( sum_c (p * log(p/q) ))
    :return:
    """
    return tf.reduce_sum(track_distrib * tf.log(track_distrib/(pred_distrib + 1e-7)), axis=[1,2])

def mean_squared_error(true, pred):
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))