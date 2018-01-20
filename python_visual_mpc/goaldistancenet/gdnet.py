import tensorflow as tf
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cPickle
import sys
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn
import matplotlib.gridspec as gridspec

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers

def get_coords(img_shape):
    y = tf.cast(tf.range(img_shape[1]), tf.float32)
    x = tf.cast(tf.range(img_shape[2]), tf.float32)
    batch_size = img_shape[0]

    X,Y = tf.meshgrid(x,y)
    coords = tf.expand_dims(tf.stack((X, Y), axis=2), axis=0)
    coords = tf.tile(coords, [batch_size, 1,1,1])
    return coords

def resample_layer(src_img, warp_pts, name="tgt_img"):
    with tf.variable_scope(name):
        return tf.contrib.resampler.resampler(src_img, warp_pts)

def warp_pts_layer(flow_field, name="warp_pts"):
    with tf.variable_scope(name):
        img_shape = flow_field.get_shape().as_list()
        return flow_field + get_coords(img_shape)

def mean_squared_error(true, pred):
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))

class GoalDistanceNet(object):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True
                 ):

        self.layer_normalization = conf['normalization']
        if conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        elif conf['normalization'] == 'None':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % self.layer_normalization)

        self.conf = conf
        self.iter_num = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        if load_data:

            train_dict = build_tfrecord_fn(conf, training=True)
            val_dict = build_tfrecord_fn(conf, training=False)
            dict = tf.cond(self.train_cond > 0,
                             # if 1 use trainigbatch else validation batch
                             lambda: train_dict,
                             lambda: val_dict)
            self.images = dict['images']
            self.states = dict['endeffector_pos']
            self.actions = dict['actions']

            if 'vidpred_data' in conf:  # register predicted video to real
                self.pred_images = tf.squeeze(dict['gen_images'])
                self.pred_states = tf.squeeze(dict['gen_states'])

            self.I0, self.I1 = self.sel_images()

        else:
            if 'orig_size' in self.conf:
                self.img_height = self.conf['orig_size'][0]
                self.img_width = self.conf['orig_size'][1]
            else:
                self.img_height = conf['row_end'] - conf['row_start']
                self.img_width = 64
            self.I0 = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.I1 = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))
        self.build()

        if build_loss:
            self.build_loss()


    def sel_images(self):
        sequence_length = self.conf['sequence_length']
        delta_t = tf.cast(tf.ceil(sequence_length * (self.iter_num + 1) / 100), dtype=tf.int32)  #################
        delta_t = tf.clip_by_value(delta_t, 1, sequence_length-1)

        self.tstart = tf.random_uniform([1], 0, sequence_length - delta_t, dtype=tf.int32)

        self.tend = self.tstart + tf.random_uniform([1], tf.ones([], dtype=tf.int32), delta_t + 1, dtype=tf.int32)

        begin = tf.stack([0, tf.squeeze(self.tstart), 0, 0, 0],0)

        if 'vidpred_data' in self.conf:
            I0 = tf.squeeze(tf.slice(self.pred_images, begin, [-1, 1, -1, -1, -1]))
        else:
            I0 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        begin = tf.stack([0, tf.squeeze(self.tend), 0, 0, 0], 0)
        I1 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        return I0, I1


    # def conv_relu_block(self, input, channel_mult, k=3, strides=2, upsmp=False):
    #     if not upsmp:
    #         h = conv_pool2d(input, self.conf['basedim'] * channel_mult, kernel_size=(k, k),
    #                         strides=(strides, strides))  # 20x32x3
    #     else:
    #         h = upsample_conv2d(input, self.conf['basedim'] * channel_mult, kernel_size=(k, k),
    #                         strides=(strides, strides))  # 20x32x3
    #
    #     h = self.normalizer_fn(h)
    #     h = tf.nn.relu(h)
    #     return h

    def conv_relu_block(self, input, out_ch, k=3, upsmp=False):
        h = slim.layers.conv2d(  # 32x32x64
            input,
            out_ch, [k, k],
            stride=1)

        if upsmp:
            mult = 2
        else: mult = 0.5
        imsize = np.array(h.get_shape().as_list()[1:3])*mult

        h = tf.image.resize_images(h, imsize, method=tf.image.ResizeMethod.BILINEAR)

        return h


    def build(self):
        I0_I1 = tf.concat([self.I0, self.I1], axis=3)

        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(I0_I1, out_ch=32)  #24x32x3

        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64)  #12x16x3

        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, out_ch=128)  #6x8x3

        with tf.variable_scope('h4'):
            h4 = self.conv_relu_block(h3, out_ch=64, upsmp=True)  #12x16x3

        with tf.variable_scope('h5'):
            h5 = self.conv_relu_block(h4, out_ch=32, upsmp=True)  #24x32x3

        with tf.variable_scope('h6'):
            h6 = self.conv_relu_block(h5, out_ch=16, upsmp=True)  #48x64x3

        with tf.variable_scope('h7'):
            self.flow_field = slim.layers.conv2d(  # 128x128xdesc_length
                h6,  2, [5, 5], stride=1, activation_fn=None)

        self.warp_pts = warp_pts_layer(self.flow_field)
        self.gen_image = resample_layer(self.I0, self.warp_pts)

    def build_loss(self):

        summaries = []
        self.loss = mean_squared_error(self.gen_image, self.I1)
        summaries.append(tf.summary.scalar('train_recon_cost', self.loss))
        self.train_summ_op = tf.summary.merge(summaries)

        summaries.append(tf.summary.scalar('val_recon_cost', self.loss))
        self.val_summ_op = tf.summary.merge(summaries)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def visualize(self, sess):

        dict = build_tfrecord_fn(self.conf)
        images, pred_images = sess.run([dict['images'], dict['gen_images']])

        pred_images = np.squeeze(pred_images)

        num_examples = self.conf['batch_size']
        I1 = images[:, -1]

        outputs = []
        I0_t_reals = []
        I0_ts = []
        flow_mags = []
        warpscores = []

        for t in range(14):
            if 'vidpred_data' in self.conf:
                I0_t = pred_images[:, t]
                I0_t_real = images[:, t]

                I0_t_reals.append(I0_t_real)
            else:
                I0_t = images[:, t]

            I0_ts.append(I0_t)

            [output, flow] = sess.run([self.gen_image, self.flow_field], {self.I0_pl:I0_t, self.I1_pl: I1})

            outputs.append(output)

            flow_mag = np.linalg.norm(flow, axis=3)
            flow_mag = np.split(np.squeeze(flow_mag), num_examples, axis=0)
            cmap = plt.cm.get_cmap('jet')
            flow_mag_ = []

            warpscores.append(np.mean(np.mean(flow_mag, axis=1),axis=1))

            for b in range(num_examples):
                f = flow_mag[b]/(np.max(flow_mag[b]) + 1e-6)
                f = cmap(f)[:, :, :3]
                flow_mag_.append(f)

            flow_mags.append(flow_mag)

        dict = {
            'I0_t_real':I0_t_reals,
            'I0_t':I0_ts,
            'flow_mags':flow_mags,
            'outputs':outputs,
            'warpscores':warpscores}

        cPickle.dump(dict, open(self.conf['output_dir'] + '/data.pkl', 'wb'))
        make_plots(dict)

        #     im_height = output.shape[1]
        #     im_width = output.shape[2]
        #     warped_column = np.squeeze(np.split(output, num_examples, axis=0))
        #
        #     I0_t = np.squeeze(np.split(I0_t, num_examples, axis=0))
        #
        #     if 'vidpred_data' in self.conf:
        #         I0_t_real = np.squeeze(np.split(I0_t_real, num_examples, axis=0))
        #         I0_t = np.squeeze(np.split(I0_t, num_examples, axis=0))
        #
        #         warped_column = [np.concatenate([inp_real, inp, warp, fmag], axis=0) for inp_real, inp, warp, fmag in
        #                          zip(I0_t_real, I0_t, warped_column, flow_mag)]
        #     else:
        #         warped_column = [np.concatenate([inp, warp, fmag], axis=0) for inp, warp, fmag in
        #                          zip(I0_t, warped_column, flow_mag)]
        #
        #     warped_column = np.concatenate(warped_column, axis=0)
        #
        #     warped_images.append(warped_column)
        #
        # warped_images = np.concatenate(warped_images, axis=1)
        #
        # dir = self.conf['output_dir']
        # Image.fromarray((warped_images*255.).astype(np.uint8)).save(dir + '/warped.png')
        #
        # I1 = I1.reshape(num_examples*im_height, im_width,3)
        # Image.fromarray((I1 * 255.).astype(np.uint8)).save(dir + '/finalimage.png')
        #
        # sys.exit('complete!')


def make_plots(conf, dict=None, filename = None):
    if dict == None:
        dict = cPickle.load(open(filename))

    print 'loaded'

    I0_t_reals = dict['I0_t_real']
    I0_ts =dict['I0_t']
    flow_mags =dict['flow_mags']
    outputs =dict['outputs']
    warpscores =dict['warpscores']

    # num_exp = I0_t_reals[0].shape[0]
    num_ex = 3
    start_ex = 10

    num_rows = num_ex*4
    num_cols = len(I0_t_reals)

    # plt.figure(figsize=(num_rows, num_cols))
    # gs1 = gridspec.GridSpec(num_rows, num_cols)
    # gs1.update(wspace=0.025, hspace=0.05)

    width_per_ex = 0.9

    standard_size = np.array([width_per_ex * num_cols, num_rows * 1.0])  ### 1.5
    figsize = (standard_size).astype(np.int)

    f, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    for col in range(num_cols):
        row = 0
        for ex in range(start_ex, start_ex + num_ex, 1):
            print 'ex{}'.format(ex)

            axarr[row, col].imshow(I0_t_reals[col][ex], interpolation='none')
            axarr[row, col].axis('off')
            row += 1

            axarr[row, col].imshow(I0_ts[col][ex], interpolation='none')
            axarr[row, col].axis('off')

            row += 1
            axarr[row, col].set_title('{:10.3f}'.format(warpscores[col][ex]), fontsize=5)
            axarr[row, col].imshow(flow_mags[col][ex], interpolation='none')
            axarr[row, col].axis('off')
            row += 1

            axarr[row, col].imshow(outputs[col][ex], interpolation='none')
            axarr[row, col].axis('off')
            row += 1

    # plt.axis('off')
    f.subplots_adjust(wspace=0, hspace=0.3)

    # f.subplots_adjust(vspace=0.1)
    # plt.show()
    plt.savefig(conf['output_dir']+'/warp_costs.png')


def calc_warpscores(flow_field):
    return np.sum(np.linalg.norm(flow_field, axis=3), axis=[2, 3])

def draw_text(img, float):
    img = (img*255.).astype(np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 5)
    draw.text((0, 0), "{}".format(float), (255, 255, 0), font=font)
    img = np.asarray(img).astype(np.float32)/255.

    return img


if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/vidpred_data/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')
