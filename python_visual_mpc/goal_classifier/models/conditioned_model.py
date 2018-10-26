from .base_model import BaseGoalClassifier
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
import h5py
from tensorflow.contrib.training import HParams

default_dict = {
    'meta_batch_size': 7,
    'meta_lr': 0.001,
    'update_batch_size': 1,
    'update_lr': 1.0,
    'num_updates': 1,
    'grad_clip': True,
    'clip_min': -80.0,
    'clip_max': 80.0,
    'stop_grad': True,
    'im_height': 48,
    'im_width': 64,
    'norm': 'layer_norm',
    'num_conv_layers': 3,
    'num_filters': 16,
    'num_fc_layers': 2,
    'hidden_dim': 40,
    'fc_bt': False,
    'bt_dim': 0,
    'fp': False,
    'vgg_path': '',
    'resume': True,
    'train': True,
    'test_iter': -1,
    'test_set': False,
    'train_update_batch_size': 5,
    'train_update_lr': -1
}

FLAGS = HParams(**default_dict)

class ConditionedGoalClassifier(BaseGoalClassifier):
    def __init__(self, _):
        pass

    def build(self, global_step=None):
        self.model = MSC(test_num_updates=1)
        self.model.construct_model(prefix='metaval_')

    def restore(self, sess, pretrained_model):
        var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if var.name[:5] == 'model']
        # var_list.extend([self.model.weights['conv1_w'], self.model.weights['conv1_b']])
        saver = tf.train.Saver(var_list, max_to_keep=0)
        tf.global_variables_initializer().run()
        saver.restore(sess, pretrained_model)

    def score(self, sess, images, goal_images):
        labels = np.zeros((1, FLAGS.update_batch_size, 2))
        labels[:, :, 1] = 1
        feed_dict = {self.model.inputa: np.reshape(goal_images[:, 0], [1, FLAGS.update_batch_size, FLAGS.im_height, FLAGS.im_width, 3]),
                     self.model.inputb: np.reshape(images[:, 0], [1, -1, FLAGS.im_height, FLAGS.im_width, 3]),
                     self.model.labela: labels,
                     self.model.labelb: 0}
        probs = sess.run(tf.nn.softmax(self.model.metaval_outputbs[-1][0]), feed_dict=feed_dict)
        return -np.log(probs[:, 1])


class MSC(object):
    def __init__(self, test_num_updates=1):
        """ must call construct_model() after initializing! """
        self.dim_output = 2
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = test_num_updates

        self.loss_func = xent

        self.im_height = FLAGS.im_height
        self.im_width = FLAGS.im_width
        self.channels = 3
        self.num_conv_layers = FLAGS.num_conv_layers

        self.num_fc_layers = FLAGS.num_fc_layers
        self.dim_conv_hidden = FLAGS.num_filters
        self.dim_fc_hidden = FLAGS.hidden_dim

        if FLAGS.fp:
            self.conv_out_size = FLAGS.num_filters * 2
        else:
            self.conv_out_size = int(np.ceil(self.im_height / 2 ** self.num_conv_layers)) * int(np.ceil(self.im_width / 2 ** self.num_conv_layers)) * FLAGS.num_filters

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela) / FLAGS.update_batch_size

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                if FLAGS.grad_clip:
                    for key in gradients.keys():
                        gradients[key] = tf.clip_by_value(gradients[key], FLAGS.clip_min, FLAGS.clip_max)
                if FLAGS.vgg_path:
                    gradients['conv1_w'] = tf.zeros_like(gradients['conv1_w'])
                    gradients['conv1_b'] = tf.zeros_like(gradients['conv1_b'])
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb) / FLAGS.train_update_batch_size)

                for j in range(num_updates - 1):
                    output = self.forward(inputa, fast_weights, reuse=True)
                    loss = self.loss_func(output, labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    if FLAGS.grad_clip:
                        for key in gradients.keys():
                            gradients[key] = tf.clip_by_value(gradients[key], FLAGS.clip_min, FLAGS.clip_max)
                    if FLAGS.vgg_path:
                        gradients['conv1_w'] = tf.zeros_like(gradients['conv1_w'])
                        gradients['conv1_b'] = tf.zeros_like(gradients['conv1_b'])
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb) / FLAGS.train_update_batch_size)

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                for j in range(num_updates):
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.outputas, self.outputbs = outputas, outputbs
            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.metaval_outputbs = outputbs

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss (a)', total_loss1)
        tf.summary.scalar(prefix+'Pre-update accuracy (a)', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss (b), step ' + str(j+1), total_losses2[j])
            tf.summary.scalar(prefix+'Post-update accuracy (b), step ' + str(j+1), total_accuracies2[j])

    def construct_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3

        for i in range(self.num_conv_layers):
            dim_input = dim_output = self.dim_conv_hidden

            if i == 0:
                dim_input = self.channels

            if FLAGS.vgg_path and i == 1:
                dim_input = conv_weight.shape[-1]

            if FLAGS.vgg_path and i == 0:
                print("using VGG weights")
                dim_output = 64
                weights['conv%d_w' % (i+1)] = tf.get_variable('conv%d_w' % (i+1), [k, k, dim_input, dim_output], initializer=conv_initializer, dtype=dtype, trainable=False)
                weights['conv%d_b' % (i+1)] = tf.Variable(tf.zeros([dim_output]), trainable=False)
                vgg_weights = h5py.File(FLAGS.vgg_path, 'r')
                conv_weight = vgg_weights['block1_conv%d' % (i+1)]['block1_conv%d_W_1:0' % (i+1)][...]
                conv_bias = vgg_weights['block1_conv%d' % (i+1)]['block1_conv%d_b_1:0' % (i+1)][...]
                weights['conv%d_w' % (i+1)].assign(conv_weight)
                weights['conv%d_b' % (i+1)].assign(conv_bias)
            else:
                weights['conv%d_w' % (i+1)] = tf.get_variable('conv%d_w' % (i+1), [k, k, dim_input, dim_output], initializer=conv_initializer, dtype=dtype)
                weights['conv%d_b' % (i+1)] = tf.Variable(tf.zeros([dim_output]))

        if FLAGS.fc_bt:
            weights['context'] = tf.Variable(tf.zeros([FLAGS.bt_dim]))

        for i in range(self.num_fc_layers):
            dim_input = dim_output = self.dim_fc_hidden

            if i == 0:
                dim_input = self.conv_out_size
                if FLAGS.fc_bt:
                    dim_input += FLAGS.bt_dim

            if i == self.num_fc_layers - 1:
                dim_output = self.dim_output

            weights['fc%d_w' % (i+1)] = tf.Variable(tf.truncated_normal([dim_input, dim_output], stddev=0.01))
            weights['fc%d_b' % (i+1)] = tf.Variable(tf.zeros([dim_output]))

        return weights

    def forward(self, inp, weights, reuse=False, scope=''):
        if FLAGS.vgg_path:
            inp = inp * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))
            inp = inp[:, :, :, ::-1]

        if FLAGS.fc_bt:
            flatten_image = tf.reshape(inp, [-1, self.im_height * self.im_width * self.channels])
            context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(FLAGS.bt_dim)))
            context += weights['context']

        conv_layer = tf.reshape(inp, [-1, self.im_height, self.im_width, self.channels])
        for i in range(self.num_conv_layers):
            conv_layer = conv_block(conv_layer, weights['conv%d_w' % (i+1)], weights['conv%d_b' % (i+1)], reuse, scope + 'conv%d' % (i+1))

        if FLAGS.fp:
            batch_size, num_rows, num_cols, num_fp = conv_layer.get_shape()

            num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]

            x_map = np.empty([num_rows, num_cols], np.float32)
            y_map = np.empty([num_rows, num_cols], np.float32)

            for i in range(num_rows):
                for j in range(num_cols):
                    x_map[i, j] = (i - num_rows / 2.0) / num_rows
                    y_map[i, j] = (j - num_cols / 2.0) / num_cols

            x_map = tf.convert_to_tensor(x_map)
            y_map = tf.convert_to_tensor(y_map)

            x_map = tf.reshape(x_map, [num_rows * num_cols])
            y_map = tf.reshape(y_map, [num_rows * num_cols])

            features = tf.reshape(tf.transpose(conv_layer, [0, 3, 1, 2]), [-1, num_rows * num_cols])
            softmax = tf.nn.softmax(features)

            fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
            fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

            conv_out_flat = tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])
        else:
            conv_out_flat = tf.reshape(conv_layer, [-1, self.conv_out_size])

        fc_layer = tf.reshape(conv_out_flat, [-1, self.conv_out_size])

        if FLAGS.fc_bt:
            fc_layer = tf.concat([fc_layer, context], 1)
        for i in range(self.num_fc_layers - 1):
            fc_layer = normalize(tf.matmul(fc_layer, weights['fc%d_w' % (i+1)]) + weights['fc%d_b' % (i+1)], tf.nn.relu, reuse, scope + 'fc%d' % (i+1))

        logits = tf.matmul(fc_layer, weights['fc%d_w' % (self.num_fc_layers)]) + weights['fc%d_b' % (self.num_fc_layers)]

        return logits


def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', use_stride=True, residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if use_stride:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size