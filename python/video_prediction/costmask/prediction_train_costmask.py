import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import pdb

import imp

from video_prediction.utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from video_prediction.utils_vpred.create_gif import *

from video_prediction.read_tf_record import build_tfrecord_input
import makegifs

from datetime import datetime
from prediction_model_costmask import construct_model

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable, -1 uses cpu')
    flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')

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


def mujoco_to_imagespace_tf(mujoco_coord, numpix = 64):
    """
    convert form Mujoco-Coord to numpix x numpix image space:
    :param numpix: number of pixels of square image
    :param mujoco_coord: batch_size x 2
    :return: pixel_coord: batch_size x 2
    """
    mujoco_coord = tf.cast(mujoco_coord, tf.float32)

    viewer_distance = .75  # distance from camera to the viewing plane
    window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
    pixelheight = window_height / numpix  # height of one pixel
    middle_pixel = numpix / 2
    r = -tf.slice(mujoco_coord,[0,1], [-1,1])
    c =  tf.slice(mujoco_coord,[0,0], [-1,1])
    pixel_coord = tf.concat(1, [r,c])/pixelheight
    pixel_coord += middle_pixel
    pixel_coord = tf.round(pixel_coord)

    return pixel_coord

def mean_squared_error_costmask(true, pred, current_rpos, conf):
    retina_size = conf['retina_size']
    half_rh = retina_size / 2  # half retina height

    true_retinas = []
    pred_retinas = []
    for b in range(conf['batch_size']):
        begin = tf.squeeze(tf.slice(current_rpos, [b, 0], [1, -1]) - tf.constant([half_rh], dtype= tf.int32))
        begin = tf.concat(0, [tf.zeros([1], dtype=tf.int32), begin, tf.zeros([1], dtype=tf.int32)])
        len = tf.constant([-1, retina_size, retina_size, -1], dtype=tf.int32)
        b_true = tf.slice(true, [b, 0, 0, 0], [1, -1, -1, -1])
        b_pred = tf.slice(pred, [b, 0, 0, 0], [1, -1, -1, -1])

        true_retinas.append(tf.slice(b_true, begin, len))
        pred_retinas.append(tf.slice(b_pred, begin, len))

    true_retinas = tf.concat(0, true_retinas)
    pred_retinas = tf.concat(0, pred_retinas)
    cost = tf.reduce_sum(tf.square(true_retinas - pred_retinas)) / tf.to_float(tf.size(pred_retinas))
    return cost, true_retinas, pred_retinas


class Model(object):
    def __init__(self,
                 conf,
                 images=None,
                 actions=None,
                 states=None,
                 init_obj_pose= None,
                 reuse_scope=None,
                 pix_distrib=None):



        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        # Split into timesteps.
        if actions != None:
            actions = tf.split(1, actions.get_shape()[1], actions)
            actions = [tf.squeeze(act) for act in actions]
        if states != None:
            states = tf.split(1, states.get_shape()[1], states)
            states = [tf.squeeze(st) for st in states]

        images = tf.split(1, images.get_shape()[1], images)
        images = [tf.squeeze(img) for img in images]

        if pix_distrib != None:
            pix_distrib = tf.split(1, pix_distrib.get_shape()[1], pix_distrib)
            pix_distrib = [tf.squeeze(pix) for pix in pix_distrib]

        if reuse_scope is None:
            gen_images, gen_states, gen_masks, gen_distrib, retpos = construct_model(
                images,
                actions,
                states,
                init_obj_pose,
                iter_num=self.iter_num,
                k=conf['schedsamp_k'],
                use_state=conf['use_state'],
                num_masks=conf['num_masks'],
                cdna=conf['model'] == 'CDNA',
                dna=conf['model'] == 'DNA',
                stp=conf['model'] == 'STP',
                context_frames=conf['context_frames'],
                pix_distributions= pix_distrib,
                conf=conf)
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images, gen_states, gen_masks, gen_distrib, retpos = construct_model(
                    images,
                    actions,
                    states,
                    init_obj_pose,
                    iter_num=self.iter_num,
                    k=conf['schedsamp_k'],
                    use_state=conf['use_state'],
                    num_masks=conf['num_masks'],
                    cdna=conf['model'] == 'CDNA',
                    dna=conf['model'] == 'DNA',
                    stp=conf['model'] == 'STP',
                    context_frames=conf['context_frames'],
                    conf= conf)

        # L2 loss, PSNR for eval.
        true_fft_list, pred_fft_list = [], []

        costmasklist, true_retinas, pred_retinas = [], [], []

        loss, psnr_all = 0.0, 0.0

        self.fft_weights = tf.placeholder(tf.float32, [64, 64])


        for i, x, gx, p in zip(
                range(len(gen_images)), images[conf['context_frames']:],
                gen_images[conf['context_frames'] - 1:], retpos[conf['context_frames'] - 1:]):
            recon_cost_mse, true_ret, pred_ret = mean_squared_error_costmask(x, gx, p, conf)
            true_retinas.append(true_ret)
            pred_retinas.append(pred_ret)

            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(
                tf.scalar_summary(prefix + '_recon_cost' + str(i), recon_cost_mse))
            summaries.append(tf.scalar_summary(prefix + '_psnr' + str(i), psnr_i))

            recon_cost = recon_cost_mse

            loss += recon_cost

        for i, state, gen_state in zip(
                range(len(gen_states)), states[conf['context_frames']:],
                gen_states[conf['context_frames'] - 1:]):
            state_cost = mean_squared_error(state, gen_state) * 1e-4 * conf['use_state']
            summaries.append(
                tf.scalar_summary(prefix + '_state_cost' + str(i), state_cost))
            loss += state_cost
        summaries.append(tf.scalar_summary(prefix + '_psnr_all', psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])

        summaries.append(tf.scalar_summary(prefix + '_loss', loss))

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)

        self.true_fft = true_fft_list
        self.pred_fft = pred_fft_list
        self.gen_images= gen_images
        self.gen_masks = gen_masks
        self.gen_distrib = gen_distrib
        self.gen_states = gen_states

        self.costmasklist = costmasklist
        self.true_retinas = true_retinas
        self.pred_retinas = pred_retinas
        self.retpos_list = retpos


def main(unused_argv, conf_script= None):

    if FLAGS.device ==-1:   # using cpu!
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tfconfig = None
    else:
        print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        tfconfig = tf.ConfigProto(gpu_options=gpu_options)

        from tensorflow.python.client import device_lib
        print device_lib.list_local_devices()

    if conf_script == None: conf_file = FLAGS.hyper
    else: conf_file = conf_script

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration
    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf['schedsamp_k'] = -1  # don't feed ground truth
        conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
        conf['visualize'] = conf['output_dir'] + '/' + FLAGS.visualize
        conf['event_log_dir'] = '/tmp'
        conf['visual_file'] = conf['data_dir'] + '/traj_0_to_255.tfrecords'

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'Constructing models and inputs.'
    with tf.variable_scope('model', reuse=None) as training_scope:

        images, actions, states, poses, max_move = build_tfrecord_input(conf, training=True)
        if 'max_move_pos' in conf:
            init_pos = tf.squeeze(tf.slice(tf.squeeze(max_move), [0, 0, 0], [-1, 1, 2]))
            print 'using max move pos'
        else:
            init_pos = tf.squeeze(tf.slice(tf.squeeze(poses), [0,0,0], [-1, 1, 2]))
        model = Model(conf, images, actions, states, init_pos)

    with tf.variable_scope('val_model', reuse=None):
        val_images, val_actions, val_states, val_poses, val_max_move = build_tfrecord_input(conf, training=False)
        if 'max_move_pos' in conf:
            init_val_pos = tf.squeeze(tf.slice(tf.squeeze(val_max_move), [0, 0, 0], [-1, 1, 2]))
            print 'using max move pos'
        else:
            init_val_pos = tf.squeeze(tf.slice(tf.squeeze(val_poses), [0, 0, 0], [-1, 1, 2]))
        val_model = Model(conf, val_images, val_actions, val_states, init_val_pos, training_scope)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)


    # Make training session.
    sess = tf.InteractiveSession(config= tfconfig)
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    if conf['visualize']:
        saver.restore(sess, conf['visualize'])

        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'vis',
                     val_model.iter_num: 50000}
        file_path = conf['output_dir']

        gen_images, images, mask_list, true_ret, pred_ret, gen_distrib, retpos = sess.run([
                                                        val_model.gen_images,
                                                        val_images,
                                                        val_model.gen_masks,
                                                        val_model.true_retinas,
                                                        val_model.pred_retinas,
                                                        val_model.gen_distrib,
                                                        val_model.retpos_list
                                                        ],
                                                       feed_dict)

        dict_ = {}
        dict_['gen_images'] = gen_images
        dict_['true_ret'] = true_ret
        dict_['pred_ret'] = pred_ret
        dict_['images'] = images
        dict_['gen_distrib'] = gen_distrib
        dict_['retpos'] = retpos

        cPickle.dump(dict_, open(file_path + '/dict_.pkl', 'wb'))
        print 'written files to:' + file_path

        makegifs.comp_pix_distrib(conf, conf['output_dir'], examples=10)
        return

    itr_0 =0

    if FLAGS.pretrained != None:
        conf['pretrained_model'] = FLAGS.pretrained

        saver.restore(sess, conf['pretrained_model'])
        # resume training at iteration step of the loaded model:
        import re
        itr_0 = re.match('.*?([0-9]+)$', conf['pretrained_model']).group(1)
        itr_0 = int(itr_0)
        print 'resuming training at iteration:  ', itr_0

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter = []
    # Run training.
    fft_weights = calc_fft_weight()

    ###### debugging
    # from PIL import Image
    # itr = 0
    # feed_dict = {model.prefix: 'train',
    #              model.iter_num: np.float32(itr),
    #              model.lr: conf['learning_rate'],
    #              }
    # true_retina, retpos, gen_distrib, initpos, imdata = sess.run([model.true_retinas, model.retpos_list, model.gen_distrib, init_pos, images ],
    #                                 feed_dict)
    # print 'retina pos:'
    # for b in range(4):
    #     Image.fromarray((true_retina[0][b] * 255).astype(np.uint8)).show()
    #     Image.fromarray((np.squeeze(gen_distrib[0][b]) * 255).astype(np.uint8)).show()
    #     Image.fromarray((imdata[b][0] * 255).astype(np.uint8)).show()
    #
    #     print 'retpos', retpos[b]
    #     print 'initpos', init_pos[0]

    ###### end debugging

    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()
        # Generate new batch of data_files.
        feed_dict = {model.prefix: 'train',
                     model.iter_num: np.float32(itr),
                     model.lr: conf['learning_rate'],
                     model.fft_weights: fft_weights}
        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        # Print info: iteration #, cost.
        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {val_model.lr: 0.0,
                         val_model.prefix: 'val',
                         val_model.iter_num: np.float32(itr),
                         val_model.fft_weights: fft_weights}
            _, val_summary_str = sess.run([val_model.train_op, val_model.summ_op],
                                          feed_dict)
            summary_writer.add_summary(val_summary_str, itr)


        if (itr) % SAVE_INTERVAL == 2:
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


def calc_fft_weight():

    weight = np.zeros((64,64))
    for row in range(64):
        for col in range(64):
            p = np.array([row,col])
            c = np.array([31,31])
            weight[row, col] = np.linalg.norm(p -c)**2

    weight /= np.max(weight)
    return weight

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
