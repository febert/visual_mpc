import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import copy

from utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import utils_vpred.create_gif

from read_tf_record import build_tfrecord_input

from utils_vpred.skip_example import skip_example

from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

FLAGS = flags.FLAGS
flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
flags.DEFINE_integer('device', None, 'the gpu number to start with')

flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')


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


def mean_squared_error(true, pred, example_wise=False):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    if not example_wise:
        return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))
    else:
        if len(true.get_shape()) == 4:
            return tf.reduce_sum(tf.square(true - pred), reduction_indices=[1, 2, 3]) \
                   / tf.to_float(tf.size(pred))
        elif len(true.get_shape()) == 2:
            return tf.reduce_sum(tf.square(true - pred), reduction_indices=[1]) \
                   / tf.to_float(tf.size(pred))

class Model(object):
    def __init__(self,
                 conf,
                 input_data=None,
                 reuse_scope=None,
                 pix_distrib=None):

        from prediction_model_stochastic_search import construct_model

        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        if input_data == None:
            self.images = images = tf.placeholder(tf.float32, name='images',
                                                  shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
            self.actions = actions = tf.placeholder(tf.float32, name='actions',
                                                    shape=(conf['batch_size'], conf['sequence_length'], 2))
            self.states = states = tf.placeholder(tf.float32, name='states',
                                                  shape=(conf['batch_size'], conf['sequence_length'], 4))
        else:
            images, states, actions = input_data

        bsize = images.get_shape()[0]
        self.noise = tf.placeholder(tf.float32, name='noise',
                                    shape=(bsize, conf['sequence_length'], conf['noise_dim']))

        # Split into timesteps.
        noise = tf.split(1, self.noise.get_shape()[1], self.noise)
        noise = [tf.squeeze(n) for n in noise]
        actions = tf.split(1, actions.get_shape()[1], actions)
        actions = [tf.squeeze(act) for act in actions]
        states = tf.split(1, states.get_shape()[1], states)
        states = [tf.squeeze(st) for st in states]
        images = tf.split(1, images.get_shape()[1], images)
        images = [tf.squeeze(img) for img in images]
        if pix_distrib != None:
            pix_distrib = tf.split(1, pix_distrib.get_shape()[1], pix_distrib)
            pix_distrib = [tf.squeeze(pix) for pix in pix_distrib]

        if reuse_scope is None:
            gen_images, gen_states, gen_masks, gen_distrib = construct_model(
                images,
                actions,
                states,
                noise,
                iter_num=self.iter_num,
                k=conf['schedsamp_k'],
                use_state=conf['use_state'],
                num_masks=conf['num_masks'],
                cdna=conf['model'] == 'CDNA',
                dna=conf['model'] == 'DNA',
                stp=conf['model'] == 'STP',
                context_frames=conf['context_frames'],
                pix_distributions=pix_distrib,
                conf=conf,
                device_for_variables='/cpu:0')
        else:  # reuse variables from reuse_scope
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images, gen_states, gen_masks, gen_distrib = construct_model(
                    images,
                    actions,
                    states,
                    noise,
                    iter_num=self.iter_num,
                    k=conf['schedsamp_k'],
                    use_state=conf['use_state'],
                    num_masks=conf['num_masks'],
                    cdna=conf['model'] == 'CDNA',
                    dna=conf['model'] == 'DNA',
                    stp=conf['model'] == 'STP',
                    context_frames=conf['context_frames'],
                    conf=conf,
                    device_for_variables='/cpu:0')

        if conf['penal_last_only']:
            cost_sel = np.zeros(conf['sequence_length'] - 2)
            cost_sel[-1] = 1
            print 'using the last state for training only:', cost_sel
        else:
            cost_sel = np.ones(conf['sequence_length'] - 2)

        # L2 loss, PSNR for eval.
        loss, psnr_all, loss_ex = 0.0, 0.0, 0.0
        for i, x, gx in zip(
                range(len(gen_images)), images[conf['context_frames']:],
                gen_images[conf['context_frames'] - 1:]):
            recon_cost = mean_squared_error(x, gx)
            recon_cost_ex = mean_squared_error(x, gx, example_wise=True)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(
                tf.scalar_summary(prefix + '_recon_cost' + str(i), recon_cost))
            summaries.append(tf.scalar_summary(prefix + '_psnr' + str(i), psnr_i))

            loss += recon_cost * cost_sel[i]
            loss_ex += recon_cost_ex * cost_sel[i]

        for i, state, gen_state in zip(
                range(len(gen_states)), states[conf['context_frames']:],
                gen_states[conf['context_frames'] - 1:]):
            state_cost = mean_squared_error(state, gen_state) * 1e-4
            state_cost_ex = mean_squared_error(state, gen_state, example_wise=True) * 1e-4
            summaries.append(
                tf.scalar_summary(prefix + '_state_cost' + str(i), state_cost))
            loss += state_cost * cost_sel[i]
            loss_ex += state_cost_ex * cost_sel[i]
        summaries.append(tf.scalar_summary(prefix + '_psnr_all', psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])
        self.loss_ex = loss_ex = loss_ex / np.float32(len(images) - conf['context_frames'])

        summaries.append(tf.scalar_summary(prefix + '_loss', loss))

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)

        self.gen_images = gen_images
        self.gen_masks = gen_masks
        self.gen_distrib = gen_distrib
        self.gen_states = gen_states

def create_fwd_pass_gpu(conf, reuse_scope, training=True):
    """
    :param fwd_conf:
    :param model:
    :param train_images: single example from queue
    :param train_states: single example from queue
    :param train_actions: single example from queue
    :param sess:
    :return:
    """


    # picking only one one video
    fwd_conf = copy.deepcopy(conf)
    fwd_conf['batch_size'] = 1
    num_smp = fwd_conf['num_smp']

    train_images, train_actions, train_states = build_tfrecord_input(fwd_conf, training=training)

    input_data = [train_images, train_states, train_actions]



    m_video = tf.tile(train_images, [num_smp, 1 , 1, 1, 1])
    m_states = tf.tile(train_states, [num_smp, 1 , 1])
    m_actions = tf.tile(train_actions, [num_smp, 1 , 1])

    model = Model(fwd_conf, reuse_scope =reuse_scope, input_data=[m_video, m_states, m_actions])

    return model, input_data, model.loss_ex

def run_foward_passes(conf, models, loss_ex_ops, input_op_list, sess, itr):

    noise_dim = conf['noise_dim']
    num_smp = conf['num_smp']

    assert conf['batch_size'] % FLAGS.ngpu == 0, 'number of samples in fwd-pass must be a multiple of ngpu'

    mean = np.zeros(noise_dim * conf['sequence_length'])
    cov = np.diag(np.ones(noise_dim * conf['sequence_length']))

    b_noise = np.zeros((conf['batch_size'], conf['sequence_length'], noise_dim))
    w_noise = np.zeros((conf['batch_size'], conf['sequence_length'], noise_dim))

    images_batch = np.zeros((conf['batch_size'], conf['sequence_length'], 64, 64, 3))
    states_batch = np.zeros((conf['batch_size'], conf['sequence_length'], 4))
    actions_batch = np.zeros((conf['batch_size'], conf['sequence_length'], 2))


    start = datetime.now()

    # using different noise for every video in batch
    for b in range(conf['batch_size']/ FLAGS.ngpu):

        for g in range(FLAGS.ngpu):
            noise_vec = np.random.multivariate_normal(mean, cov, size=num_smp)
            noise_vec = noise_vec.reshape((num_smp, conf['sequence_length'], noise_dim))

            model = models[g]
            feed_dict = {
                model.noise: noise_vec,
                model.prefix: 'train',
                model.iter_num: np.float32(itr),
                model.lr: 0
            }

            input_images, input_states, input_actions, cost = sess.run(input_op_list[g] + [loss_ex_ops[g]], feed_dict)

            images_batch[b*FLAGS.ngpu + g] = input_images
            states_batch[b*FLAGS.ngpu + g] = input_states
            actions_batch[b*FLAGS.ngpu + g] = input_actions

            best_index = cost.argsort()[0]
            worst_index = cost.argsort()[-1]

            b_noise[b*FLAGS.ngpu + g] = noise_vec[best_index]
            w_noise[b*FLAGS.ngpu + g] = noise_vec[worst_index]


    if itr % 10 == 0:

        print 'lowest cost of {0}-th sample group: {1}'.format(b, cost[best_index])
        print 'highest cost of {0}-th sample group: {1}'.format(b, cost[worst_index])
        print 'mean cost: {0}, cost std: {1}'.format(np.mean(cost), np.sqrt(np.cov(cost)))


        print 'time for {0} forward passes {1}'.format(conf['batch_size'],
               (datetime.now() - start).seconds + (datetime.now()-start).microseconds / 1e6)

    return images_batch, states_batch, actions_batch, b_noise, w_noise

def construct_towers(conf,training, reusescope=None):
    """

    :param conf:
    :param training: whether tf records uses training or validation data
    :param reusescope: if validation model, this is the scope to be reused from the training model
    :return:
    """

    model_inputs = []
    model_losses = []
    fwd_models = []
    for i in xrange(FLAGS.ngpu):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % (i)) as tower_opscope:

                print('creating tower %d: in scope %s' % (i, tf.get_variable_scope()))
                print 'reuse: ', tf.get_variable_scope().reuse
                import pdb; pdb.set_trace()
                model, input, loss_ex = create_fwd_pass_gpu(conf, reusescope, training)
                model_inputs.append(input)
                model_losses.append(loss_ex)
                fwd_models.append(model)
                tf.get_variable_scope().reuse_variables()

    return fwd_models, model_inputs, model_losses

def main(conf_script=None):
    if FLAGS.device != None:
        start_id = FLAGS.device
    else: start_id = 0
    indexlist = [str(i) for i in range(start_id, start_id + FLAGS.ngpu)]
    var = ','.join(indexlist)
    print 'using CUDA_VISIBLE_DEVICES=', var
    os.environ["CUDA_VISIBLE_DEVICES"] = var
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', FLAGS.hyper)
    conf = hyperparams.configuration
    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf = adapt_params_visualize(conf, FLAGS.visualize)
    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    # with tf.variable_scope('train_model', reuse=None) as training_scope:
    #     model = Model(conf)

    with tf.variable_scope('train', reuse=None) as training_scope:
        fwd_models, inputs_op_list, loss_ex_op  = construct_towers(conf,
                                                                    training= True)

    fwd_models_val, inputs_op_list_val, loss_ex_op_val = construct_towers(conf,
                                                                            reusescope= training_scope,
                                                                            training=False)


    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,
                                                       allow_soft_placement=True,
                                                       log_device_placement=True))
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    # if conf['visualize']:
    #     saver.restore(sess, conf['visualize'])
    #
    #     itr = 0
    #
    #     ## visualize the videos with worst score !!!
    #     videos, states, actions, bestnoise, worstnoise = run_foward_passes(conf, fwd_models,
    #                                                                        loss_ex_op,
    #                                                                        inputs_op_list,
    #                                                                        sess, itr)
    #     feed_dict = {model.images: videos,
    #                  model.states: states,
    #                  model.actions: actions,
    #                  model.noise: bestnoise,
    #                  model.prefix: 'visual',
    #                  model.iter_num: 0,
    #                  model.lr: 0,
    #                  }
    #
    #     gen_images, mask_list = sess.run([model.gen_images, model.gen_masks], feed_dict)
    #     file_path = conf['output_dir']
    #     cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl', 'wb'))
    #     cPickle.dump(videos, open(file_path + '/ground_truth.pkl', 'wb'))
    #     cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
    #     print 'written files to:' + file_path
    #
    #     trajectories = utils_vpred.create_gif.comp_video(conf['output_dir'], conf, suffix='_best')
    #     utils_vpred.create_gif.comp_masks(conf['output_dir'], conf, trajectories, suffix='_best')
    #
    #     ### visualizing videos with highest cost noise:
    #     feed_dict = {model.images: videos,
    #                  model.states: states,
    #                  model.actions: actions,
    #                  model.noise: worstnoise,
    #                  model.prefix: 'visual',
    #                  model.iter_num: 0,
    #                  model.lr: 0,
    #                  }
    #
    #     gen_images, mask_list = sess.run([model.gen_images, model.gen_masks], feed_dict)
    #     file_path = conf['output_dir']
    #     cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl', 'wb'))
    #     cPickle.dump(videos, open(file_path + '/ground_truth.pkl', 'wb'))
    #     cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
    #     print 'written files to:' + file_path
    #
    #     trajectories = utils_vpred.create_gif.comp_video(conf['output_dir'], conf, suffix='_worst')
    #     utils_vpred.create_gif.comp_masks(conf['output_dir'], conf, trajectories, suffix='_worst')
    #
    #     return

    itr_0 = 0
    if conf['pretrained_model']:  # is the order of initialize_all_variables() and restore() important?!?
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
    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()

        # Generate new batch of data_files.
        videos, states, actions, bestnoise, worstnoise = run_foward_passes(conf, fwd_models,
                                                                           loss_ex_op,
                                                                           inputs_op_list,
                                                                           sess, itr)
        # feed_dict = {model.images: videos,
        #              model.states: states,
        #              model.actions: actions,
        #              model.noise: bestnoise,
        #              model.prefix: 'train',
        #              model.iter_num: np.float32(itr),
        #              model.lr: conf['learning_rate'],
        #              }
        # cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
        #                                 feed_dict)

        cost = np.nan
        # Print info: iteration #, cost.
        if (itr) % 10 == 0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            videos, states, actions, bestnoise, worstnoise = run_foward_passes(conf, fwd_models_val,
                                                                               loss_ex_op_val,
                                                                               inputs_op_list_val,
                                                                               sess, itr)
            # feed_dict = {model.images: videos,
            #              model.states: states,
            #              model.actions: actions,
            #              model.noise: bestnoise,
            #              model.prefix: 'val',
            #              model.iter_num: np.float32(itr),
            #              model.lr: 0.0,
            #              }
            # _, val_summary_str = sess.run([model.train_op, model.summ_op],
            #                               feed_dict)
            #
            # summary_writer.add_summary(val_summary_str, itr)

        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model to' + conf['output_dir'])
            saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        t_iter.append((datetime.now() - t_startiter).seconds * 1e6 + (datetime.now() - t_startiter).microseconds)

        if itr % 100 == 1:
            hours = (datetime.now() - starttime).seconds / 3600
            tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                (datetime.now() - starttime).days,
                hours,
                (datetime.now() - starttime).seconds / 60 - hours * 60))
            avg_t_iter = np.sum(np.asarray(t_iter)) / len(t_iter)
            tf.logging.info('time per iteration: {0}'.format(avg_t_iter / 1e6))
            tf.logging.info(
                'expected for complete training: {0}h '.format(avg_t_iter / 1e6 / 3600 * conf['num_iterations']))

        # if (itr) % SUMMARY_INTERVAL:
        #     summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, conf['output_dir'] + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
