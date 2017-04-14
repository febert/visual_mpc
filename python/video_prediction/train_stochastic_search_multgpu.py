import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import copy

from utils_vpred.adapt_params_visualize import adapt_params_visualize
# from video_prediction.utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import utils_vpred.create_gif

import utils_vpred

from read_tf_record import build_tfrecord_input


from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

FLAGS = flags.FLAGS


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

        # self.iter_num = tf.placeholder(tf.float32, shape=(1),name='iter_num')
        self.iter_num = tf.placeholder(tf.float32, [], name= 'iternum')

        self.prefix = prefix = tf.placeholder(tf.string, [])

        summaries = []

        if input_data == None:
            self.images = images = tf.placeholder(tf.float32, name='images',
                                                  shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
            self.actions = actions = tf.placeholder(tf.float32, name='actions',
                                                    shape=(conf['batch_size'], conf['sequence_length'], 2))
            self.states = states = tf.placeholder(tf.float32, name='states',
                                                  shape=(conf['batch_size'], conf['sequence_length'], 4))
            self.noise = noise = tf.placeholder(tf.float32, name='noise',
                                                shape=(conf['batch_size'], conf['sequence_length'], conf['noise_dim']))
        else:
            if len(input_data) == 4:
                images, states, actions, noise = input_data
            elif len(input_data) == 5:
                images, states, actions, noise, pix_distrib = input_data
            else:
                raise ValueError()

        # Split into timesteps.
        noise = tf.split(1, noise.get_shape()[1], noise)
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
                    pix_distributions=pix_distrib,
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
            summaries.append(tf.summary.scalar('psnr' + str(i), psnr_i))

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
        summaries.append(tf.scalar_summary('_psnr_all', psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])
        self.loss_ex = loss_ex = loss_ex / np.float32(len(images) - conf['context_frames'])

        summaries.append(tf.scalar_summary('loss', loss))

        self.lr = tf.placeholder_with_default(conf['learning_rate'], (), name='learningrate')

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)

        self.gen_images = gen_images
        self.gen_masks = gen_masks
        self.gen_distrib = gen_distrib
        self.gen_states = gen_states

class Tower(object):
    def __init__(self, conf, gpu_id, reuse_scope, train_images, train_states, train_actions):

        num_smp = conf['num_smp']

        #pick the right example from the batch of size ngpu:
        cp_images = tf.slice(train_images, [gpu_id,0,0,0,0], [1,-1,-1,-1,-1])
        # and replicate with number num_smp
        self.cp_images = tf.tile(cp_images, [num_smp, 1 , 1, 1, 1])

        cp_states = tf.slice(train_states, [gpu_id, 0, 0], [1, -1, -1])
        self.cp_states = tf.tile(cp_states, [num_smp, 1 , 1])

        cp_actions = tf.slice(train_actions, [gpu_id, 0, 0], [1, -1, -1])
        self.cp_actions = tf.tile(cp_actions, [num_smp, 1 , 1])

        self.noise = tf.truncated_normal([num_smp, conf['sequence_length'], conf['noise_dim']],
                                         mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

        self.model = Model(conf, reuse_scope =reuse_scope, input_data=[self.cp_images,
                                                                       self.cp_states,
                                                                       self.cp_actions,
                                                                       self.noise])


def run_foward_passes(conf, sess, itr, towers, train_images, train_states, train_actions, print_timing= False):

    noise_dim = conf['noise_dim']

    assert conf['batch_size'] % FLAGS.ngpu == 0, 'number of samples in fwd-pass must be a multiple of ngpu'

    b_noise = np.zeros((conf['batch_size'], conf['sequence_length'], noise_dim))
    w_noise = np.zeros((conf['batch_size'], conf['sequence_length'], noise_dim))

    images_batch = np.zeros((conf['batch_size'], conf['sequence_length'], 64, 64, 3))
    states_batch = np.zeros((conf['batch_size'], conf['sequence_length'], 4))
    actions_batch = np.zeros((conf['batch_size'], conf['sequence_length'], 2))

    start = datetime.now()

    # using different noise for every video in batch
    for b in range(conf['batch_size']/ FLAGS.ngpu):

        # put all tower's placeholders in one feedict
        feed_dict = {}
        for t in towers:
            feed_dict[t.model.iter_num] = np.float32(itr)
            feed_dict[t.model.lr] = 0.0

        #pack all towers operations which need to be evaluated in one list
        op_list = []
        op_list += [train_images, train_states, train_actions]
        for t in towers:
            op_list.append(t.model.loss_ex)
            op_list.append(t.noise)

        if not print_timing:
            res_list = sess.run(op_list, feed_dict)
        else:
            run_metadata = tf.RunMetadata()
            res_list = sess.run(op_list,
                         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                         run_metadata=run_metadata)

            tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                run_meta=run_metadata,
                tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

        # unpack results
        input_images = res_list.pop(0)
        input_states = res_list.pop(0)
        input_actions = res_list.pop(0)

        for g in range(FLAGS.ngpu):

            images_batch[b*FLAGS.ngpu + g] = input_images[g]
            states_batch[b*FLAGS.ngpu + g] = input_states[g]
            actions_batch[b*FLAGS.ngpu + g] = input_actions[g]

            cost = res_list.pop(0)
            noise_vec = res_list.pop(0)

            best_index = cost.argsort()[0]
            worst_index = cost.argsort()[-1]

            b_noise[b*FLAGS.ngpu + g] = noise_vec[best_index]
            w_noise[b*FLAGS.ngpu + g] = noise_vec[worst_index]

    if itr % 10 == 0:

        print 'lowest cost of {0}-th sample group: {1}'.format(b, cost[best_index])
        print 'highest cost of {0}-th sample group: {1}'.format(b, cost[worst_index])
        print 'mean cost: {0}, cost std: {1}'.format(np.mean(cost), np.sqrt(np.cov(cost)))

        print 'step {0}, time for {1} forward passes {2}'.format(itr, conf['batch_size'],
               (datetime.now() - start).seconds + (datetime.now()-start).microseconds / 1e6)

    return images_batch, states_batch, actions_batch, b_noise, w_noise

def construct_towers(conf ,training, reusescope=None):
    """
    :param conf:
    :param training: whether tf records uses training or validation data
    :param reusescope: if validation model, this is the scope to be reused from the training model
    :return:
    """

    # picking only one one video
    fwd_conf = copy.deepcopy(conf)
    fwd_conf['batch_size'] = FLAGS.ngpu
    train_images, train_actions, train_states = build_tfrecord_input(fwd_conf, training=training)

    towers = []

    for i in xrange(FLAGS.ngpu):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % (i)):
                print('creating tower %d: in scope %s' % (i, tf.get_variable_scope()))
                # print 'reuse: ', tf.get_variable_scope().reuse

                towers.append(Tower(conf, i, reusescope, train_images, train_states, train_actions))
                tf.get_variable_scope().reuse_variables()

    return towers, train_images, train_states, train_actions


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

    with tf.variable_scope('train_model', reuse=None) as training_scope:
        model = Model(conf)


    train_towers, train_images, train_states, train_actions  = construct_towers(conf, training= True, reusescope= training_scope)

    val_towers, val_images, val_states, val_actions = construct_towers(conf, reusescope= training_scope,
                                                                            training=False)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,
                                                       allow_soft_placement=True,
                                                       log_device_placement=False))
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    if conf['visualize']:
        saver.restore(sess, conf['visualize'])

        itr = 0

        ## visualize the videos with worst score !!!
        videos, states, actions, bestnoise, worstnoise = run_foward_passes(conf,
                                                                           sess,
                                                                           itr,
                                                                           train_towers,
                                                                           train_images,
                                                                           train_states,
                                                                           train_actions)
        feed_dict = {model.images: videos,
                     model.states: states,
                     model.actions: actions,
                     model.noise: bestnoise,
                     model.iter_num: 0,
                     model.lr: 0,
                     model.prefix: ''
                     }

        gen_images, mask_list = sess.run([model.gen_images, model.gen_masks], feed_dict)
        file_path = conf['output_dir']
        cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl', 'wb'))
        cPickle.dump(videos, open(file_path + '/ground_truth.pkl', 'wb'))
        cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
        print 'written files to:' + file_path

        trajectories = utils_vpred.create_gif.comp_video(conf['output_dir'], conf, suffix='_best')
        utils_vpred.create_gif.comp_masks(conf['output_dir'], conf, trajectories, suffix='_best')

        ### visualizing videos with highest cost noise:
        feed_dict = {model.images: videos,
                     model.states: states,
                     model.actions: actions,
                     model.noise: worstnoise,
                     model.iter_num: 0,
                     model.lr: 0,
                     model.prefix: ''
                     }

        gen_images, mask_list = sess.run([model.gen_images, model.gen_masks], feed_dict)
        file_path = conf['output_dir']
        cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl', 'wb'))
        cPickle.dump(videos, open(file_path + '/ground_truth.pkl', 'wb'))
        cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
        print 'written files to:' + file_path

        trajectories = utils_vpred.create_gif.comp_video(conf['output_dir'], conf, suffix='_worst')
        utils_vpred.create_gif.comp_masks(conf['output_dir'], conf, trajectories, suffix='_worst')

        return

    itr_0 = 0
    if FLAGS.pretrained:  # is the order of initialize_all_variables() and restore() important?!?

        pretr_model = conf['output_dir'] + '/' + FLAGS.pretrained
        print 'using pretrained model from :' + pretr_model
        saver.restore(sess, pretr_model)
        # resume training at iteration step of the loaded model:
        import re
        itr_0 = re.match('.*?([0-9]+)$', pretr_model).group(1)
        itr_0 = int(itr_0)
        print 'resuming training at iteration:  ', itr_0


    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter_list = []
    timing = False
    # Run training.
    for itr in range(itr_0, conf['num_iterations'], 1):

        # if itr%10 == 0:
        #     timing = True

        t_startiter = datetime.now()

        # Generate new batch of data_files.
        videos, states, actions, bestnoise, worstnoise = run_foward_passes(conf,
                                                                           sess,
                                                                           itr,
                                                                           train_towers,
                                                                           train_images,
                                                                           train_states,
                                                                           train_actions,
                                                                           print_timing=timing)
        timing = False

        start_fwd_backwd =datetime.now()
        feed_dict = {model.images: videos,
                     model.states: states,
                     model.actions: actions,
                     model.noise: bestnoise,
                     model.iter_num: np.float32(itr),
                     model.lr: conf['learning_rate'],
                     model.prefix: 'train'
                     }
        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        if itr % 10 == 0:
            print 'time training step {0} (single forward-backward pass) {1}'.format(itr,
                                                           (datetime.now() - start_fwd_backwd).seconds + (
                                                           datetime.now() - start_fwd_backwd).microseconds / 1e6)


        # Print info: iteration #, cost.
        if (itr) % 10 == 0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            videos, states, actions, bestnoise, worstnoise = run_foward_passes(conf,
                                                                                sess,
                                                                                itr,
                                                                                val_towers,
                                                                                val_images,
                                                                                val_states,
                                                                                val_actions)
            feed_dict = {model.images: videos,
                         model.states: states,
                         model.actions: actions,
                         model.noise: bestnoise,
                         model.iter_num: np.float32(itr),
                         model.lr: 0.0,
                         model.prefix: 'val'
                         }
            _, val_summary_str = sess.run([model.train_op, model.summ_op],
                                          feed_dict)

            summary_writer.add_summary(val_summary_str, itr)

        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model to' + conf['output_dir'])
            saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        t_iter_list.append((datetime.now() - t_startiter).seconds * 1e6 + (datetime.now() - t_startiter).microseconds)

        if itr % 10 == 0:
            hours = (datetime.now() - starttime).seconds / 3600
            tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                (datetime.now() - starttime).days,
                hours,
                (datetime.now() - starttime).seconds / 60 - hours * 60))

            t_iter = (datetime.now() - t_startiter).seconds + (datetime.now() - t_startiter).microseconds / 1e6
            tf.logging.info('time for iteration {0}: {1}'.format(itr, t_iter))

            avg_t_iter = np.sum(np.asarray(t_iter_list)) / len(t_iter_list)
            tf.logging.info(
                'expected time for complete training: {0}h '.format(avg_t_iter / 1e6 / 3600 * conf['num_iterations']))

        if (itr) % SUMMARY_INTERVAL:
            summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, conf['output_dir'] + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', None, 'the gpu number to start with')
    flags.DEFINE_string('pretrained', None, 'name of the model to resume training from. e.g. model10002')
    flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')

    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
