import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import pdb

import imp
import matplotlib.pyplot as plt
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from makegifs import comp_gif


from datetime import datetime
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 200

# How often to run a batch through the validation model.
VAL_INTERVAL = 400

# How often to save a model checkpoint
SAVE_INTERVAL = 2000


from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter

from PIL import Image

FLAGS = flags.FLAGS
flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')
flags.DEFINE_bool('diffmotions', False, 'visualize several different motions for a single scene')




def main(unused_argv, conf_script= None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    if conf_script == None: conf_file = FLAGS.hyper
    else: conf_file = conf_script

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    inference = False
    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf['schedsamp_k'] = -1  # don't feed ground truth
        conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
        conf['visualize'] = conf['output_dir'] + '/' + FLAGS.visualize
        conf['event_log_dir'] = '/tmp'
        conf.pop('use_len', None)
        conf['batch_size'] = 10

        conf['sequence_length'] = 14
        if FLAGS.diffmotions:
            inference = True
            conf['sequence_length'] = 30

    Model= conf['model']

    print 'Constructing models and inputs.'
    if FLAGS.diffmotions:

        model = Model(load_data = False)
        model.visualize_diffmotions


    print 'Constructing saver.'
    # Make saver.

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # remove all states from group of variables which shall be saved and restored:
    vars_no_state = filter_vars(vars)
    saver = tf.train.Saver(vars_no_state, max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    if conf['visualize']:
        print '-------------------------------------------------------------------'
        print 'verify current settings!! '
        for key in conf.keys():
            print key, ': ', conf[key]
        print '-------------------------------------------------------------------'

        import re
        itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)

        saver.restore(sess, conf['visualize'])
        print 'restore done.'

        feed_dict = {val_model.lr: 0.0,
                     val_model.iter_num: 0 }

        file_path = conf['output_dir']

        if FLAGS.diffmotions:

            b_exp, ind0 =0, 0

            img, state = sess.run([val_images, val_states])
            sel_img= img[b_exp,ind0:ind0+2]

            # c = Getdesig(sel_img[0], conf, 'b{}'.format(b_exp))
            # desig_pos_aux1 = c.coords.astype(np.int32)
            desig_pos_aux1 = np.array([30, 31])
            # print "selected designated position for aux1 [row,col]:", desig_pos_aux1

            one_hot = create_one_hot(conf, desig_pos_aux1)

            feed_dict[pix_distrib_pl] = one_hot

            sel_state = np.stack([state[b_exp,ind0],state[b_exp,ind0+1]], axis=0)

            start_states = np.concatenate([sel_state,np.zeros((conf['sequence_length']-2, 3))])
            start_states = np.expand_dims(start_states, axis=0)
            start_states = np.repeat(start_states, conf['batch_size'], axis=0)  # copy over batch
            feed_dict[states_pl] = start_states

            start_images = np.concatenate([sel_img,np.zeros((conf['sequence_length']-2, 64, 64, 3))])

            start_images = np.expand_dims(start_images, axis=0)
            start_images = np.repeat(start_images, conf['batch_size'], axis=0)  # copy over batch
            feed_dict[images_pl] = start_images

            actions = np.zeros([conf['batch_size'], conf['sequence_length'], 4])

            # step = .025
            step = .055
            n_angles = 8
            for b in range(n_angles):
                for i in range(conf['sequence_length']):
                    actions[b,i] = np.array([np.cos(b/float(n_angles)*2*np.pi)*step, np.sin(b/float(n_angles)*2*np.pi)*step, 0, 0])

            b+=1
            actions[b, 0] = np.array([0, 0, 4, 0])
            actions[b, 1] = np.array([0, 0, 4, 0])

            b += 1
            actions[b, 0] = np.array([0, 0, 0, 4])
            actions[b, 1] = np.array([0, 0, 0, 4])

            feed_dict[actions_pl] = actions

            gen_images, gen_distrib, gen_masks, moved_parts, moved_images, moved_bckgd = sess.run([val_model.m.gen_images,
                                                            val_model.m.gen_distrib1,
                                                            val_model.m.gen_masks,
                                                            val_model.m.movd_parts_list,
                                                            val_model.m.moved_images,
                                                            val_model.m.moved_bckgd
                                                            ]
                                                           ,feed_dict)
            dict = {}
            dict['gen_images'] = gen_images
            dict['gen_masks'] = gen_masks
            dict['gen_distrib'] = gen_distrib
            dict['iternum'] = itr_vis
            # dict['moved_parts'] = moved_parts
            # dict['moved_images'] = moved_images
            # dict['moved_bckgd'] = moved_bckgd

            cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
            print 'written files to:' + file_path

            make_gif = False
            if make_gif:
                comp_gif(conf, conf['output_dir'], append_masks=False,
                         suffix='_diffmotions_b{}_l{}'.format(b_exp, conf['sequence_length']))
            else:
                v = Visualizer_tkinter(dict, numex=4, append_masks=False,
                                       gif_savepath=conf['output_dir'],
                                       suf='_diffmotions_b{}_l{}'.format(b_exp, conf['sequence_length']))
                v.build_figure()

        else:
            ground_truth, gen_images, gen_masks = sess.run([val_images,
                                                            val_model.m.gen_images,
                                                            val_model.m.gen_masks
                                                            ],
                                                            feed_dict)
            dict = {}
            dict['iternum'] = itr_vis
            dict['gen_images'] = gen_images
            dict['ground_truth'] = ground_truth
            dict['gen_masks'] = gen_masks

            cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
            print 'written files to:' + file_path

            make_gif = False
            if make_gif:
                comp_gif(conf, conf['output_dir'], append_masks=True)
            else:
                Visualizer_tkinter(dict, numex=4, append_masks=True, gif_savepath=conf['output_dir'])

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

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter = []
    # Run training.

    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()
        # Generate new batch of data_files.
        feed_dict = {model.iter_num: np.float32(itr)}

        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {val_model.iter_num: np.float32(itr)}
            [val_summary_str] = sess.run([val_model.summ_op], feed_dict)
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


def create_one_hot(conf, desig_pix):
    one_hot = np.zeros((1, 1, 64, 64, 1), dtype=np.float32)
    # switch on pixels
    one_hot[0, 0, desig_pix[0], desig_pix[1]] = 1.
    one_hot = np.repeat(one_hot, conf['context_frames'], axis=1)
    app_zeros = np.zeros((1, conf['sequence_length']- conf['context_frames'], 64, 64, 1), dtype=np.float32)
    one_hot = np.concatenate([one_hot, app_zeros], axis=1)
    one_hot = np.repeat(one_hot, conf['batch_size'], axis=0)

    return one_hot


def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print 'removed state variable from saving-list: ', v.name

    return newlist

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
