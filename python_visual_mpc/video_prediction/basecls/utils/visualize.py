import numpy as np
import collections
import cPickle
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
import tensorflow as tf
import pdb

from collections import OrderedDict

from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig

def create_one_hot(conf, desig_pix_l):
    """
    :param conf:
    :param desig_pix:
    :param repeat_b: create batch of same distribs
    :return:
    """
    if isinstance(desig_pix_l, list):
        one_hot_l = []
        for i in range(len(desig_pix_l)):
            desig_pix = desig_pix_l[i]
            one_hot = np.zeros((1, 1, 64, 64, 1), dtype=np.float32)
            # switch on pixels
            one_hot[0, 0, desig_pix[0], desig_pix[1]] = 1.
            one_hot = np.repeat(one_hot, conf['context_frames'], axis=1)
            app_zeros = np.zeros((1, conf['sequence_length'] - conf['context_frames'], 64, 64, 1), dtype=np.float32)
            one_hot = np.concatenate([one_hot, app_zeros], axis=1)
            one_hot_l.append(one_hot)

        return one_hot_l
    else:
        one_hot = np.zeros((1, 1, 64, 64, 1), dtype=np.float32)
        # switch on pixels
        if 'modelconfiguration' in conf:
            if conf['modelconfiguration']['dilation_rate'] == [2,2]:
                one_hot[0, 0, desig_pix_l[0]-1:desig_pix_l[0]+1:, desig_pix_l[1]-1:desig_pix_l[1]+1] = 0.25
            else:
                one_hot[0, 0, desig_pix_l[0], desig_pix_l[1]] = 1.
        else:
            one_hot[0, 0, desig_pix_l[0], desig_pix_l[1]] = 1.

        one_hot = np.repeat(one_hot, conf['context_frames'], axis=1)
        app_zeros = np.zeros((1, conf['sequence_length']- conf['context_frames'], 64, 64, 1), dtype=np.float32)
        one_hot = np.concatenate([one_hot, app_zeros], axis=1)
        one_hot = np.repeat(one_hot, conf['batch_size'], axis=0)

        return one_hot

def visualize(sess, conf, model):
    feed_dict = {model.train_cond: 1}
    if hasattr(model, 'lr'):
        feed_dict[model.lr] = 0.0
    if hasattr(model, 'iter_num'):
        feed_dict[model.iter_num] = 0

    file_path = conf['output_dir']

    if not isinstance(model.gen_images, list):
        model.gen_images = tf.unstack(model.gen_images, axis=1)

    ground_truth, gen_images = sess.run([model.images,
                                       model.gen_images,
                                       # model.gen_masks,
                                       # model.prediction_flow,
                                       ],
                                       feed_dict)

    dict = OrderedDict()
    import re
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    dict['iternum'] = itr_vis
    dict['ground_truth'] = ground_truth
    dict['gen_images'] = gen_images
    # dict['prediction_flow'] = pred_flow
    # dict['gen_masks'] = gen_masks

    cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
    print 'written files to:' + file_path

    v = Visualizer_tkinter(dict, numex=conf['batch_size'], append_masks=True, filepath=conf['output_dir'])
    v.build_figure()

def visualize_diffmotions(sess, conf, model):

    try:
        feed_dict = {model.iter_num: 0}
    except AttributeError:
        feed_dict = {}

    b_exp, ind0 =1, 0

    if 'adim' in conf:
        from python_visual_mpc.video_prediction.read_tf_record_wristrot import \
            build_tfrecord_input as build_tfrecord_fn
    else:
        from python_visual_mpc.video_prediction.read_tf_record_sawyer12 import \
            build_tfrecord_input as build_tfrecord_fn
    val_images, _, val_states = build_tfrecord_fn(conf, training=False)

    tf.train.start_queue_runners(sess)
    img, state = sess.run([val_images, val_states])

    sel_img= img[b_exp,ind0:ind0+2]

    statedim = conf['sdim']
    adim = conf['adim']

    c = Getdesig(sel_img[0], conf, 'b{}'.format(b_exp))
    desig_pos = c.coords.astype(np.int32)
    # desig_pos = np.array([29, 37])
    print "selected designated position for aux1 [row,col]:", desig_pos

    one_hot = create_one_hot(conf, desig_pos)

    feed_dict[model.pix_distrib_pl] = one_hot

    sel_state = np.stack([state[b_exp, ind0], state[b_exp, ind0 + 1]], axis=0)

    start_states = np.concatenate([sel_state, np.zeros((conf['sequence_length'] - 2, statedim))])
    start_states = np.expand_dims(start_states, axis=0)
    start_states = np.repeat(start_states, conf['batch_size'], axis=0)  # copy over batch
    feed_dict[model.states_pl] = start_states

    start_images = np.concatenate([sel_img, np.zeros((conf['sequence_length'] - 2, 64, 64, 3))])

    start_images = np.expand_dims(start_images, axis=0)
    start_images = np.repeat(start_images, conf['batch_size'], axis=0)  # copy over batch
    feed_dict[model.images_pl] = start_images

    actions = np.zeros([conf['batch_size'], conf['sequence_length'], adim])

    # step = .025
    step = .055
    n_angles = 8
    col_titles = []
    for b in range(n_angles):
        col_titles.append('move')
        for i in range(conf['sequence_length']):
            actions[b, i][:2] = np.array(
                [np.cos(b / float(n_angles) * 2 * np.pi) * step, np.sin(b / float(n_angles) * 2 * np.pi) * step])

    if adim == 5:
        b += 1
        actions[b, 0] = np.array([0, 0, 4, 0, 0])
        actions[b, 1] = np.array([0, 0, 4, 0, 0])
        col_titles.append('up/down')

        b += 1
        actions[b, 0] = np.array([0, 0, 0, 0, 4])
        actions[b, 1] = np.array([0, 0, 0, 0, 4])
        col_titles.append('close/open')

        delta_rot = 0.4
        b += 1
        for i in range(conf['sequence_length']):
            actions[b, i] = np.array([0, 0, 0, delta_rot, 0])
        col_titles.append('rot +')

        b += 1
        for i in range(conf['sequence_length']):
            actions[b, i] = np.array([0, 0, 0, -delta_rot, 0])
        col_titles.append('rot -')

        col_titles.append('noaction')

    elif adim == 4:
        b += 1
        actions[b, 0] = np.array([0, 0, 4, 0])
        actions[b, 1] = np.array([0, 0, 4, 0])

        b += 1
        actions[b, 0] = np.array([0, 0, 0, 4])
        actions[b, 1] = np.array([0, 0, 0, 4])

    feed_dict[model.actions_pl] = actions

    if not isinstance(model.gen_images, list):
        model.gen_images = tf.unstack(model.gen_images, axis=1)

    if model.gen_pix_distribs is not None:
        if not isinstance(model.gen_pix_distribs, list):
            model.gen_pix_distribs = tf.unstack(model.gen_pix_distribs, axis=1)

    gen_images, gen_distrib = sess.run([model.gen_images, model.gen_pix_distribs]
                                        ,feed_dict)


    # gen_images, gen_distrib, gen_masks, moved_parts, moved_images, moved_bckgd = sess.run([model.gen_images,
    #                                                                                        model.gen_distrib1,
    #                                                                                        model.gen_masks,
    #                                                                                        model.moved_parts_list,
    #                                                                                        model.moved_images,
    #                                                                                        model.moved_bckgd
    #                                                                                        ]
    #                                                                                     ,feed_dict)
    dict = OrderedDict()
    dict['gen_images'] = gen_images
    # dict['gen_masks'] = gen_masks
    dict['gen_distrib'] = gen_distrib
    import re
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    dict['iternum'] = itr_vis
    dict['desig_pos'] = desig_pos
    # dict['moved_parts'] = moved_parts
    # dict['moved_images'] = moved_images
    # dict['moved_bckgd'] = moved_bckgd

    file_path = conf['output_dir']
    cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
    print 'written files to:' + file_path

    v = Visualizer_tkinter(dict, numex=b+1, append_masks=False,
                           filepath=conf['output_dir'],
                           suf='_diffmotions_b{}_l{}'.format(b_exp, conf['sequence_length']),
                           renorm_heatmaps=False)
    v.build_figure()