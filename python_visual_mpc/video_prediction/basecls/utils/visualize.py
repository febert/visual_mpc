import numpy as np
import  sys
import collections
import pickle
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
from python_visual_mpc.video_prediction.utils_vpred.create_images import Image_Creator
import tensorflow as tf
import pdb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from copy import deepcopy
from collections import OrderedDict

from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig
from python_visual_mpc.video_prediction.read_tf_records2 import build_tfrecord_input

def create_one_hot(conf, desig_pix_l, batch_mode=False):
    """
    :param conf:
    :param desig_pix: list of np.arrays of size [ndesig, 2]
    :param repeat_b: create batch of same distribs
    :return: one_hot_l with [batch_size, sequence_l, ndesig, conf['img_height'], conf['img_width'],1]
    """
    one_hot_l = []
    ndesig = desig_pix_l[0].shape[0]
    orig_size = conf['orig_size']

    for i in range(len(desig_pix_l)):

        desig_pix = desig_pix_l[i]
        one_hot = np.zeros((1, conf['context_frames'], ndesig, orig_size[0], orig_size[1], 1), dtype=np.float32)
        for p in range(ndesig):
            # switch on pixels
            if 'modelconfiguration' in conf:
                if conf['modelconfiguration']['dilation_rate'] == [2, 2]:
                    one_hot[0, 0, p, desig_pix[p, 0] - 1:desig_pix[p, 0] + 1:, desig_pix[p, 1] - 1:desig_pix[p, 1] + 1] = 0.25
                else:
                    one_hot[0, 0, p, desig_pix[p, 0], desig_pix[p, 1]] = 1.
            else:
                one_hot[0, 0, p, desig_pix[p, 0], desig_pix[p, 1]] = 1.

        app_zeros = np.zeros((1, conf['sequence_length'] - conf['context_frames'], ndesig, orig_size[0], orig_size[1], 1), dtype=np.float32)
        one_hot = np.concatenate([one_hot, app_zeros], axis=1)
        one_hot_l.append(one_hot)

    if len(one_hot_l) == 1:
        one_hot_l = [np.repeat(one_hot_l[0], conf['batch_size'], axis=0)]

    return one_hot_l

def visualize(sess, conf, model):
    feed_dict = {model.train_cond: 1}
    if hasattr(model, 'lr'):
        feed_dict[model.lr] = 0.0
    if hasattr(model, 'iter_num'):
        feed_dict[model.iter_num] = 0

    file_path = conf['output_dir']

    if not isinstance(model.gen_images, list):
        model.gen_images = tf.unstack(model.gen_images[:,:,0], axis=1)
        model.images = tf.unstack(model.images, axis=1)

    ground_truth, gen_images, states, actions = sess.run([model.images,
                                         model.gen_images,
                                         model.states,
                                         model.actions,
                                         # model.gen_masks,
                                       # model.prediction_flow,
                                       ],
                                       feed_dict)

    pdb.set_trace()
    dict = OrderedDict()
    dict['iternum'] = conf['num_iter']
    dict['ground_truth'] = ground_truth
    dict['gen_images'] = gen_images
    # dict['actions'] = actions
    # dict['states'] = states
    # dict['prediction_flow'] = pred_flow
    # dict['gen_masks_l'] = gen_masks

    pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
    print('written files to:' + file_path)

    # v = Visualizer_tkinter(dict, numex=conf['batch_size'], append_masks=False, filepath=conf['output_dir'],
    #                        col_titles=[str(i) for i in range(conf['batch_size'])])
    v = Visualizer_tkinter(dict, numex=conf['batch_size'], append_masks=False, filepath=conf['output_dir'])
    v.make_direct_vid()

    sys.exit()


def visualize_diffmotions(sess, conf, model):

    try:
        feed_dict = {model.iter_num: 0}
    except AttributeError:
        feed_dict = {}

    b_exp, ind0 = 28, 0

    copied_conf = deepcopy(conf)
    copied_conf['sequence_length'] = 2

    dict = build_tfrecord_input(conf, training=False)
    val_images, val_actions, val_states = dict['images'], dict['actions'], dict['endeffector_pos']

    tf.train.start_queue_runners(sess)
    img, state, gtruth_actions = sess.run([val_images, val_states, val_actions])

    sel_img= img[b_exp,ind0:ind0+2]

    statedim = conf['sdim']
    adim = conf['adim']

    #c = Getdesig(sel_img[0], conf, 'b{}'.format(b_exp))
    #desig_pos = c.coords.astype(np.int32).reshape([1,2])
    desig_pos = np.array([36, 16]).reshape([1,2])
    print("selected designated position for aux1 [row,col]:", desig_pos)

    one_hot = create_one_hot(conf, [desig_pos])[0]

    sel_state = np.stack([state[b_exp, ind0], state[b_exp, ind0 + 1]], axis=0)

    start_states = np.concatenate([sel_state, np.zeros((conf['sequence_length'] - 2, statedim))])
    start_states = np.expand_dims(start_states, axis=0)
    start_states = np.repeat(start_states, conf['batch_size'], axis=0)  # copy over batch

    orig_size = conf['orig_size']

    start_images = np.concatenate([sel_img, np.zeros((conf['sequence_length'] - 2, orig_size[0], orig_size[1], 3))])
    start_images = np.expand_dims(start_images, axis=0)
    start_images = np.repeat(start_images, conf['batch_size'], axis=0)  # copy over batch

    actions = np.zeros([conf['batch_size'], conf['sequence_length'], adim])

    # step = 10
    if 'vis_step' in conf:
        step = conf['vis_step']
    else:
        step = .055

    n_angles = 8  # 8
    col_titles = []
    for b in range(n_angles):
        col_titles.append('move')
        for i in range(conf['sequence_length']):
            actions[b, i][:2] = np.array(
                [np.cos(b / float(n_angles) * 2 * np.pi) * step, np.sin(b / float(n_angles) * 2 * np.pi) * step])

    if adim >= 3:
        b += 1
        if 'vis_updown_step' in conf:
            updown_step = conf['vis_updown_step']
            actions[b, 0:7, 2] = updown_step
            actions[b, 7:15, 2] = -updown_step
        else:
            updown_step = 4
            actions[b, 0, 2] = updown_step
            actions[b, 1, 2] = updown_step
        col_titles.append('up/down')

    if adim >= 4:
        if 'vis_rot_step' in conf:
            rot_step = conf['vis_rot_step']
        else: rot_step = 4

        b += 1
        for i in range(conf['sequence_length']):
            actions[b, i, 3] = rot_step
        col_titles.append('rot +')

        b += 1
        for i in range(conf['sequence_length']):
            actions[b, i, 3] = -rot_step
        col_titles.append('rot -')

    if adim >= 5:
        b += 1
        actions[b, 0, 4] = 4
        actions[b, 1, 4] = 4
        col_titles.append('close/open')


    if 'float16' in conf:
        use_dtype = np.float16
    else: use_dtype = np.float32

    feed_dict[model.actions_pl] = actions.astype(use_dtype)
    feed_dict[model.states_pl] = start_states.astype(use_dtype)
    feed_dict[model.pix_distrib_pl] = one_hot.astype(use_dtype)
    feed_dict[model.images_pl] = start_images.astype(use_dtype)

    gen_images, gen_distrib = sess.run([model.gen_images,
                                        model.gen_distrib,
                                        # model.gen_masks,
                                        # model.gen_transformed_images,
                                        # model.gen_transformed_pixdistribs
                                        ]
                                        ,feed_dict)


    dict = OrderedDict()
    gen_images = [im.astype(np.float32) for im in gen_images]
    dict['gen_images'] = gen_images

    gen_distrib = [im.astype(np.float32) for im in gen_distrib]
    assert gen_distrib[0].shape[1] == 1
    gen_distrib = [d[:,0] for d in gen_distrib]
    dict['gen_distrib'] = gen_distrib

    # dict['gen_masks_l'] = gen_masks
    # dict['gen_transf_images_l'] = gen_transf_images
    # dict['gen_transf_distribs_l'] = gen_transf_distribs

    dict['iternum'] = conf['num_iter']
    dict['desig_pos'] = desig_pos[0]
    # dict['moved_parts'] = moved_parts
    # dict['moved_images'] = moved_images
    # dict['moved_bckgd'] = moved_bckgd

    file_path = conf['output_dir']
    pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
    print('written files to:' + file_path)
    if 'test_data_ind' in conf:
        suf = '_dataind{}'.format(conf['test_data_ind'])
    else: suf = ''

    v = Visualizer_tkinter(dict, numex=b+1, append_masks=True,
                           filepath=conf['output_dir'],
                           suf='_diffmotions_b{}_l{}'.format(b_exp, conf['sequence_length'])+suf,
                           renorm_heatmaps=True)
    v.build_figure()


def compute_metric(sess, conf, model, create_images=False):

    if 'adim' in conf:
        from python_visual_mpc.video_prediction.read_tf_record_wristrot import \
            build_tfrecord_input as build_tfrecord_fn
    else:
        from python_visual_mpc.video_prediction.read_tf_record_sawyer12 import \
            build_tfrecord_input as build_tfrecord_fn

    conf['test_metric'] = {'robot_pos': 1, 'object_pos': 2}
    conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test_annotations'])
    conf['train_val_split'] = 1.

    image_batch, action_batch, endeff_pos_batch, robot_pos_batch, object_pos_batch = build_tfrecord_fn(conf, training=True)

    tf.train.start_queue_runners(sess)

    n_runs = int(128/ conf['batch_size'])

    rob_exp_dist_l, rob_log_prob_l = [], []
    pos0_exp_dist_l, pos0_log_prob_l = [], []
    pos1_exp_dist_l, pos1_log_prob_l = [], []
    for i_run in range(n_runs):
        images, actions, endeff, robot_pos, object_pos = sess.run([image_batch, action_batch, endeff_pos_batch,
                                                                   robot_pos_batch, object_pos_batch])

        rob_gen_images_, rob_gen_distrib_, rob_exp_dist, rob_log_prob, flow = compute_exp_distance(sess, conf, model, robot_pos,
                                                                               images, actions, endeff)
        pos0_gen_images_, pos0_gen_distrib_, pos0_exp_dist, pos0_log_prob, _ = compute_exp_distance(sess, conf, model, object_pos[:,:,0],
                                                                                images, actions, endeff)  #for object_pos 0
        pos1_gen_images_, pos1_gen_distrib_, pos1_exp_dist, pos1_log_prob, _ = compute_exp_distance(sess, conf, model, object_pos[:,:,1],
                                                                                images, actions, endeff)  #for object_pos 1

        rob_exp_dist_l.append(rob_exp_dist)
        rob_log_prob_l.append(rob_log_prob)

        pos0_exp_dist_l.append(pos0_exp_dist)
        pos0_log_prob_l.append(pos0_log_prob)

        pos1_exp_dist_l.append(pos1_exp_dist)
        pos1_log_prob_l.append(pos1_log_prob)

        #get first frst batch data for visualizing
        if i_run == 0:
            fb_images = deepcopy(images)
            fb_robot_pos = deepcopy(robot_pos)
            fb_object_pos = deepcopy(object_pos)

            fb_rob_gen_images, fb_rob_gen_distrib = rob_gen_images_, rob_gen_distrib_
            fb_pos0_gen_images, fb_pos0_gen_distrib = pos0_gen_images_, pos0_gen_distrib_
            fb_pos1_gen_images, fb_pos1_gen_distrib = pos1_gen_images_, pos1_gen_distrib_

    mean_rob_exp_dist = np.mean(np.stack(rob_exp_dist_l))
    std_rob_exp_dist = np.std(np.stack(rob_exp_dist_l)) / np.sqrt(128)
    mean_rob_log_prob = np.mean(rob_log_prob_l)
    std_rob_log_prob = np.std(rob_log_prob_l) / np.sqrt(128)
    print('expected distance to true robot position: mean {}, std err {}'.format(mean_rob_exp_dist, std_rob_exp_dist))
    print('negative logprob of distrib of robot position: mean {}, std err {}'.format(mean_rob_log_prob, std_rob_log_prob))

    mean_pos_exp_dist = np.mean(np.stack([np.stack(pos0_exp_dist_l),
                                          np.stack(pos1_exp_dist_l)]))
    std_pos_exp_dist = np.std(np.stack([np.stack(pos0_exp_dist_l),
                                          np.stack(pos1_exp_dist_l)])) / np.sqrt(128)

    mean_pos_log_prob = np.mean(np.stack([np.stack(pos0_log_prob_l),
                                          np.stack(pos1_log_prob_l)]))

    std_pos_log_prob = np.mean(np.stack([np.stack(pos0_log_prob_l),
                                          np.stack(pos1_log_prob_l)])) / np.sqrt(128)

    print('averaged expected distance to true object position mean {}, std error {}'.format(mean_pos_exp_dist, std_pos_exp_dist))
    print('negative averaged negative logprob of distribtion of ob position {}, std error {}'.format(mean_pos_log_prob, std_pos_log_prob))

    with open(conf['output_dir'] + '/metric.txt', 'w+') as f:
        f.write('averages over batchsize {} \n'.format(conf['batch_size']))
        f.write('robot position \n')
        f.write('expected distance: mean {}, std err {} \n'.format(mean_rob_exp_dist, std_rob_exp_dist))
        f.write('negative logprob of distrib over positions {}, std errr {}\n'.format(mean_rob_log_prob, std_rob_log_prob))
        f.write('-----------\n')
        f.write('object positions \n')
        f.write('expected distance: mean {}, std err {} \n'.format(mean_pos_exp_dist, std_pos_exp_dist))
        f.write('negative logprob of distrib over positions {} stderr {} \n'.format(mean_pos_log_prob, std_pos_log_prob))

    pickle.dump({'pos0_exp_dist_l':pos0_exp_dist_l,
                  'pos1_exp_dist_l':pos1_exp_dist_l
                  }, open(conf['output_dir'] + '/metric_values.pkl', 'wb'))

    num_ex = 10
    fb_images = fb_images[:num_ex]
    if not create_images:
        fb_images = add_crosshairs(fb_images, fb_robot_pos, np.array([1,1,0]))
    fb_images = add_crosshairs(fb_images, fb_object_pos[:,:,0], np.array([1, 0, 1]))
    if not create_images:
        fb_images = add_crosshairs(fb_images, object_pos[:,:,1], np.array([0, 1, 1]))
    dict = OrderedDict()
    dict['ground_truth'] = fb_images

    fb_rob_gen_images = [r[:num_ex] for r in fb_rob_gen_images]
    dict['gen_images'] = fb_rob_gen_images

    if not create_images:
        fb_rob_gen_distrib = [r[:num_ex] for r in fb_rob_gen_distrib]
        dict['gen_distrib_robot_pos'] = fb_rob_gen_distrib
        # dict['rob_gen_distrib_overlay'] = compute_overlay(conf, rob_gen_images, rob_gen_distrib)

    fb_pos0_gen_distrib = [r[:num_ex] for r in fb_pos0_gen_distrib]
    dict['gen_distrib_object0'] = fb_pos0_gen_distrib
    # dict['rob_gen_distrib_overlay'] = compute_overlay(conf, pos0_gen_images, pos0_gen_distrib)

    if not create_images:
        fb_pos1_gen_distrib = [r[:num_ex] for r in fb_pos1_gen_distrib]
        dict['gen_distrib_object1'] = fb_pos1_gen_distrib
        # dict['rob_gen_distrib_overlay'] = compute_overlay(conf, pos1_gen_images, pos1_gen_distrib)

    flow = [r[:num_ex] for r in flow]
    dict['flow'] = flow

    file_path = conf['output_dir']
    dict['exp_name'] = conf['experiment_name']

    pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
    print('written files to:' + file_path)

    suf = '_metric_l{}_images'.format(conf['sequence_length']) if create_images else '_metric_l{}'.format(conf['sequence_length'])

    if not create_images:
        v = Visualizer_tkinter(dict, numex=num_ex, append_masks=True,
                               filepath=conf['output_dir'],
                               suf=suf,
                               renorm_heatmaps=True)
        v.build_figure()

    if create_images:
        for i in range(num_ex):
            Image_Creator(i, dict_=dict, filepath=conf['output_dir'])


def compute_exp_distance(sess, conf, model, true_pos, images, actions, endeff):
    statedim = conf['sdim']

    try:
        feed_dict = {model.iter_num: 0}
    except AttributeError:
        feed_dict = {}

    one_hot = create_one_hot(conf, true_pos[:,0], batch_mode=True)  # use true_pos at t0
    one_hot = np.concatenate(one_hot, axis=0)

    feed_dict[model.pix_distrib_pl] = one_hot

    sel_state = endeff[:,:2]
    start_states = np.concatenate([sel_state, np.zeros((conf['batch_size'], conf['sequence_length'] - 2, statedim))], axis=1)
    feed_dict[model.states_pl] = start_states

    sel_img = images[:,:2]
    start_images = np.concatenate([sel_img, np.zeros((conf['batch_size'], conf['sequence_length'] - 2, conf['img_height'], conf['img_width'], 3))], axis= 1)
    feed_dict[model.images_pl] = start_images

    feed_dict[model.actions_pl] = actions


    gen_images, gen_distrib, flow = sess.run([model.gen_images,
                                              model.gen_distrib,
                                              model.prediction_flow,
                                              ]
                                              , feed_dict)
    # visualize_annotation(conf, images[0], true_pos[0])
    # plt.figure()
    # plt.imshow(np.squeeze(one_hot[0,0]))
    # plt.show()

    print('calc exp dist')
    assert gen_distrib.shape[2] == 1
    gen_distrib = gen_distrib[:,:,0]
    exp_dist = calc_exp_dist(conf, gen_distrib, true_pos)
    print('calc log prob')
    log_prob = calc_log_prob(conf, gen_distrib, true_pos)

    return gen_images, gen_distrib, exp_dist, log_prob, flow

def calc_log_prob(conf, gen_distrib, true_pos):
    log_prob = np.zeros((conf['batch_size'], conf['sequence_length'] - 1))
    for b in range(conf['batch_size']):
        for tstep in range(conf['sequence_length'] - 1):
            distrib = gen_distrib[tstep][b].squeeze() / (np.sum(gen_distrib[tstep][b]) + 1e-7)
            pos = true_pos[b, tstep]
            log_prob[b, tstep] = -np.log(distrib[pos[0], pos[1]] + 1e-7)
    return log_prob

def calc_exp_dist(conf, gen_distrib, true_pos):
    expected_distance = np.zeros((conf['batch_size'],conf['sequence_length'] - 1))

    #discard the first true pos because there is no prediction for it:
    true_pos = true_pos[:,1:]

    pre_comp_distance = get_precomp_dist()

    for b in range(conf['batch_size']):
        for tstep in range(conf['sequence_length'] - 1):

            gen = gen_distrib[tstep][b].squeeze() / (np.sum(gen_distrib[tstep][b]) + 1e-7)

            # distance_grid = get_distancegrid(true_pos[b, tstep])
            distance_grid_fast = get_distance_fast(pre_comp_distance, true_pos[b, tstep])
            expected_distance[b, tstep] = np.sum(np.multiply(gen, distance_grid_fast))

            # plt.subplot(131)
            # plt.imshow(distance_grid)
            # plt.subplot(132)
            # plt.imshow(gen)
            # plt.subplot(133)
            # plt.imshow(gen_images[tstep][b])
            # plt.show()
    return expected_distance

def get_distancegrid(goal_pix, conf):
    distance_grid = np.empty((conf['img_height'], conf['im_widht']))
    for i in range(conf['img_height']):
        for j in range(conf['im_widht']):
            pos = np.array([i, j])
            distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

    # plt.imshow(distance_grid, zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
    # plt.show()
    # pdb.set_trace()
    return distance_grid

def get_precomp_dist(conf):
    goal_pix = np.array([conf['img_height'], conf['img_width']])
    distance_grid = np.empty((conf['img_height']*2, conf['img_width']*2))
    for i in range(conf['img_height']*2):
        for j in range(conf['im_widht']*2):
            pos = np.array([i, j])
            distance_grid[i, j] = np.linalg.norm(goal_pix - pos)

    # plt.imshow(distance_grid, zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
    # plt.show()
    # pdb.set_trace()
    return distance_grid

def get_distance_fast(precomp, goal_pix, conf):
    topleft = np.array([conf['img_height'], conf['img_width']]) - goal_pix
    distance_grid = precomp[topleft[0]:topleft[0]+conf['img_height'], topleft[1]:topleft[1]+conf['img_width']]

    return distance_grid


def add_crosshairs(images, pos, color=None):
    """
    :param images: shape: batch, t, r, c, 3 or list of lenght t with batch, r, c, 3
    :param pos: shape: batch, t, 2
    :param color: color needs to be vector with in [0,1]
    :return: images in the same shape as input
    """
    assert len(pos.shape) == 3

    if isinstance(images, list):
        make_list_output = True
        images = np.stack(images, axis=1)
    else:
        make_list_output = False

    assert len(images.shape) == 5
    assert images.shape[0] == pos.shape[0]


    pos = np.clip(pos, np.zeros(2).reshape((1,1,2)),np.array(images.shape[2:4]).reshape((1,1,2)) -1)

    if color == None:
        if images.dtype == np.float32:
            color = np.array([0., 1., 1.], np.float32)
        else:
            color = np.array([0, 255, 255], np.uint8)

    out = np.zeros_like(images)
    for b in range(pos.shape[0]):
        for t in range(images.shape[1]):
            im = np.squeeze(images[b,t])
            p = pos[b,t].astype(np.int)
            im[p[0]-5:p[0]-2,p[1]] = color
            im[p[0]+3:p[0]+6, p[1]] = color

            im[p[0],p[1]-5:p[1]-2] = color

            im[p[0], p[1]+3:p[1]+6] = color

            im[p[0], p[1]] = color
            out[b, t] = im

    if make_list_output:
        out = np.split(out, images.shape[1], axis=1)
        out = [np.squeeze(el) for el in out]
    return out

def add_crosshairs_single(im, pos, color=None, thick=False):
    """
    :param im:  shape: r,c,3
    :param pos:
    :param color:
    :return:
    """

    if color is None:
        if im.dtype == np.float32:
            color = np.array([0., 1., 1.])
        else:
            color = np.array([0, 255, 255])
    assert isinstance(color, np.ndarray)
    assert len(im.shape) and im.shape[2] == 3

    pos = np.clip(pos, [0,0], np.array(im.shape[:2]) -1)

    p = pos.astype(np.int)
    if thick:
        thickness = 1
        im[p[0]-8:p[0]-2,p[1] - thickness:p[1] + thickness + 1] = color
        im[p[0]+3:p[0]+9,p[1] - thickness:p[1] + thickness + 1] = color

        im[p[0] - thickness:p[0] + thickness + 1 ,p[1]-8:p[1]-2] = color
        im[p[0] - thickness:p[0] + thickness + 1 , p[1]+3:p[1]+9] = color
    else:
        im[p[0] - 5:p[0] - 2, p[1]] = color
        im[p[0] + 3:p[0] + 6, p[1]] = color

        im[p[0], p[1] - 5:p[1] - 2] = color

        im[p[0], p[1] + 3:p[1] + 6] = color

    im[p[0], p[1]] = color

    return im

def visualize_annotation(conf, images, pos):
    for t in range(conf['sequence_length']):
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.set_xlim(0, 63)
        ax.set_ylim(63, 0)

        plt.imshow(images[t])
        ax.scatter(pos[t][1], pos[t][0], s=20, marker="D", facecolors='r')
        plt.show()
