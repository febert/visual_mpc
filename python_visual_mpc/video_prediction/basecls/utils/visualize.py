import numpy as np
import collections
import cPickle
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter

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
        one_hot[0, 0, desig_pix_l[0], desig_pix_l[1]] = 1.
        one_hot = np.repeat(one_hot, conf['context_frames'], axis=1)
        app_zeros = np.zeros((1, conf['sequence_length']- conf['context_frames'], 64, 64, 1), dtype=np.float32)
        one_hot = np.concatenate([one_hot, app_zeros], axis=1)
        one_hot = np.repeat(one_hot, conf['batch_size'], axis=0)

        return one_hot

def visualize(sess, m):
    conf = m.conf

    feed_dict = {m.lr: 0.0,
                 m.iter_num: 0,
                 m.train_cond: 1}

    file_path = conf['output_dir']

    ground_truth, gen_images, gen_masks, pred_flow = sess.run([m.images,
                                                                m.gen_images,
                                                                m.gen_masks,
                                                                m.prediction_flow,
                                                                ],
                                                               feed_dict)

    dict = collections.OrderedDict()
    import re
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    dict['iternum'] = itr_vis
    dict['ground_truth'] = ground_truth
    dict['gen_images'] = gen_images
    dict['prediction_flow'] = pred_flow
    # dict['gen_masks'] = gen_masks

    cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
    print 'written files to:' + file_path

    v = Visualizer_tkinter(dict, numex=conf['batch_size'], append_masks=True, gif_savepath=conf['output_dir'])
    v.build_figure()

def visualize_diffmotions(sess, m):
    conf = m.conf

    feed_dict = {m.iter_num: 0,
                 }

    b_exp, ind0 =0, 0

    img, state = sess.run([m.val_images, m.val_states])
    sel_img= img[b_exp,ind0:ind0+2]

    # c = Getdesig(sel_img[0], conf, 'b{}'.format(b_exp))
    # desig_pos_aux1 = c.coords.astype(np.int32)
    desig_pos_aux1 = np.array([30, 31])
    # print "selected designated position for aux1 [row,col]:", desig_pos_aux1

    one_hot = create_one_hot(conf, desig_pos_aux1)

    feed_dict[m.pix_distrib_pl] = one_hot

    sel_state = np.stack([state[b_exp,ind0],state[b_exp,ind0+1]], axis=0)

    start_states = np.concatenate([sel_state,np.zeros((conf['sequence_length']-2, 3))])
    start_states = np.expand_dims(start_states, axis=0)
    start_states = np.repeat(start_states, conf['batch_size'], axis=0)  # copy over batch
    feed_dict[m.states_pl] = start_states

    start_images = np.concatenate([sel_img,np.zeros((conf['sequence_length']-2, 64, 64, 3))])

    start_images = np.expand_dims(start_images, axis=0)
    start_images = np.repeat(start_images, conf['batch_size'], axis=0)  # copy over batch
    feed_dict[m.images_pl] = start_images

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

    feed_dict[m.actions_pl] = actions

    gen_images, gen_distrib, gen_masks, moved_parts, moved_images, moved_bckgd = sess.run([m.gen_images,
                                                    m.gen_distrib1,
                                                    m.gen_masks,
                                                    m.movd_parts_list,
                                                    m.moved_images,
                                                    m.moved_bckgd
                                                    ]
                                                   ,feed_dict)
    dict = {}
    dict['gen_images'] = gen_images
    dict['gen_masks'] = gen_masks
    dict['gen_distrib'] = gen_distrib
    import re
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    dict['iternum'] = itr_vis
    # dict['moved_parts'] = moved_parts
    # dict['moved_images'] = moved_images
    # dict['moved_bckgd'] = moved_bckgd

    file_path = conf['output_dir']
    cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
    print 'written files to:' + file_path

    v = Visualizer_tkinter(dict, numex=4, append_masks=False,
                           gif_savepath=conf['output_dir'],
                           suf='_diffmotions_b{}_l{}'.format(b_exp, conf['sequence_length']))
    v.build_figure()