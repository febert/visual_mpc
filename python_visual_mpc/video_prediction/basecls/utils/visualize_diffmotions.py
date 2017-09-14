import numpy as np


def visualize_diffmotions(sess, m):

    b_exp, ind0 =0, 0

    img, state = sess.run([m.val_images, m.val_states])
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