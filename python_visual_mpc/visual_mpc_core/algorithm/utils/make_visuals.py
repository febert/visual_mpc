import numpy as np
import os
import collections
from PIL import Image
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter


def image_addgoalpix(bsize, seqlen, image_l, goal_pix):
    goal_pix_ob = np.tile(goal_pix[None, :], [bsize, seqlen - 1, 1])
    return add_crosshairs(image_l, goal_pix_ob)

def images_addwarppix(gen_images, warp_pts_l, pix, num_objects):
    warp_pts_arr = np.stack(warp_pts_l, axis=1)
    for ob in range(num_objects):
        warp_pts_ob = warp_pts_arr[:, :, pix[ob, 0], pix[ob, 1]]
        gen_images = add_crosshairs(gen_images, np.flip(warp_pts_ob, 2))
    return gen_images


def make_visuals(tstep, actions, bestindices, cem_itr, flow_fields, gen_distrib, gen_images, gen_states, start_frame, goal_image,
                 goal_warp_pts_l, scores, warped_image_goal, warped_image_start, warped_images, desig_pix, desig_pix_t0, goal_pix,
                 agentparams, netconf, policyparams, K, ndesig, save_subdir, dict_):

        seqlen = netconf['sequence_length']
        bsize = netconf['batch_size']


        if save_subdir != None:
            file_path = netconf['current_dir'] + '/' + save_subdir + '/verbose'
        else:
            file_path = netconf['current_dir'] + '/verbose'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        def best(inputlist):
            """
            get the K videos with the lowest cost
            """
            outputlist = [np.zeros_like(a)[:K] for a in inputlist]
            for ind in range(K):
                for tstep in range(len(inputlist)):
                    outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
            return outputlist

        def get_first_n(inputlist):
            return [inp[:K] for inp in inputlist]

        sel_func = best
        t_dict_ = collections.OrderedDict()

        if 'warp_objective' in policyparams:
            warped_images = image_addgoalpix(bsize, seqlen, warped_images, goal_pix)
            gen_images = images_addwarppix(gen_images, goal_warp_pts_l, goal_pix, agentparams['num_objects'])
            warped_images = np.split(warped_images[bestindices[:K]], warped_images.shape[1], 1)
            warped_images = list(np.squeeze(warped_images))
            t_dict_['warped_im_t{}'.format(tstep)] = warped_images

        if 'register_gtruth' in policyparams:
            t_dict_['warped_image_start '] = [
                np.repeat(np.expand_dims(warped_image_start.squeeze(), axis=0), K, axis=0) for _ in
                range(len(gen_images))]

            startimage = [np.repeat(np.expand_dims(start_frame, axis=0), K, axis=0) for _ in
                          range(len(gen_images))]
            desig_pix_t0 = np.tile(desig_pix_t0[None, :], [K, seqlen - 1, 1])
            t_dict_['start_image'] = add_crosshairs(startimage, desig_pix_t0)

            desig_pix = np.tile(desig_pix[0][None, None, :], [bsize, seqlen - 1, 1])
            gen_images = add_crosshairs(gen_images, desig_pix, color=[0, 1., 0])
            desig_pix = np.tile(desig_pix[1][None, None, :], [bsize, seqlen - 1, 1])
            gen_images = add_crosshairs(gen_images, desig_pix, color=[1., 0, 0])

            t_dict_['warped_image_goal'] = [
                np.repeat(np.expand_dims(warped_image_goal.squeeze(), axis=0), K, axis=0) for _ in
                range(len(gen_images))]
        goal_image = [np.repeat(np.expand_dims(goal_image, axis=0), K, axis=0) for _ in
                      range(len(gen_images))]
        goal_image_annotated = image_addgoalpix(bsize, seqlen, goal_image, goal_pix)
        t_dict_['goal_image'] = goal_image_annotated
        if 'use_goal_image' not in policyparams or 'comb_flow_warp' in policyparams or 'register_gtruth' in policyparams:
            for p in range(ndesig):
                gen_distrib_p = [g[:, p] for g in gen_distrib]
                sel_gen_distrib_p = sel_func(gen_distrib_p)
                t_dict_['gen_distrib{}_t{}'.format(p, tstep)] = sel_gen_distrib_p
                t_dict_['gen_distrib_goalim_overlay{}_t{}'.format(p, tstep)] = (image_addgoalpix(bsize, seqlen, goal_image,
                                                                                                 goal_pix[p]), sel_gen_distrib_p)
        t_dict_['gen_images_t{}'.format(tstep)] = sel_func(gen_images)
        print('itr{} best scores: {}'.format(cem_itr, [scores[bestindices[ind]] for ind in range(K)]))
        dict_.update(t_dict_)
        if 'no_instant_gif' not in agentparams:
            v = Visualizer_tkinter(t_dict_, append_masks=False,
                                   filepath=agentparams['record'] + '/plan/',
                                   numex=K, suf='t{}iter_{}'.format(tstep, cem_itr))
            # v.build_figure()
            v.make_direct_vid()

            start_frame_conc = np.concatenate([start_frame[0,0], start_frame[0,1]], 0).squeeze()
            start_frame_conc = (start_frame_conc*255.).astype(np.uint8)
            Image.fromarray(start_frame_conc).save(agentparams['record'] + '/plan/start_frame{}iter_{}.png'.format(tstep, cem_itr))

            make_state_action_summary(K, actions, agentparams, bestindices, cem_itr, gen_states, seqlen, tstep)
            if 'warp_objective' in policyparams:
                t_dict_['warp_pts_t{}'.format(tstep)] = sel_func(goal_warp_pts_l)
                t_dict_['flow_fields{}'.format(tstep)] = flow_fields[bestindices[:K]]

        return gen_images


def make_state_action_summary(K, actions, agentparams, bestindices, cem_itr, gen_states, seqlen, tstep):
    gen_states = np.stack(gen_states, 1)
    with open(agentparams['record'] + '/plan/actions_states_t{}iter_{}'.format(tstep, cem_itr), 'w') as f:
        f.write('actions, states \n')
        for i in range(K):
            f.write('k{}\n'.format(i))
            for t_ in range(seqlen):
                if t_ == 0:
                    f.write('t{}  {}\n'.format(t_, actions[bestindices][i, t_]))
                else:
                    f.write(
                        't{}  {}  {}\n'.format(t_, actions[bestindices][i, t_], gen_states[bestindices][i, t_ - 1]))
