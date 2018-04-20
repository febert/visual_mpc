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


def make_visuals(ctrl, actions, bestindices, cem_itr, flow_fields, gen_distrib, gen_images, gen_states,
                 last_frames, goal_warp_pts_l, scores, warped_image_goal, warped_image_start, warped_images):

        seqlen = ctrl.netconf['sequence_length']
        bsize = ctrl.netconf['batch_size']

        if ctrl.save_subdir != None:
            file_path = ctrl.policyparams['current_dir'] + '/' + ctrl.save_subdir + '/verbose'
        else:
            file_path = ctrl.policyparams['current_dir'] + '/verbose'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        def best(inputlist):
            """
            get the K videos with the lowest cost
            """
            outputlist = [np.zeros_like(a)[:ctrl.K] for a in inputlist]
            for ind in range(ctrl.K):
                for tstep in range(len(inputlist)):
                    outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
            return outputlist

        def get_first_n(inputlist):
            return [inp[:ctrl.K] for inp in inputlist]

        sel_func = best
        t_dict_ = collections.OrderedDict()

        if 'warp_objective' in ctrl.policyparams:
            warped_images = image_addgoalpix(bsize, seqlen, warped_images, ctrl.goal_pix)
            gen_images = images_addwarppix(gen_images, goal_warp_pts_l, ctrl.goal_pix, ctrl.agentparams['num_objects'])
            warped_images = np.split(warped_images[bestindices[:ctrl.K]], warped_images.shape[1], 1)
            warped_images = list(np.squeeze(warped_images))
            t_dict_['warped_im_t{}'.format(ctrl.t)] = warped_images

        if 'register_gtruth' in ctrl.policyparams:
            current_image = [np.repeat(np.expand_dims(last_frames[0, 1], axis=0), ctrl.K, axis=0) for _ in
                           range(len(gen_images))]
            t_dict_['current_image'] = current_image
            t_dict_['warped_image_start '] = [np.repeat(np.expand_dims(warped_image_start.squeeze(), axis=0), ctrl.K, axis=0) for _ in
                range(len(gen_images))]

            startimages = [np.repeat(np.expand_dims(ctrl.start_image, axis=0), ctrl.K, axis=0) for _ in
                           range(len(gen_images))]
            desig_pix_t0 = np.tile(ctrl.desig_pix_t0[None, :], [ctrl.K, seqlen - 1, 1])
            t_dict_['start_image'] = add_crosshairs(startimages, desig_pix_t0)

            desig_pix_start = np.tile(ctrl.desig_pix[0][None, :], [bsize, seqlen - 1, 1])
            gen_images = add_crosshairs(gen_images, desig_pix_start, color=[1., 0., 0])
            desig_pix_goal = np.tile(ctrl.desig_pix[1][None, :], [bsize, seqlen - 1, 1])
            gen_images = add_crosshairs(gen_images, desig_pix_goal, color=[0, 0, 1.])

            t_dict_['warped_image_goal'] = [
                np.repeat(np.expand_dims(warped_image_goal.squeeze(), axis=0), ctrl.K, axis=0) for _ in
                range(len(gen_images))]
        goal_image = [np.repeat(np.expand_dims(ctrl.goal_image, axis=0), ctrl.K, axis=0) for _ in
                      range(len(gen_images))]
        goal_image_annotated = image_addgoalpix(bsize, seqlen, goal_image, ctrl.goal_pix)
        t_dict_['goal_image'] = goal_image_annotated
        if 'use_goal_image' not in ctrl.policyparams or 'comb_flow_warp' in ctrl.policyparams or 'register_gtruth' in ctrl.policyparams:
            for p in range(ctrl.ndesig):
                gen_distrib_p = [g[:, p] for g in gen_distrib]
                sel_gen_distrib_p = sel_func(gen_distrib_p)
                t_dict_['gen_distrib{}_t{}'.format(p, ctrl.t)] = sel_gen_distrib_p
                t_dict_['gen_distrib_goalim_overlay{}_t{}'.format(p, ctrl.t)] = (image_addgoalpix(bsize, seqlen, goal_image,
                                                                                                 ctrl.goal_pix[p]), sel_gen_distrib_p)
        t_dict_['gen_images_t{}'.format(ctrl.t)] = sel_func(gen_images)
        print('itr{} best scores: {}'.format(cem_itr, [scores[bestindices[ind]] for ind in range(ctrl.K)]))
        ctrl.dict_.update(t_dict_)
        if 'no_instant_gif' not in ctrl.agentparams:
            v = Visualizer_tkinter(t_dict_, append_masks=False,
                                   filepath=ctrl.agentparams['record'] + '/plan/',
                                   numex=ctrl.K, suf='t{}iter_{}'.format(ctrl.t, cem_itr))
            # v.build_figure()
            v.make_direct_vid()

            start_frame_conc = np.concatenate([last_frames[0, 0], last_frames[0, 1]], 0).squeeze()
            start_frame_conc = (start_frame_conc*255.).astype(np.uint8)
            Image.fromarray(start_frame_conc).save(ctrl.agentparams['record'] + '/plan/start_frame{}iter_{}.png'.format(ctrl.t, cem_itr))

            make_state_action_summary(ctrl.K, actions, ctrl.agentparams, bestindices, cem_itr, gen_states, seqlen, ctrl.t)
            if 'warp_objective' in ctrl.policyparams:
                t_dict_['warp_pts_t{}'.format(ctrl.t)] = sel_func(goal_warp_pts_l)
                t_dict_['flow_fields{}'.format(ctrl.t)] = flow_fields[bestindices[:K]]

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
