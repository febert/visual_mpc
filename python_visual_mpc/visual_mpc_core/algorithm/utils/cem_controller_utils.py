import numpy as np
import os
import pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import copy
import pdb
import time


def save_track_pkl(ctrl, t, cem_itr):

    pix_pos_dict = {}
    pix_pos_dict['desig_pix_t0'] = ctrl.desig_pix_t0
    pix_pos_dict['goal_pix'] = ctrl.goal_pix
    pix_pos_dict['desig'] = ctrl.desig_pix
    if ctrl.reg_tradeoff is not None:
        pix_pos_dict['reg_tradeoff'] = ctrl.reg_tradeoff
    dir = ctrl.agentparams['record'] + '/plan'
    if not os.path.exists(dir):
        os.makedirs(dir)
    pickle.dump(pix_pos_dict, open(dir + 'pix_pos_dict{}iter{}.pkl'.format(ctrl.t, cem_itr), 'wb'))


def make_blockdiagonal(cov, nactions, adim):
    mat = np.zeros_like(cov)
    for i in range(nactions-1):
        mat[i*adim:i*adim + adim*2, i*adim:i*adim + adim*2] = np.ones([adim*2, adim*2])
    # plt.switch_backend('TkAgg')
    # plt.imshow(mat)
    # plt.show()
    newcov = cov*mat
    return newcov

def standardize_and_tradeoff(flow_sc, warp_sc, flow_warp_tradeoff):
    """
    standardize cost vectors ca and cb, compute new scores weighted by tradeoff factor
    :param ca:
    :param cb:
    :return:
    """
    stand_flow_scores = (flow_sc - np.mean(flow_sc)) / (np.std(flow_sc) + 1e-7)
    stand_ws_costs = (warp_sc - np.mean(warp_sc)) / (np.std(warp_sc) + 1e-7)
    w = flow_warp_tradeoff
    return stand_flow_scores * w + stand_ws_costs * (1 - w)


def compute_warp_cost(logger, policyparams, flow_field, goal_pix=None, warped_images=None, goal_image=None, goal_mask=None):
    """
    :param flow_field:  shape: batch, time, r, c, 2
    :param goal_pix: if not None evaluate flowvec only at position of goal pix
    :return:
    """
    tc1 = time.time()
    flow_mags = np.linalg.norm(flow_field, axis=4)
    logger.log('tc1 {}'.format(time.time() - tc1))

    tc2 = time.time()
    if 'compute_warp_length_spot' in policyparams:
        flow_scores = []
        for t in range(flow_field.shape[1]):
            flow_scores_t = 0
            for ob in range(goal_pix.shape[0]):
                flow_scores_t += flow_mags[:,t,goal_pix[ob, 0], goal_pix[ob, 1]]
            flow_scores.append(np.stack(flow_scores_t))
        flow_scores = np.stack(flow_scores, axis=1)
        logger.log('evaluating at goal point only!!')
    elif goal_mask is not None:
        flow_scores = flow_mags*goal_mask[None, None,:,:]
        #compute average warp-length per per pixel which is part
        # of the object of interest i.e. where the goal mask is 1
        flow_scores = np.sum((flow_scores).reshape([flow_field.shape[0], flow_field.shape[1], -1]), -1)/np.sum(goal_mask)
    else:
        flow_scores = np.mean(np.mean(flow_mags, axis=2), axis=2)

    logger.log('tc2 {}'.format(time.time() - tc2))

    per_time_multiplier = np.ones([1, flow_scores.shape[1]])
    per_time_multiplier[:, -1] = policyparams['finalweight']

    if 'warp_success_cost' in policyparams:
        logger.log('adding warp warp_success_cost')
        if goal_mask is not None:
            diffs = (warped_images - goal_image[:, None])*goal_mask[None, None, :, :, None]
            #TODO: check this!
            ws_costs = np.sum(sqdiffs.reshape([flow_field.shape[0], flow_field.shape[1], -1]), axis=-1)/np.sum(goal_mask)
        else:
            ws_costs = np.mean(np.mean(np.mean(np.square(warped_images - goal_image[:,None]), axis=2), axis=2), axis=2)*\
                                        policyparams['warp_success_cost']

        flow_scores = np.sum(flow_scores*per_time_multiplier, axis=1)
        ws_costs = np.sum(ws_costs * per_time_multiplier, axis=1)
        stand_flow_scores = (flow_scores - np.mean(flow_scores)) / (np.std(flow_scores) + 1e-7)
        stand_ws_costs = (ws_costs - np.mean(ws_costs)) / (np.std(ws_costs) + 1e-7)
        w = policyparams['warp_success_cost']
        scores = stand_flow_scores*(1-w) + stand_ws_costs*w
    else:
        scores = np.sum(flow_scores*per_time_multiplier, axis=1)

    logger.log('tcg {}'.format(time.time() - tc1))
    return scores

def construct_initial_sigma(hp, adim, t=None):
    xy_std = hp.initial_std
    diag = [xy_std**2, xy_std**2]

    if hp.action_order[0] is not None:
        diag = []
        for a in hp.action_order:
            if a == 'x' or a == 'y':
                diag.append(xy_std**2)
            elif a == 'z':
                diag.append(hp.initial_std_lift ** 2)
            elif a == 'theta':
                diag.append(hp.initial_std_rot ** 2)
            elif a == 'grasp':
                diag.append(hp.initial_std_grasp ** 2)
            else:
                raise NotImplementedError
    else:
        if adim >= 3:
            diag.append(hp.initial_std_lift ** 2)
        if adim >= 4:
            diag.append(hp.initial_std_rot ** 2)
        if adim == 5:
            diag.append(hp.initial_std_grasp ** 2)

    adim = len(diag)
    diag = np.tile(diag, hp.nactions)
    diag = np.array(diag)

    if 'reduce_std_dev' in hp:
        assert 'reuse_mean' in hp
        if t >= 2:
            print('reducing std dev by factor', hp.reduce_std_dev)
            # reducing all but the last repeataction in the sequence since it can't be reused.
            diag[:(hp.nactions - 1) * adim] *= hp.reduce_std_dev

    sigma = np.diag(diag)
    return sigma


def reuse_cov(sigma, adim, hp):
    assert hp.replan_interval == 3
    print('reusing cov form last MPC step...')
    sigma_old = copy.deepcopy(sigma)
    sigma = np.zeros_like(sigma)
    #reuse covariance and add a fraction of the initial covariance to it
    sigma[0:-adim,0:-adim] = sigma_old[adim:,adim: ] + \
                             construct_initial_sigma(hp)[:-adim, :-adim] * hp.reuse_cov
    sigma[-adim:, -adim:] = construct_initial_sigma(hp)[:adim, :adim]
    return sigma


def reuse_action(prev_action, hp):
    assert hp.replan_interval == 3
    print('reusing mean form last MPC step...')
    action = np.zeros_like(prev_action)
    action[:-1] = prev_action[1:]
    return action.flatten()


def apply_ag_epsilon(actions, state, hp, close_override=False, no_close_first_repeat = False):
    z_thresh, epsilon, norm = hp.autograsp_epsilon
    assert 0 <= epsilon <= 1, "epsilon should be a valid probability"

    z_dim, gripper_dim = 2, -1
    if hp.action_order[0] is not None:
        assert 'z' in hp.action_order and 'grasp' in hp.action_order, "Ap epsilon requires z and grasp action"
        for i, a in enumerate(hp.action_order):
            if a == 'grasp':
                gripper_dim = i
            elif a == 'z':
                z_dim = i

    cumulative_zs = np.cumsum(actions[:, :, z_dim] / norm, 1) + state[-1, z_dim]
    z_thresh_check = (cumulative_zs <= z_thresh).astype(np.float32) * 2 - 1
    first_close_pos = np.argmax(z_thresh_check, axis = 1)
    if close_override:
        actions[:, :, gripper_dim] = 1
    else:
        for i, p in enumerate(first_close_pos):
            pivot = p - p % hp.repeat    # ensure that pivots only occur on repeat boundry
            if no_close_first_repeat:
                pivot = max(pivot, hp.repeat)
            actions[i, :pivot, gripper_dim] = -1
            actions[i, pivot:, gripper_dim] = 1
    epsilon_vec = np.random.choice([-1, 1], size=actions.shape[:-1], p=[epsilon, 1 - epsilon])
    actions[:, :, gripper_dim] *= epsilon_vec

    return actions


def discretize(actions, M, naction_steps, discrete_ind):
    """
    discretize and clip between 0 and 4
    :param actions:
    :return:
    """
    for b in range(M):
        for a in range(naction_steps):
            for ind in discrete_ind:
                actions[b, a, ind] = np.clip(np.floor(actions[b, a, ind]), 0, 4)
    return actions

def truncate_movement(actions, hp):
    maxshift = hp.initial_std * 2

    if len(actions.shape) == 3:
        if hp.action_order[0] is not None:
            for i, a in enumerate(hp.action_order):
                if a == 'x' or a == 'y':
                    maxshift = hp.initial_std * 2
                elif a == 'theta':
                    maxshift = np.pi / 4
                else:
                    continue
                actions[:, :, i] = np.clip(actions[:, :, i], -maxshift, maxshift)
            return actions

        actions[:,:,:2] = np.clip(actions[:,:,:2], -maxshift, maxshift)  # clip in units of meters
        if actions.shape[-1] >= 4: # if rotation is enabled
            maxrot = np.pi / 4
            actions[:, :, 3] = np.clip(actions[:, :, 3], -maxrot, maxrot)

    elif len(actions.shape) == 2:
        if hp.action_order[0] is not None:
            for i, a in enumerate(hp.action_order):
                if a == 'x' or a == 'y':
                    maxshift = hp.initial_std * 2
                elif a == 'theta':
                    maxshift = np.pi / 4
                else:
                    continue
                actions[:, i] = np.clip(actions[:, i], -maxshift, maxshift)
            return actions

        actions[:,:2] = np.clip(actions[:,:2], -maxshift, maxshift)  # clip in units of meters
        if actions.shape[-1] >= 4: # if rotation is enabled
            maxrot = np.pi / 4
            actions[:, 3] = np.clip(actions[:, 3], -maxrot, maxrot)
    else:
        raise NotImplementedError
    return actions


def get_mask_trafo_scores(policyparams, gen_distrib, goal_mask):
    scores = []
    bsize = gen_distrib[0].shape[0]
    for t in range(len(gen_distrib)):
        score = np.abs(np.clip(gen_distrib[t], 0,1) - goal_mask[None, None, ..., None])
        score = np.mean(score.reshape(bsize, -1), -1)
        scores.append(score)
    scores = np.stack(scores, axis=1)
    per_time_multiplier = np.ones([1, len(gen_distrib)])
    per_time_multiplier[:, -1] = policyparams['finalweight']
    return np.sum(scores * per_time_multiplier, axis=1)

