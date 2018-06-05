import numpy as np

def get_target_qpos(target_qpos, _hyperparams, mj_U, t, gripper_up, gripper_closed, t_down, gripper_zpos):
    adim = _hyperparams['adim']

    target_qpos = target_qpos.copy()
    mj_U = mj_U.copy()

    if 'discrete_adim' in _hyperparams:
        up_cmd = mj_U[2]
        assert np.floor(up_cmd) == up_cmd
        if up_cmd != 0:
            t_down = t + up_cmd
            target_qpos[2] = _hyperparams['targetpos_clip'][1][2]
            gripper_up = True
        if gripper_up:
            if t == t_down:
                target_qpos[2] = _hyperparams['targetpos_clip'][0][2]
                gripper_up = False
        target_qpos[:2] += mj_U[:2]
        if adim == 4:
            target_qpos[3] += mj_U[3]
        assert adim <= 4
    elif 'close_once_actions' in _hyperparams:
        assert adim == 5
        target_qpos[:4] = mj_U[:4] + target_qpos[:4]
        grasp_thresh = 0.5
        if mj_U[4] > grasp_thresh:
            gripper_closed = True
        if gripper_closed:
            target_qpos[4] = 0.1
        else:
            target_qpos[4] = 0.0
        # print('target_qpos', target_qpos)
    elif 'autograsp' in _hyperparams:
        assert adim == 5
        target_qpos[:4] = mj_U[:4] + target_qpos[:4]
        if gripper_zpos < _hyperparams.get('autograsp_thresh', -0.06):
            gripper_closed = True
        if gripper_closed:
            target_qpos[4] = 0.1
        else:
            target_qpos[4] = 0.0
        #print('target_qpos', target_qpos)

    else:
        mode_rel = _hyperparams['mode_rel']
        target_qpos = target_qpos + mj_U * mode_rel
        for dim in range(adim):
            if not mode_rel[dim]:  # set all action dimension that are absolute
                target_qpos[dim] = mj_U[dim]

    pos_clip = _hyperparams['targetpos_clip']
    target_qpos = np.clip(target_qpos, pos_clip[0], pos_clip[1])
    # print('mjU', mj_U)
    # print('targetqpos', target_qpos)
    return target_qpos, t_down, gripper_up, gripper_closed