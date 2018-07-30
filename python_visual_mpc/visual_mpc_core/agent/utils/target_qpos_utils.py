import numpy as np

def get_target_qpos(target_qpos, _hyperparams, mj_U, t, gripper_up, gripper_closed, t_down, gripper_zpos, touch_sensors=None):
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

        z_height = _hyperparams['targetpos_clip'][1][2] - _hyperparams['targetpos_clip'][0][2]
        zthresh = _hyperparams['targetpos_clip'][0][2] + z_height * _hyperparams['autograsp']['zthresh']

        if gripper_zpos < zthresh:
            gripper_closed = True

        if 'reopen' in _hyperparams['autograsp'] and 'robot_name' in _hyperparams:
            if 'reopen' in _hyperparams['autograsp']:
                robot_sensor = touch_sensors[t, 0]         #robots only have one sensor
                if t > 0:
                    robot_sensor = max(touch_sensors[t, 0], touch_sensors[t - 1, 0])  #add some smoothing to fix gripper 0 force bug

                touch_threshold = _hyperparams['autograsp']['touchthresh']  # if touchthresh not specified never re-open
                if robot_sensor <= touch_threshold and gripper_zpos > zthresh:
                    gripper_closed = False
        elif 'reopen' in _hyperparams['autograsp']:
            touch_threshold = _hyperparams['autograsp']['touchthresh']   #if touchthresh not specified never re-open
            if touch_sensors[t, 0] <= touch_threshold and touch_sensors[t, 1] <= touch_threshold and gripper_zpos > zthresh:
                gripper_closed = False
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