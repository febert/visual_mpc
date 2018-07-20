#!/usr/bin/env python
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.visual_mpc_client import Visual_MPC_Client

import os

class Sawyer_Data_Collector(Visual_MPC_Client):
    def __init__(self, *args, **kwargs):
        Visual_MPC_Client.__init__(self, *args, **kwargs)
        self.policy = self.policyparams['type'](self.agentparams, self.policyparams)

        if 'mode_rel' in self.agentparams:
            self.mode_rel = self.agentparams['mode_rel']
        else:
            self.mode_rel = np.array([True, True, True, True, False]),

        self.checkpoint_file = os.path.join(self.recorder.save_dir, 'checkpoint.txt')
        # self.rpn_tracker = RPN_Tracker(self.recorder_save_dir, self.recorder)
        self.rpn_tracker = None
        self.run_data_collection()


    def query_action(self, istep):
        return self.policy.act(None, istep)

    def apply_act(self, des_pos, mj_U, i_act):
        """
        :param des_pos:  the action range for the gripper is 0. (open) to 0.1 (close)
        :param mj_U:
        :param i_act:
        :return:
        """

        if 'discrete_adim' in self.agentparams:
            # when rotation is enabled
            posshift = mj_U[:2]
            up_cmd = mj_U[2]

            delta_rot = mj_U[3]
            close_cmd = mj_U[4]

            des_pos[3] += delta_rot
            des_pos[:2] += posshift

            if close_cmd != 0:
                if self.ctrl.sawyer_gripper:
                    self.topen = i_act + close_cmd
                    self.ctrl.gripper.close()
                    self.gripper_closed = True
                    des_pos[4] = 0.1

            if self.gripper_closed:
                if i_act == self.topen:
                    self.ctrl.gripper.open()
                    print('opening gripper')
                    self.gripper_closed = False
                    des_pos[4] = 0.0

            if up_cmd != 0:
                self.t_down = i_act + up_cmd
                des_pos[2] = self.lower_height + self.delta_up
                self.gripper_up = True

            going_down = False
            if self.gripper_up:
                if i_act == self.t_down:
                    des_pos[2] = self.lower_height
                    print('going down')
                    self.gripper_up = False
                    going_down = True

            return des_pos, going_down
        else:
            des_pos = mj_U + des_pos * self.mode_rel
            des_pos = des_pos.squeeze()
            going_down = False

        des_pos = self.truncate_pos(des_pos)  # make sure not outside defined region
        return des_pos, going_down


    def truncate_pos(self, pos):
        if 'targetpos_clip' in self.agentparams:
            pos_clip = self.agentparams['targetpos_clip']
        else:
            pos_clip = [[0.46, -0.17, self.lower_height, -np.pi * 2, 0.],
                        [0.83, 0.17, self.lower_height + self.delta_up, np.pi * 2, 0.1]]
        return np.clip(pos, pos_clip[0], pos_clip[1])


if __name__ == '__main__':
    mpc = Sawyer_Data_Collector(cmd_args=True)
