# creates a collection of random configurations for pushing
import numpy as np
import random
import pickle
import argparse
import os
import python_visual_mpc
import imp
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
from python_visual_mpc.visual_mpc_core.infrastructure.sim import Sim
from pyquaternion import Quaternion
import cv2
import copy

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pdb

from python_visual_mpc.visual_mpc_core.agent.general_agent import Image_dark_except

class CollectGoalImageSim(Sim):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, config):
        Sim.__init__(self, config)
        self.num_ob = self.agentparams['num_objects']

    def take_sample(self, sample_index):
        if "gen_xml" in self.agentparams:
            if sample_index % self.agentparams['gen_xml'] == 0:
                self.agent._setup_world()

        traj_ok = False
        i_trial = 0
        imax = 20
        while not traj_ok and i_trial < imax:
            i_trial += 1
            try:
                traj_ok, traj = self.rollout()
            except Image_dark_except:
                traj_ok = False

        if self._hyperparams['save_data']:
            self.save_data(traj, sample_index)

    def rollout(self):
        traj = Trajectory(self.agentparams)
        self.agent.large_images_traj = []
        self.agent.large_images = []

        self.agent._init()

        if 'gen_xml' in self.agentparams:
            traj.obj_statprop = self.agent.obj_statprop

        # apply action of zero for the first few steps, to let the scene settle
        for t in range(self.agentparams['skip_first']):
            for _ in range(self.agentparams['substeps']):
                ctrl = np.zeros(self.agentparams['adim'])
                if 'posmode' in self.agentparams:
                    #keep gripper at default x,y positions
                    ctrl[:3] = self.agent.sim.data.qpos[:3].squeeze()
                self.agent.sim.data.ctrl[:] = ctrl
                self.agent.sim.step()

        for t in range(self.agentparams['T']-1):
            self.store_data(t, traj)
            if 'make_gtruth_flows' in self.agentparams:
                traj.ob_masks[t], traj.arm_masks[t], traj.large_ob_masks[t], traj.large_arm_masks[t] = self.get_obj_masks()
                if t > 0:
                    traj.bwd_flow[t-1] = self.compute_gtruth_flow(t, traj)
            self.move_objects(t, traj)
        t += 1
        self.store_data(t, traj)
        if 'make_gtruth_flows' in self.agentparams:
            traj.ob_masks[t], traj.arm_masks[t], traj.large_ob_masks[t], traj.large_arm_masks[t] = self.get_obj_masks()
            traj.bwd_flow[t - 1] = self.compute_gtruth_flow(t, traj)

        if 'goal_mask' in self.agentparams:
            traj.goal_mask, _, _, _ = self.get_obj_masks()

        # discarding trajecotries where an object falls out of the bin:
        end_zpos = [traj.Object_full_pose[-1, i, 2] for i in range(self.agentparams['num_objects'])]
        if any(zval < -2e-2 for zval in end_zpos):
            print('object fell out!!!')
            traj_ok = False
        else:
            traj_ok = True

        return traj_ok, traj


    def compute_gtruth_flow(self, t, traj):
        """
        computes backward flow field, form I1 to I0 (which could be used to warp I0 to I1)
        :param t:
        :param traj:
        :return:
        """
        # loop through pixels of object:
        l_img_height = self.agentparams['viewer_image_height']
        l_img_width = self.agentparams['viewer_image_width']

        flow_field = np.zeros([l_img_height, l_img_width, 2])

        #compte flow for arm:
        prev_pose = np.concatenate([traj.X_full[t-1], np.array((0.0, 0.0, 0.0, 1.0))])
        curr_pose = np.concatenate([traj.X_full[t], np.array((0.0, 0.0, 0.0, 1.0))])
        curr_mask = traj.large_arm_masks[t]
        self.comp_flow_perob(prev_pose, curr_pose, flow_field, t, curr_mask, traj)

        for ob in range(self.agentparams['num_objects']):
            prev_pose = traj.Object_full_pose[t - 1, ob]
            curr_pose = traj.Object_full_pose[t, ob]
            curr_mask = traj.large_ob_masks[t, ob]
            self.comp_flow_perob(prev_pose, curr_pose, flow_field, t, curr_mask, traj)

        flow_field_smallim = cv2.resize(flow_field, dsize=(self.agentparams['image_width'], self.agentparams['image_height']),
                                interpolation=cv2.INTER_AREA)*self.agentparams['image_width']/self.agentparams['viewer_image_width']
        return flow_field_smallim

    def comp_flow_perob(self, prev_pose, curr_pose, flow_field, t, ob_mask, traj):
        prev_pos = prev_pose[:3]
        prev_quat = Quaternion(prev_pose[3:])
        curr_pos = curr_pose[:3]
        curr_quat = Quaternion(curr_pose[3:])
        inds = np.stack(np.where(ob_mask != 0.0), 1)
        diff_quat = curr_quat.conjugate * prev_quat  # rotates vector form curr_quat to prev_quat
        for i in range(inds.shape[0]):
            coord = inds[i]
            abs_pos_curr_sys = self.agent.viewer.get_3D(coord[0], coord[1], traj.largedimage[t, coord[0], coord[1]])
            rel_pos_curr_sys = abs_pos_curr_sys - curr_pos
            rel_pos_curr_sys = Quaternion(scalar=.0, vector=rel_pos_curr_sys)
            rel_pos_prev_sys = diff_quat * rel_pos_curr_sys * diff_quat.conjugate
            abs_pos_prev_sys = prev_pos + rel_pos_prev_sys.elements[1:]
            pos_prev_sys_imspace = self.agent.viewer.project_point(abs_pos_prev_sys)
            flow_field[coord[0], coord[1]] = pos_prev_sys_imspace - coord
        # plt.figure()
        # plt.imshow(np.squeeze(flow_field[:, :, 0]))
        # plt.title('rowflow')
        # plt.figure()
        # plt.imshow(np.squeeze(flow_field[:, :, 1]))
        # plt.title('colflow')
        # visualize_corresp(t, flow_field, traj, inds)

    def get_image(self):
        self.agent.viewer.loop_once()
        img_string, width, height = self.agent.viewer.get_image()
        large_img = np.fromstring(img_string, dtype='uint8').reshape((height, width, 3))[::-1, :, :]
        # img = cv2.resize(large_img, dsize=(
        # self.agentparams['image_width'], self.agentparams['image_height']), interpolation=cv2.INTER_AREA)
        return large_img

    def get_obj_masks(self):
        complete_img = self.get_image()
        ob_masks = []
        large_ob_masks = []

        armmask = None
        large_armmask = None

        qpos = copy.deepcopy(self.agent._model.data.qpos)
        qpos[2] -= 10
        self.agent._model.data.qpos = qpos
        self.agent._model.data.ctrl = np.zeros(3)
        self.agent._model.step()
        img = self.get_image()
        mask = 1 - np.uint8(np.all(complete_img == img, axis=-1)) * 1
        qpos[2] += 10
        self.agent._model.data.qpos = qpos
        self.agent._model.data.ctrl = np.zeros(3)
        self.agent._model.step()
        self.agent._model.data.qpos = qpos
        self.agent.viewer.loop_once()
        large_armmask = mask
        mask = cv2.resize(mask, dsize=(self.agentparams['image_width'], self.agentparams['image_height']), interpolation=cv2.INTER_NEAREST)
        armmask = mask
        # plt.figure()
        # plt.imshow(np.squeeze(mask))
        # plt.title('armmask')
        # plt.show()

        for i in range(self.num_ob):
            qpos = copy.deepcopy(self.agent._model.data.qpos)
            qpos[3+2+i*7] -= 1
            self.agent._model.data.qpos = qpos
            self.agent._model.data.ctrl = np.zeros(3)
            self.agent._model.step()
            self.agent._model.data.qpos = qpos
            img = self.get_image()
            mask = 1 - np.uint8(np.all(complete_img == img, axis=-1))*1
            qpos[3 + 2 + i * 7] += 1
            self.agent._model.data.qpos = qpos
            self.agent._model.data.ctrl = np.zeros(3)
            self.agent._model.step()
            self.agent._model.data.qpos = qpos
            self.agent.viewer.loop_once()

            large_ob_masks.append(mask)
            mask = cv2.resize(mask, dsize=(self.agentparams['image_width'], self.agentparams['image_height']), interpolation=cv2.INTER_NEAREST)
            ob_masks.append(mask)

            # plt.figure()
            # plt.imshow(masks[-1])
            # plt.title('objectmask')
            # plt.show()
        ob_masks = np.stack(ob_masks, 0)
        large_ob_masks = np.stack(large_ob_masks, 0)

        return ob_masks, armmask, large_ob_masks, large_armmask

    def store_data(self, t, traj):
        qpos_dim = self.agent.sdim // 2  # the states contains pos and vel
        traj.X_full[t, :] = self.agent.sim.data.qpos[:qpos_dim].squeeze()
        traj.Xdot_full[t, :] = self.agent.sim.data.qvel[:qpos_dim].squeeze()
        traj.X_Xdot_full[t, :] = np.concatenate([traj.X_full[t, :], traj.Xdot_full[t, :]])
        assert self.agent.sim.data.qpos.shape[0] == qpos_dim + 7 * self.num_ob
        for i in range(self.num_ob):
            fullpose = self.agent.sim.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze()
            traj.Object_full_pose[t, i, :] = fullpose

        self.agent._store_image(t, traj)

    def move_objects(self, t, traj):
        if 'gen_new_goalpose' in self.agentparams:
            newobj_poses = self.gen_new_goalpose(t, traj)
        else:
            newobj_poses = self.agent.goal_obj_pose

        arm_disp_range = self.agentparams['arm_disp_range']
        arm_disp = np.random.uniform(-arm_disp_range, arm_disp_range, 2)

        new_armpos = traj.X_full[0].copy()
        new_armpos[:2] = new_armpos[:2] + arm_disp
        new_armpos = np.clip(new_armpos, -0.35, 0.35)

        new_q = np.concatenate([new_armpos, newobj_poses.flatten()])

        sim_state = self.agent.sim.get_state()
        sim_state.qpos[:] = new_q
        sim_state.qvel[:] = np.zeros_like(sim_state.qvel)
        self.agent.sim.set_state(sim_state)
        self.agent.sim.forward()

        return new_q

    def gen_new_goalpose(self, t, traj):
        new_poses = []
        for iob in range(self.num_ob):
            angular_disp = self.agentparams['ang_disp_range']
            delta_alpha = np.random.uniform(-angular_disp, angular_disp)

            delta_rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_alpha)
            curr_quat = Quaternion(traj.Object_full_pose[t, iob, 3:])
            newquat = delta_rot * curr_quat

            pos_ok = False
            while not pos_ok:
                if 'const_dist' in self.agentparams:
                    alpha = np.random.uniform(-np.pi, np.pi, 1)
                    d = self.agentparams['const_dist']
                    delta_pos = np.array([d * np.cos(alpha), d * np.sin(alpha), 0.])
                else:
                    pos_disp = self.agentparams['pos_disp_range']
                    delta_pos = np.concatenate([np.random.uniform(-pos_disp, pos_disp, 2), np.zeros([1])])
                newpos = traj.Object_full_pose[t, iob][:3] + delta_pos
                if 'lift_object' in self.agentparams:
                    newpos[2] = 0.15

                if np.any(newpos[:2] > 0.35) or np.any(newpos[:2] < -0.35):
                    pos_ok = False
                else:
                    pos_ok = True

            new_poses.append(np.concatenate([newpos, newquat.elements]))
        newobj_poses = np.concatenate(new_poses, axis=0)
        return newobj_poses


def visualize_corresp(t, flow, traj, mask_coords):
    plt.figure()
    ax1 = plt.subplot(143)
    ax2 = plt.subplot(144)

    largeim = traj.largeimage
    im0 = largeim[t-1]
    im1 = largeim[t]
    ax1.imshow(im0)
    ax2.imshow(im1)

    coordsA = "data"
    coordsB = "data"

    imheight = largeim.shape[1]
    imwidth = largeim.shape[2]

    num_samples = 10
    cols, rows = np.meshgrid(np.arange(imwidth), np.arange(imheight))
    pos = np.stack([rows, cols], 2)
    warp_pts = pos + np.squeeze(flow)

    coords = np.random.randint(0, mask_coords.shape[0], num_samples)
    on_object = [mask_coords[c] for c in coords]
    pts_output = on_object

    # pts_output = np.concatenate([pts_output, on_object], axis=0)

    for p in range(num_samples):
        pt_output = pts_output[p]
        sampled_location = warp_pts[pt_output[0], pt_output[1]].astype('uint32')
        # sampled_location = np.flip(sampled_location, 0)
        print("point in warped img", pt_output, "sampled location", sampled_location)
        con = ConnectionPatch(xyA=np.flip(pt_output, 0), xyB=np.flip(sampled_location, 0), coordsA=coordsA,
                              coordsB=coordsB,
                              axesA=ax2, axesB=ax1,
                              arrowstyle="<->", shrinkB=5, linewidth=1., color=np.random.uniform(0, 1., 3))
        ax2.add_artist(con)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='create goal configs')
    parser.add_argument('experiment', type=str, help='experiment name')

    args = parser.parse_args()
    exp_name = args.experiment

    basepath = os.path.abspath(python_visual_mpc.__file__)
    basepath = '/'.join(str.split(basepath, '/')[:-2])
    data_coll_dir = basepath + '/pushing_data/' + exp_name
    hyperparams_file = data_coll_dir + '/hyperparams.py'

    hyperparams = imp.load_source('hyperparams', hyperparams_file).config

    c =CollectGoalImageSim(hyperparams)
    c.run()

if __name__ == "__main__":
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # print 'using seed', seed
    main()