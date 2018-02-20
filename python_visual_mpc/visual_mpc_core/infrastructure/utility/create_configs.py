# creates a collection of random configurations for pushing
import numpy as np
import random
import cPickle
import argparse
import os
import python_visual_mpc
import imp
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
from python_visual_mpc.visual_mpc_core.infrastructure.run_sim import Sim
from pyquaternion import Quaternion
import cv2
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

class CollectGoalImageSim(Sim):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, config):
        Sim.__init__(self, config)

        self.num_ob = self.agentparams['num_objects']

    def _take_sample(self, sample_index):
        if "gen_xml" in self.agentparams:
            if sample_index % self.agentparams['gen_xml'] == 0:
                self.agent.viewer.finish()
                self.agent._setup_world()

        traj_ok = False
        i_trial = 0
        imax = 20
        while not traj_ok and i_trial < imax:
            i_trial += 1
            traj_ok, traj = self.take_sample()

        if self._hyperparams['save_data']:
            self.save_data(traj, sample_index)

    def take_sample(self):
        traj = Trajectory(self.agentparams)
        self.agent.large_images_traj = []
        self.agent.large_images = []

        self.agent.viewer.set_model(self.agent._model)
        self.agent.viewer.cam.camid = 0

        self.agent._init()

        if 'gen_xml' in self.agentparams:
            traj.obj_statprop = self.agent.obj_statprop

        # apply action of zero for the first few steps, to let the scene settle
        for t in range(self.agentparams['skip_first']):
            for _ in range(self.agentparams['substeps']):
                self.agent._model.data.ctrl = np.zeros(self.agentparams['adim'])
                self.agent._model.step()
                # self.agent.viewer.loop_once()

        for t in range(self.agentparams['T']-1):
            self.store_data(t, traj)
            traj.large_masks[t] = self.get_obj_masks()

            if t> 0:
                flow = self.compute_gtruth_flow(t, traj)

            self.move_objects(t, traj)
        t += 1
        self.store_data(t, traj)

        if 'goal_mask' in self.agentparams:
            traj.goal_mask = self.get_obj_masks(small=True)

        # discarding trajecotries where an object falls out of the bin:
        end_zpos = [traj.Object_full_pose[-1, i, 2] for i in range(self.agentparams['num_objects'])]
        if any(zval < -2e-2 for zval in end_zpos):
            print 'object fell out!!!'
            traj_ok = False
        else:
            traj_ok = True

        image_sums = np.sum(traj._sample_images.reshape([self.agentparams['T'], -1]), axis=-1)
        if any(image_sums<10):
            traj_ok = False
            print 'image black!'
        print image_sums

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

        flow_field = np.zeros([self.agentparams['num_objects'], l_img_height, l_img_width, 2])

        for ob in range(self.agentparams['num_objects']):
            prev_pos = traj.Object_full_pose[t-1, ob, :3]
            prev_quat = Quaternion(traj.Object_full_pose[t-1, ob,3:])
            curr_pos = traj.Object_full_pose[t, ob,:3]
            curr_quat = Quaternion(traj.Object_full_pose[t, ob,3:])

            inds = np.stack(np.where(traj.large_masks[t,ob]!=0.0), 1)

            diff_quat = curr_quat.conjugate * prev_quat  # rotates vector form curr_quat to prev_quat


            # plt.ion()
            # plt.figure()
            # plt.imshow(traj.largedimage[t])
            # plt.title('depthimage t1')
            #
            # plt.figure()
            # plt.imshow(traj.large_masks[t,0])
            # plt.title('masks t1')
            #
            # plt.figure()
            # plt.imshow(traj._sample_images[t])
            # plt.title('im1')
            #
            # plt.figure()
            # plt.imshow(traj._sample_images[t - 1])
            # plt.title('im0')
            #
            # plt.draw()

            for i in range(inds.shape[0]):
                coord = inds[i]
                # # begin debug
                # point_2d = self.agent.viewer.project_point(traj.Object_full_pose[t,0,:3], return_zval=True)
                # print 'Object pos',  traj.Object_full_pose[t,0,:3]
                # print 'proj point 2d', point_2d
                #
                # img = traj.largeimage[t] # verify desigpos
                # desig_pix = np.around(point_2d).astype(np.int)
                # # img = large_img
                # img[desig_pix[0]-1:desig_pix[0]+1, :] = np.array([255, 255, 255])
                # img[:, desig_pix[1]-1:desig_pix[1]+1] = np.array([255, 255, 255])
                # print 'desig_pix', desig_pix
                # plt.imshow(img)
                # plt.show()
                #
                # point_3d = self.agent.viewer.get_3D(point_2d[0], point_2d[1], point_2d[2])
                # print 'point 3d from float'
                # point_3d = self.agent.viewer.get_3D(desig_pix.astype(np.float32)[0], desig_pix.astype(np.float32)[1], traj.largedimage[t, desig_pix[0], desig_pix[1]])
                # print 'point 3d from integer'
                #end debug

                abs_pos_curr_sys = self.agent.viewer.get_3D(coord[0], coord[1], traj.largedimage[t, coord[0], coord[1]])
                rel_pos_curr_sys = abs_pos_curr_sys - curr_pos

                rel_pos_curr_sys = Quaternion(scalar= .0, vector=rel_pos_curr_sys)
                rel_pos_prev_sys = diff_quat*rel_pos_curr_sys*diff_quat.conjugate
                abs_pos_prev_sys = prev_pos + rel_pos_prev_sys.elements[1:]

                pos_prev_sys_imspace = self.agent.viewer.project_point(abs_pos_prev_sys)

                flow_field[ob, coord[0], coord[1]] = pos_prev_sys_imspace - coord

            plt.figure()
            plt.imshow(np.squeeze(flow_field[ob, :, :, 0]))
            plt.title('rowflow')

            plt.figure()
            plt.imshow(np.squeeze(flow_field[ob, :, :, 1]))
            plt.title('colflow')

            # plt.imshow(np.linalg.norm(flow_field[ob], axis =-1))

            visualize_corresp(t, flow_field, traj.largeimage, inds)


    def get_image(self):
        self.agent.viewer.loop_once()
        img_string, width, height = self.agent.viewer.get_image()
        large_img = np.fromstring(img_string, dtype='uint8').reshape((height, width, 3))[::-1, :, :]
        # img = cv2.resize(large_img, dsize=(
        # self.agentparams['image_width'], self.agentparams['image_height']), interpolation=cv2.INTER_AREA)
        return large_img

    def get_obj_masks(self, small=False):
        complete_img = self.get_image()
        masks = []
        for i in range(self.num_ob):
            qpos = copy.deepcopy(self.agent._model.data.qpos)
            qpos[3+2+i*7] -= 1
            self.agent._model.data.qpos = qpos
            self.agent._model.step()
            img = self.get_image()
            mask = 1 - np.uint8(np.all(complete_img == img, axis=-1))*1
            qpos[3 + 2 + i * 7] += 1
            self.agent._model.data.qpos = qpos

            if small:
                mask = cv2.resize(mask, dsize=(
                self.agentparams['image_width'], self.agentparams['image_height']), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)

            # plt.imshow(masks[-1])
            # plt.show()

        return np.stack(masks, 0)

    def store_data(self, t, traj):
        qpos_dim = self.agent.sdim / 2  # the states contains pos and vel
        traj.X_full[t, :] = self.agent._model.data.qpos[:qpos_dim].squeeze()
        traj.Xdot_full[t, :] = self.agent._model.data.qvel[:qpos_dim].squeeze()
        traj.X_Xdot_full[t, :] = np.concatenate([traj.X_full[t, :], traj.Xdot_full[t, :]])
        assert self.agent._model.data.qpos.shape[0] == qpos_dim + 7 * self.num_ob
        for i in range(self.num_ob):
            fullpose = self.agent._model.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze()
            traj.Object_full_pose[t, i, :] = fullpose

        self.agent._store_image(t, traj)

    def move_objects(self, t, traj):

        new_poses = []
        for iob in range(self.num_ob):
            pos_disp = self.agentparams['pos_disp_range']
            angular_disp = self.agentparams['ang_disp_range']

            delta_pos = np.concatenate([np.random.uniform(-pos_disp, pos_disp, 2), np.zeros([1])])
            # delta_pos = np.array([0.1, 0., 0.])
            # print 'const delta pos!!!!!!!!!!!!!!!!!!!! for test'

            delta_alpha = np.random.uniform(-angular_disp, angular_disp)
            # delta_alpha = 0.
            # print 'delta aplpha 0 !!!!!!!!!!!!!!!!!!!! for test'

            delta_rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_alpha)
            curr_quat =  Quaternion(traj.Object_full_pose[t, iob, 3:])
            newquat = delta_rot*curr_quat
            newpos = traj.Object_full_pose[t, iob][:3] + delta_pos
            newpos = np.clip(newpos, -0.35, 0.35)

            new_poses.append(np.concatenate([newpos, newquat.elements]))

        newobj_poses = np.concatenate(new_poses, axis=0)

        arm_disp_range = self.agentparams['arm_disp_range']
        arm_disp = np.concatenate([np.random.uniform(-arm_disp_range, arm_disp_range, 2), np.zeros([1])])

        new_armpos = traj.X_full[0] + arm_disp
        new_armpos = np.clip(new_armpos, -0.35, 0.35)

        new_q = np.concatenate([new_armpos, newobj_poses])

        self.agent._model.data.qpos = new_q
        self.agent._model.step()

        return new_q


def visualize_corresp(t, flow, largeim, mask_coords):
    plt.figure()
    ax1 = plt.subplot(143)
    ax2 = plt.subplot(144)

    im0 = largeim[t-1]
    im1 = largeim[t]
    ax1.imshow(im0)
    ax2.imshow(im1)

    coordsA = "data"
    coordsB = "data"
    # random pts

    imheight = largeim.shape[1]
    imwidth = largeim.shape[2]

    num_samples = 10

    cols, rows = np.meshgrid(np.arange(imwidth), np.arange(imheight))

    pos = np.stack([rows, cols], 2)

    warp_pts = pos + np.squeeze(flow)

    # row_inds = np.random.randint(0, imheight, size=(num_samples/2)).reshape((num_samples/2, 1))
    # col_inds = np.random.randint(0, imwidth, size=(num_samples/2)).reshape((num_samples/2, 1))
    # pts_output = np.concatenate([row_inds, col_inds], axis=1)

    coords = np.random.randint(0, mask_coords.shape[0], num_samples)
    on_object = [mask_coords[c] for c in coords]
    pts_output = on_object

    # pts_output = np.concatenate([pts_output, on_object], axis=0)

    for p in range(num_samples):
        pt_output = pts_output[p]
        sampled_location = warp_pts[pt_output[0], pt_output[1]].astype('uint32')
        # sampled_location = np.flip(sampled_location, 0)
        print "point in warped img", pt_output, "sampled location", sampled_location
        con = ConnectionPatch(xyA=np.flip(pt_output, 0), xyB=np.flip(sampled_location, 0), coordsA=coordsA,
                              coordsB=coordsB,
                              axesA=ax2, axesB=ax1,
                              arrowstyle="<->", shrinkB=5, linewidth=1., color=np.random.uniform(0, 1., 3))
        ax2.add_artist(con)
    # ax1.set_xlim(0, 128)
    # ax1.set_ylim(0, 128)
    # ax2.set_xlim(0, 128)
    # ax2.set_ylim(0, 128)
    # plt.draw()
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

    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    c =CollectGoalImageSim(hyperparams.config)
    c.run()

if __name__ == "__main__":
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # print 'using seed', seed
    main()