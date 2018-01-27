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


class CollectGoalImageSim(Sim):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, config):
        Sim.__init__(self, config)

        self.num_ob = self.agentparams['num_objects']

        self.run()

    def _take_sample(self, sample_index):
        if "gen_xml" in self.agentparams:
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

        self.store_data(0, traj)
        self.move_objects(traj)
        self.store_data(1, traj)

        # for t in range(self.agentparams['skip_first']):
        #     for _ in range(20):
        #         self.agent._model.data.ctrl = np.zeros(self.agentparams['adim'])
        #         self.agent._model.step()
                # self.agent.viewer.loop_once()

        # discarding trajecotries where an object falls out of the bin:
        end_zpos = [traj.Object_full_pose[-1, i, 2] for i in range(self.agentparams['num_objects'])]
        if any(zval < -2e-2 for zval in end_zpos):
            print 'object fell out!!!'
            traj_ok = False
        else:
            traj_ok = True

        return traj_ok, traj


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

    def move_objects(self, traj):

        new_poses = []
        for iob in range(self.num_ob):
            pos_disp = self.agentparams['pos_disp_range']
            angular_disp = self.agentparams['ang_disp_range']
            delta_pos = np.concatenate([np.random.uniform(-pos_disp, pos_disp, 2), np.zeros([1])])
            delta_alpha = np.random.uniform(-angular_disp, angular_disp)
            delta_rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_alpha)
            curr_quat =  Quaternion(traj.Object_full_pose[0, iob, 3:])
            newquat = delta_rot*curr_quat
            newpos = traj.Object_full_pose[0, iob][:3] + delta_pos
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
    CollectGoalImageSim(hyperparams.config)

if __name__ == "__main__":
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # print 'using seed', seed
    main()