""" This file defines the linear Gaussian policy class. """
import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *
import copy
import time
import imp
import cPickle
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
from datetime import datetime
import os

from PIL import Image
import pdb


class CEM_controller(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, predictor = None):
        Policy.__init__(self)

        self.agentparams = ag_params
        self.policyparams = policyparams

        self.t = None

        if self.policyparams['low_level_ctrl']:
            self.low_level_ctrl = policyparams['low_level_ctrl']['type'](None, policyparams['low_level_ctrl'])

        self.model = mujoco_py.MjModel(self.agentparams['filename'])

        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False

        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

        if 'usenet' in self.policyparams:
            self.use_net = True
        else: self.use_net = False

        self.action_list = []

        if self.use_net:
            hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
            self.netconf = hyperparams.configuration

        self.nactions = self.policyparams['nactions']
        self.repeat = self.policyparams['repeat']
        if self.use_net:
            self.M = self.netconf['batch_size']
            assert self.nactions * self.repeat == self.netconf['sequence_length']
            self.predictor = predictor
            self.K = 10  # only consider K best samples for refitting
        else:
            self.M = self.policyparams['num_samples']
            self.K = 10  # only consider K best samples for refitting

        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]

        # the full horizon is actions*repeat
        if 'action_cost_factor' in self.policyparams:
            self.action_cost_mult = self.policyparams['action_cost_factor']
        else: self.action_cost_mult = 0.00005

        self.adim = 2  # action dimension
        self.initial_std = policyparams['initial_std']
        if 'exp_factor' in policyparams:
            self.exp_factor = policyparams['exp_factor']

        gofast = True
        self.viewer = mujoco_py.MjViewer(visible=True, init_width=480,
                                         init_height=480, go_fast=gofast)
        self.viewer.start()
        self.viewer.set_model(self.model)
        self.viewer.cam.camid = 0

        self.small_viewer = mujoco_py.MjViewer(visible=True, init_width=64,
                                         init_height=64, go_fast=gofast)
        self.small_viewer.start()
        self.small_viewer.set_model(self.model)
        self.small_viewer.cam.camid = 0

        self.init_model = []
        #history of designated pixels
        self.desig_pix = []

        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.target = np.zeros(2)

        self.rec_input_distrib =[]  # record the input distributions
        self.corr_gen_images = []
        self.corrector = None

        self.mean =None
        self.sigma =None

    def reinitialize(self):
        self.use_net = self.policyparams['usenet']
        self.action_list = []
        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]
        self.initial_std = self.policyparams['initial_std']
        # history of designated pixels
        self.desig_pix = []
        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices = []

        self.target = np.zeros(2)


    def finish(self):
        self.small_viewer.finish()
        self.viewer.finish()

    def setup_mujoco(self):

        # set initial conditions
        self.model.data.qpos = self.init_model.data.qpos
        self.model.data.qvel = self.init_model.data.qvel

    def eval_action(self):
        goalpoint = np.array(self.agentparams['goal_point'])
        refpoint = self.model.data.site_xpos[0,:2]

        return np.linalg.norm(goalpoint - refpoint)

    def calc_action_cost(self, actions):
        actions_costs = np.zeros(self.M)
        for smp in range(self.M):
            force_magnitudes = np.array([np.linalg.norm(actions[smp, t]) for
                                         t in range(self.nactions * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self.action_cost_mult
        return actions_costs

    def perform_CEM(self,last_frames, last_states, last_action, t):
        # initialize mean and variance

        if 'exp_factor' in self.policyparams:  # if reusing covariance matrix and mean...
            if t < 2:
                self.mean = np.zeros(self.adim * self.nactions)
                self.sigma = np.diag(np.ones(self.adim * self.nactions) * self.initial_std ** 2)
            else:
                print 'reusing mean form last MPC step...'
                # if 'reduce_iter' in self.policyparams:
                #     self.niter = 2

                pdb.set_trace()
                mean_old = copy.deepcopy(self.mean)

                self.mean = np.zeros_like(mean_old)
                self.mean[:-2] = mean_old[2:]
                self.mean = self.mean.reshape(self.adim * self.nactions)

                sigma_old = copy.deepcopy(self.sigma)
                self.sigma = np.zeros_like(self.sigma)
                self.sigma[0:-2,0:-2] = sigma_old[2:,2: ]
                self.sigma[0:-2, 0:-2] += np.diag(np.ones(self.adim * (self.nactions-1)))*(self.initial_std/5)**2
                self.sigma[-1,-1] = self.initial_std**2
                self.sigma[-2,-2] = self.initial_std**2

                # self.sigma = np.diag(np.ones(self.adim * self.nactions) * self.initial_std ** 2)
        else:
            self.mean = np.zeros(self.adim * self.nactions)
            self.sigma = np.diag(np.ones(self.adim * self.nactions) * self.initial_std ** 2)


        print '------------------------------------------------'
        print 'starting CEM cylce'

        # last_action = np.expand_dims(last_action, axis=0)
        # last_action = np.repeat(last_action, self.netconf['batch_size'], axis=0)
        # last_action = last_action.reshape(self.netconf['batch_size'], 1, self.adim)

        for itr in range(self.niter):
            print '------------'
            print 'iteration: ', itr
            t_startiter = datetime.now()

            actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
            actions = actions.reshape(self.M, self.nactions, self.adim)
            # import pdb; pdb.set_trace()

            if self.verbose or not self.use_net:
                scores = self.take_mujoco_smp(actions, itr)

            actions = np.repeat(actions, self.repeat, axis=1)

            t_start = datetime.now()

            if self.use_net:
                scores = self.video_pred(last_frames, last_states, actions, itr)
                print('overall time for evaluating actions {}'.format(
                    (datetime.now() - t_start).seconds + (datetime.now() - t_start).microseconds / 1e6))

            actioncosts = self.calc_action_cost(actions)
            scores += actioncosts

            self.indices = scores.argsort()[:self.K]
            self.bestindices_of_iter[itr] = self.indices

            self.bestaction_withrepeat = actions[self.indices[0]]

            actions = actions.reshape(self.M, self.nactions, self.repeat, self.adim)
            actions = actions[:,:,-1,:] #taking only one of the repeated actions
            actions_flat = actions.reshape(self.M, self.nactions * self.adim)

            self.bestaction = actions[self.indices[0]]
            # print 'bestaction:', self.bestaction

            arr_best_actions = actions_flat[self.indices]  # only take the K best actions
            self.sigma = np.cov(arr_best_actions, rowvar= False, bias= False)
            self.mean = np.mean(arr_best_actions, axis= 0)

            print 'iter {0}, bestscore {1}'.format(itr, scores[self.indices[0]])
            print 'action cost of best action: ', actioncosts[self.indices[0]]

            print 'overall time for iteration {}'.format(
                (datetime.now() - t_startiter).seconds + (datetime.now() - t_startiter).microseconds / 1e6)

    def take_mujoco_smp(self, actions, itr):

        scores = np.empty(self.M, dtype=np.float64)
        for smp in range(self.M):
            self.setup_mujoco()
            accum_score = self.sim_rollout(actions[smp], smp, itr)

            if 'rew_all_steps' in self.policyparams:
                scores[smp] = accum_score
            else:
                scores[smp] = self.eval_action()

        return scores


    def mujoco_one_hot_images(self):
        one_hot_images = np.zeros((1, self.netconf['context_frames'], 64, 64, 1), dtype=np.float32)
        # switch on pixels
        one_hot_images[0, 0, self.desig_pix[-2][0], self.desig_pix[-2][1]] = 1
        one_hot_images[0, 1, self.desig_pix[-1][0], self.desig_pix[-1][1]] = 1

        return one_hot_images

    def correct_distrib(self, full_images, t):
        full_images = full_images.astype(np.float32) / 255.
        if t == 0:  # use one hot image from mujoco
            desig_pos = self.desig_pix[-1]
            one_hot_image = np.zeros((1, 64, 64, 1), dtype=np.float32)
            # switch on pixels
            one_hot_image[0, desig_pos[0], desig_pos[1]] = 1

            self.rec_input_distrib.append(one_hot_image)
            self.corr_gen_images.append(np.expand_dims(full_images[0], axis=0))

        else:
            corr_input_distrib = self.rec_input_distrib[-1]

            input_images = full_images[t-1:t+1]
            input_images = np.expand_dims(input_images, axis= 0)
            gen_image, _, output_distrib = self.corrector(input_images, corr_input_distrib)
            self.rec_input_distrib.append(output_distrib)
            self.corr_gen_images.append(gen_image)

            if t == (self.agentparams['T']-1):
                self.save_distrib_visual(full_images)

    def save_distrib_visual(self, full_images, use_genimg = True):
        #assumes full_images is already rescaled to [0,1]
        orig_images = np.split(full_images, full_images.shape[0], axis = 0)
        orig_images = [im.reshape(1,64,64,3) for im in orig_images]

        # the first image of corr_gen_images is the first image of the original images!
        file_path =self.policyparams['current_dir'] + '/videos_distrib'
        if use_genimg:
            cPickle.dump([orig_images, self.corr_gen_images, self.rec_input_distrib, self.desig_pix],
                         open(file_path + '/correction.pkl', 'wb'))
            distrib = make_color_scheme(self.rec_input_distrib)
            distrib = add_crosshairs(distrib, self.desig_pix)
            frame_list = assemble_gif([orig_images, self.corr_gen_images, distrib], num_exp=1)
        else:
            cPickle.dump([orig_images, self.rec_input_distrib],
                         open(file_path + '/correction.pkl', 'wb'))
            distrib = make_color_scheme(self.rec_input_distrib)
            distrib = add_crosshairs(distrib, self.desig_pix)
            frame_list = assemble_gif([orig_images, distrib], num_exp=1)

        npy_to_gif(frame_list, self.policyparams['rec_distrib'])

    def mujoco_to_imagespace(self, mujoco_coord, numpix = 64, truncate = False):
        """
        convert form Mujoco-Coord to numpix x numpix image space:
        :param numpix: number of pixels of square image
        :param mujoco_coord:
        :return: pixel_coord
        """
        viewer_distance = .75  # distance from camera to the viewing plane
        window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
        pixelheight = window_height / numpix  # height of one pixel
        pixelwidth = pixelheight
        window_width = pixelwidth * numpix
        middle_pixel = numpix / 2
        pixel_coord = np.rint(np.array([-mujoco_coord[1], mujoco_coord[0]]) /
                              pixelwidth + np.array([middle_pixel, middle_pixel]))
        pixel_coord = pixel_coord.astype(int)

        if truncate:
            if np.any(pixel_coord < 0) or np.any(pixel_coord > numpix -1):
                print '###################'
                print 'designated pixel is outside the field!! Resetting it to be inside...'
                print 'truncating...'
                if np.any(pixel_coord < 0):
                    pixel_coord[pixel_coord < 0] = 0
                if np.any(pixel_coord > numpix-1):
                    pixel_coord[pixel_coord > numpix-1]  = numpix-1

        return pixel_coord


    def video_pred(self, last_frames, last_states, actions, itr):

        self.pred_pos[:, itr, 0] = self.mujoco_to_imagespace(last_states[-1, :2] , numpix=480)


        if 'predictor_propagation' in self.policyparams:  #using the predictor's DNA to propagate, no correction
            print 'using predictor_propagation'
            if self.t < self.netconf['context_frames']:
                input_distrib = self.mujoco_one_hot_images()
                if itr == 0:
                    self.rec_input_distrib.append(input_distrib[:,1])
            else:
                input_distrib = [self.rec_input_distrib[-2], self.rec_input_distrib[-1]]
                input_distrib = [np.expand_dims(elem, axis=1) for elem in input_distrib]
                input_distrib = np.concatenate(input_distrib, axis=1)

        else: input_distrib = self.mujoco_one_hot_images()

        last_states = np.expand_dims(last_states, axis=0)

        last_states = np.repeat(last_states, self.netconf['batch_size'], axis=0)

        input_distrib = np.repeat(input_distrib, self.netconf['batch_size'], axis=0)

        last_frames = np.expand_dims(last_frames, axis=0)
        last_frames = np.repeat(last_frames, self.netconf['batch_size'], axis=0)
        app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length']-
                                    self.netconf['context_frames'], 64, 64, 3))
        last_frames = np.concatenate((last_frames, app_zeros), axis=1)
        last_frames = last_frames.astype(np.float32)/255.

        gen_images, gen_distrib1, _,gen_states, gen_masks,  = self.predictor(input_images=last_frames,
                                                            input_state=last_states,
                                                            input_actions=actions,
                                                            input_one_hot_images1=input_distrib)

        for tstep in range(self.netconf['sequence_length']-1):
            for smp in range(self.M):
                self.pred_pos[smp, itr, tstep+1] = self.mujoco_to_imagespace(gen_states[tstep][smp, :2], numpix=480)

        goalpoint = self.mujoco_to_imagespace(self.agentparams['goal_point'])
        distance_grid = np.empty((64,64))
        for i in range(64):
            for j in range(64):
                pos = np.array([i,j])
                distance_grid[i,j] = np.linalg.norm(goalpoint - pos)

        expected_distance = np.zeros(self.netconf['batch_size'])
        if 'rew_all_steps' in self.policyparams:
            for tstep in range(self.netconf['sequence_length']-1):
                t_mult = 1
                if 'finalweight' in self.policyparams:
                    if tstep == self.netconf['sequence_length']-2:
                        t_mult = self.policyparams['finalweight']

                for b in range(self.netconf['batch_size']):
                    gen = gen_distrib1[tstep][b].squeeze() / np.sum(gen_distrib1[tstep][b])
                    expected_distance[b] += np.sum(np.multiply(gen, distance_grid)) * t_mult
        else:
            for b in range(self.netconf['batch_size']):
                gen = gen_distrib1[-1][b].squeeze()/ np.sum(gen_distrib1[-1][b])
                expected_distance[b] = np.sum(np.multiply(gen, distance_grid))

        # for predictor_propagation only!!
        if 'predictor_propagation' in self.policyparams:
            assert not 'correctorconf' in self.policyparams
            if itr == (self.policyparams['iterations']-1):
                # pick the prop distrib from the action actually chosen after the last iteration (i.e. self.indices[0])
                self.indices = expected_distance.argsort()[:self.K]
                self.rec_input_distrib.append(gen_distrib1[2][self.indices[0]].reshape(1, 64, 64, 1))

        # compare prediciton with simulation
        if self.verbose and itr == self.policyparams['iterations']-1:
            print 'creating visuals for best sampled actions at last iteration...'

            # concat_masks = [np.stack(gen_masks[t], axis=1) for t in range(14)]

            file_path = self.netconf['current_dir'] + '/verbose'

            bestindices = expected_distance.argsort()[:self.K]

            def best(inputlist):
                outputlist = [np.zeros_like(a)[:self.K] for a in inputlist]

                for ind in range(self.K):
                    for tstep in range(len(inputlist)):
                        outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
                return outputlist

            self.gtruth_images = [img.astype(np.float) / 255. for img in self.gtruth_images]  #[1:]
            cPickle.dump(best(gen_distrib1), open(file_path + '/gen_distrib.pkl', 'wb'))
            cPickle.dump(best(gen_images), open(file_path + '/gen_images.pkl', 'wb'))
            # cPickle.dump(best(concat_masks), open(file_path + '/gen_masks.pkl', 'wb'))
            cPickle.dump(best(self.gtruth_images), open(file_path + '/gtruth_images.pkl', 'wb'))
            print 'written files to:' + file_path
            comp_pix_distrib(file_path, name='check_eval_t{}'.format(self.t), masks=False, examples=self.K)

            f = open(file_path + '/actions_last_iter_t{}'.format(self.t), 'w')
            sorted = expected_distance.argsort()
            for i in range(actions.shape[0]):
                f.write('index: {0}, score: {1}, rank: {2}'.format(i, expected_distance[i], np.where(sorted == i)[0][0]))
                f.write('action {}\n'.format(actions[i]))

        return expected_distance

    def sim_rollout(self, actions, smp, itr):
        accum_score = 0

        if self.policyparams['low_level_ctrl']:
            rollout_ctrl = self.policyparams['low_level_ctrl']['type'](None, self.policyparams['low_level_ctrl'])
            roll_target_pos = copy.deepcopy(self.init_model.data.qpos[:2].squeeze())

        for hstep in range(self.nactions):
            currentaction = actions[hstep]

            if self.policyparams['low_level_ctrl']:
                roll_target_pos += currentaction

            for r in range(self.repeat):
                t = hstep*self.repeat + r
                # print 'time ',t, ' target pos rollout: ', roll_target_pos

                if not self.use_net:
                    ball_coord = self.model.data.qpos[:2].squeeze()
                    self.pred_pos[smp, itr, t] = mujoco_to_imagespace(ball_coord, numpix=480)
                    if self.policyparams['low_level_ctrl']:
                        self.rec_target_pos[smp, itr, t] = mujoco_to_imagespace(roll_target_pos, numpix=480)

                if self.policyparams['low_level_ctrl'] == None:
                    force = currentaction
                else:
                    qpos = self.model.data.qpos[:2].squeeze()
                    qvel = self.model.data.qvel[:2].squeeze()
                    force = rollout_ctrl.act(qpos, qvel, None, t, roll_target_pos)

                for _ in range(self.agentparams['substeps']):
                    self.model.data.ctrl = force
                    self.model.step()  # simulate the model in mujoco

                accum_score += self.eval_action()

                if self.verbose:
                    self.viewer.loop_once()

                    self.small_viewer.loop_once()
                    img_string, width, height = self.small_viewer.get_image()
                    img = np.fromstring(img_string, dtype='uint8').reshape(
                        (height, width, 3))[::-1, :, :]
                    self.gtruth_images[t][smp] = img
                    # self.check_conversion()

        return accum_score

    def check_conversion(self):
        # check conversion
        img_string, width, height = self.viewer.get_image()
        img = np.fromstring(img_string, dtype='uint8').reshape(
            (height, width, 3))[::-1, :, :]

        refpoint = self.model.data.site_xpos[0, :2]
        refpoint = self.mujoco_to_imagespace(refpoint, numpix=480)
        img[refpoint[0], refpoint[1]] = np.array([255, 255, 255])
        goalpoint = np.array(self.agentparams['goal_point'])
        goalpoint = self.mujoco_to_imagespace(goalpoint, numpix=480)
        img[goalpoint[0], goalpoint[1], :] = np.uint8(255)
        from PIL import Image
        Image.fromarray(img).show()
        import pdb
        pdb.set_trace()

    def quat_to_zangle(self, quat):
        """
        :param quat: quaternion 
        :return: zangle in rad
        """
        theta = np.arctan2(2*quat[0]*quat[3], 1-2*quat[3]**2)
        return np.array([theta])


    def act(self,  traj, t, init_model= None):
        """
        Return a random action for a state.
        Args:
            x_full, xdot_full history of states.
            ref_point: a reference point on the object which shall be moved to a goal
            dref_point: speed of reference point
            t: the current controller's Time step
            init_model: mujoco model to initialize from
        """
        self.t = t

        self.init_model = init_model

        desig_pos = self.init_model.data.site_xpos[0, :2]
        self.desig_pix.append(mujoco_to_imagespace(desig_pos, truncate=True))

        fullpose = self.init_model.data.qpos[2: 9].squeeze()
        zangle = self.quat_to_zangle(fullpose[3:])
        fullpose = np.concatenate([fullpose[:2], zangle])

        if t == 0:
            action = np.zeros(2)
            self.target = copy.deepcopy(self.init_model.data.qpos[:2].squeeze())
            self.init_retpos = copy.deepcopy(fullpose)
        else:

            last_images = traj._sample_images[t-1:t+1]
            last_states = traj.X_Xdot_full[t-1: t+1]
            last_action = self.action_list[-1]

            if 'use_first_plan' in self.policyparams:
                print 'using actions of first plan, no replanning!!'
                if t == 1:
                    self.perform_CEM(last_images, last_states, last_action, t)
                else:
                    # only showing last iteration
                    self.pred_pos = self.pred_pos[:,-1].reshape((self.M, 1, self.repeat * self.nactions, 2))
                    self.rec_target_pos = self.rec_target_pos[:, -1].reshape((self.M, 1, self.repeat * self.nactions, 2))
                    self.bestindices_of_iter = self.bestindices_of_iter[-1, :].reshape((1, self.K))
                action = self.bestaction_withrepeat[t - 1]

            else:
                self.perform_CEM(last_images, last_states, last_action, t)
                action = self.bestaction[0]

            self.setup_mujoco()


        self.action_list.append(action)
        print 'timestep: ', t, ' taking action: ', action

        if self.policyparams['low_level_ctrl'] == None:
            force = action
        else:
            if (t-1) % self.repeat == 0:
                self.target += action

            force = self.low_level_ctrl.act(traj.X_full[t], traj.Xdot_full[t], None, t, self.target)

        return force, self.pred_pos, self.bestindices_of_iter, self.rec_target_pos


def mujoco_to_imagespace(mujoco_coord, numpix = 64, truncate = False):
    """
    convert form Mujoco-Coord to numpix x numpix image space:
    :param numpix: number of pixels of square image
    :param mujoco_coord:
    :return: pixel_coord
    """
    viewer_distance = .75  # distance from camera to the viewing plane
    window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
    pixelheight = window_height / numpix  # height of one pixel
    pixelwidth = pixelheight
    window_width = pixelwidth * numpix
    middle_pixel = numpix / 2
    pixel_coord = np.rint(np.array([-mujoco_coord[1], mujoco_coord[0]]) /
                          pixelwidth + np.array([middle_pixel, middle_pixel]))
    pixel_coord = pixel_coord.astype(int)

    if truncate:
        if np.any(pixel_coord < 0) or np.any(pixel_coord > numpix -1):
            print '###################'
            print 'designated pixel is outside the field!! Resetting it to be inside...'
            print 'truncating...'
            if np.any(pixel_coord < 0):
                pixel_coord[pixel_coord < 0] = 0
            if np.any(pixel_coord > numpix-1):
                pixel_coord[pixel_coord > numpix-1]  = numpix-1

    return pixel_coord