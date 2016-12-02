""" This file defines the linear Gaussian policy class. """
import numpy as np

from lsdc.algorithm.policy.policy import Policy
from lsdc.utility.general_utils import check_shape
import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *
from lsdc.agent.config import AGENT_MUJOCO
import copy
import time
import imp
import cPickle
from video_prediction.setup_predictor import setup_predictor
from video_prediction.utils_vpred.create_gif import comp_pix_distrib
from PIL import Image
import pdb


class CEM_controller(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams):
        Policy.__init__(self)
        self.agentparams = copy.deepcopy(AGENT_MUJOCO)
        self.agentparams.update(ag_params)

        self.low_level_ctrl = policyparams['low_level_ctrl']['type'](None, policyparams['low_level_ctrl'])

        self.policyparams = policyparams

        self.model = mujoco_py.MjModel(self.agentparams['filename'])
        self._data = {}  #dictionary for storing the data_files

        self.verbose = False
        self.compare_sim_net = True
        self.use_first_plan = True

        self.niter = 10  # number of iterations

        self.use_net = False
        self.action_list = []

        hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
        self.netconf = hyperparams.configuration

        self.horizon = 5
        self.repeat = 3
        if self.use_net:
            self.M = self.netconf['batch_size']
            assert self.horizon*self.repeat == self.netconf['sequence_length']
            self.predictor = setup_predictor(self.policyparams['netconf'])
            self.K = 10  # only consider K best samples for refitting
        else:
            self.M = 200 #200
            self.K = 10  # only consider K best samples for refitting

        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.horizon*self.repeat)]

        # the full horizon is actions*repeat
        self.action_cost_mult = 0.05
        self.adim = 2  # action dimension
        self.initial_std = policyparams['initial_std']

        gofast = False
        self.viewer = mujoco_py.MjViewer(visible=True, init_width=480,
                                         init_height=480, go_fast=gofast)
        self.viewer.start()
        self.viewer.cam.camid = 0
        self.viewer.set_model(self.model)

        self.small_viewer = mujoco_py.MjViewer(visible=True, init_width=64,
                                         init_height=64, go_fast=gofast)
        self.small_viewer.start()
        self.small_viewer.cam.camid = 0
        self.small_viewer.set_model(self.model)


        self.init_model = []

        #history of designated pixels
        self.desig_pix = np.zeros((self.agentparams['T'], 2), dtype=np.int)

        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat*self.horizon, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.horizon, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.target = np.zeros(2)




    def setup_mujoco(self):

        # set initial conditions
        self.model.data.qpos = self.init_model.data.qpos
        self.model.data.qvel = self.init_model.data.qvel

    def eval_action(self):
        goalpoint = np.array(self.agentparams['goal_point'])
        refpoint = self.model.data.site_xpos[0,:2]

        return np.linalg.norm(goalpoint - refpoint)

    def calc_action_cost(self, actions_of_smp):
        force_magnitudes = np.array([np.linalg.norm(actions_of_smp[t]) for t in range(self.horizon)])
        return np.square(force_magnitudes)*self.action_cost_mult

    def perform_CEM(self,last_frames, last_states, last_desig_pix, last_action):
        # initialize mean and variance
        mean = np.zeros(self.adim*self.horizon)
        sigma = np.diag(np.ones(self.adim*self.horizon) * self.initial_std**2)

        print '------------------------------------------------'
        print 'starting CEM cylce'

        # last_action = np.expand_dims(last_action, axis=0)
        # last_action = np.repeat(last_action, self.netconf['batch_size'], axis=0)
        # last_action = last_action.reshape(self.netconf['batch_size'], 1, self.adim)

        scores = np.empty(self.M, dtype=np.float64)
        actioncosts = np.empty(self.M, dtype=np.float64)

        for itr in range(self.niter):
            print '------------'
            print 'iteration: ', itr

            # print 'mean:'
            # print mean
            # print 'covariance:'
            # print sigma

            actions = np.random.multivariate_normal(mean, sigma, self.M)
            actions = actions.reshape(self.M, self.horizon, self.adim)

            if self.compare_sim_net:
                for smp in range(self.M):
                    self.setup_mujoco()
                    self.sim_rollout(actions[smp], smp, itr)
                    scores[smp] = self.eval_action()
                    actioncosts[smp] = np.sum(self.calc_action_cost(actions[smp]))
                    scores[smp] += actioncosts[smp]  # adding action costs!

            actions = np.repeat(actions, self.repeat, axis=1)

            # prepending last action to the sampled actions:
            # actions = np.concatenate((last_action, actions), axis=1)
            # actions = actions[:,:self.netconf['sequence_length'],:]
            if self.use_net:
                scores = self.video_pred(last_frames, last_states, actions, last_desig_pix, itr)
            self.indices = scores.argsort()[:self.K]
            self.bestindices_of_iter[itr] = self.indices

            self.bestaction_withrepeat = actions[self.indices[0]]

            actions = actions.reshape(self.M, self.horizon, self.repeat, self.adim)
            actions = actions[:,:,-1,:] #taking only one of the repeated actions
            actions_flat = actions.reshape(self.M , self.horizon*self.adim)

            self.bestaction = actions[self.indices[0]]

            arr_best_actions = actions_flat[self.indices]  # only take the K best actions
            sigma = np.cov(arr_best_actions, rowvar= False, bias= False)
            mean = np.mean(arr_best_actions, axis= 0)

            print 'iter {0}, bestscore {1}'.format(itr, scores[self.indices[0]])
            print 'action cost of best action: ', actioncosts[self.indices[0]]


    def mujoco_to_imagespace(self, mujoco_coord, numpix = 64):
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
        return pixel_coord


    def video_pred(self, last_frames, last_states, actions, last_desig_pix, itr):
        one_hot_images = np.zeros((1, self.netconf['context_frames'], 64, 64 , 1), dtype= np.float32)

        self.pred_pos[:, itr, 0] = last_states[-1,:2]

        # switch on pixels
        for i in range(self.netconf['context_frames']):
            # only for testing!:
            one_hot_images[0, i, last_desig_pix[i, 0], last_desig_pix[i, 1]] = 1

        one_hot_images = np.repeat(one_hot_images, self.netconf['batch_size'], axis=0)

        last_states = np.expand_dims(last_states, axis=0)
        last_states = np.repeat(last_states, self.netconf['batch_size'], axis=0)

        last_frames = np.expand_dims(last_frames, axis=0)
        last_frames = np.repeat(last_frames, self.netconf['batch_size'], axis=0)
        app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length']-
                                    self.netconf['context_frames'], 64, 64, 3))
        last_frames = np.concatenate((last_frames, app_zeros), axis=1)
        last_frames = last_frames.astype(np.float32)/255.

        gen_distrib, gen_images, gen_masks, gen_states = self.predictor(last_frames, one_hot_images,
                                                            last_states, actions)

        # import pdb;pdb.set_trace()


        #compare prediciton with simulation
        if self.verbose:
            file_path = self.netconf['current_dir'] + '/data_files'
            cPickle.dump(gen_distrib, open(file_path + '/gen_distrib.pkl', 'wb'))
            cPickle.dump(gen_images, open(file_path + '/gen_images.pkl', 'wb'))
            cPickle.dump(gen_masks, open(file_path + '/gen_masks.pkl', 'wb'))
            self.gtruth_images = [img.astype(np.float)/255. for img in self.gtruth_images]
            cPickle.dump(self.gtruth_images, open(file_path + '/gtruth_images.pkl', 'wb'))
            print 'written files to:' + file_path

            comp_pix_distrib(file_path,name='check_eval_hor10', masks= False, examples = 20)

        # import pdb; pdb.set_trace()

        np.zeros((self.M, self.niter, self.repeat * self.horizon, 2))



        for t in range(1,self.netconf['sequence_length']):
            for smp in range(self.M):
                self.pred_pos[smp, itr, t] = self.mujoco_to_imagespace(gen_states[t][smp, :2], numpix=480)

        # import pdb;
        # pdb.set_trace()

        # scores = np.zeros((self.netconf['batch_size'], self.netconf['sequence_length']-1))
        scores = np.zeros((self.netconf['batch_size']))

        goalpoint = self.mujoco_to_imagespace(self.agentparams['goal_point'])
        for i in range(self.netconf['batch_size']):
            # for t in range(len(gen_distrib)):
            peak_pix = np.argmax(gen_distrib[-1][i])
            peak_pix = np.unravel_index(peak_pix,
                                        (self.agentparams['image_width'],
                                         self.agentparams['image_width']))
            peak_pix = np.array(peak_pix)
            scores[i] = np.linalg.norm(goalpoint.astype(float) - peak_pix.astype(float))


        distance_grid = np.empty((64,64))
        for i in range(64):
            for j in range(64):
                pos = np.array([i,j])
                distance_grid[i,j] = np.linalg.norm(goalpoint - pos)

        expected_distance = np.empty(self.netconf['batch_size'])
        # for t in range (len(gen_distrib)):

        for b in range(self.netconf['batch_size']):
            gen = gen_distrib[-1][b].squeeze()/ np.sum(gen_distrib[-1][b])
            expected_distance[b] = np.sum(np.multiply(gen, distance_grid))

            # print 'score action {0}: {1}'.format(i, scores[i])
                # scores[i, t] = np.squeeze(gen_distrib[t][i, goalpoint[0], goalpoint[1]])


        # from PIL import Image
        # for i in range(5): img = last_frames[0,i,...]; img = np.uint8(img * 255.); Image.fromarray(img).show()
        # for i in range(2): img = one_hot_images[0, i, ...].squeeze(); img = np.uint8(img * 255.); Image.fromarray(img).show()

        # import pdb
        # pdb.set_trace()

        return expected_distance


    def sim_rollout(self, actions, smp, itr):
        rollout_ctrl = self.policyparams['low_level_ctrl']['type'](None, self.policyparams['low_level_ctrl'])
        roll_target_pos = copy.deepcopy(self.init_model.data.qpos[:2].squeeze())

        for hstep in range(self.horizon):
            currentaction = actions[hstep]

            if self.policyparams['low_level_ctrl']:
                roll_target_pos += currentaction

            for r in range(self.repeat):
                t = hstep*self.repeat + r
                # print 'time ',t, ' target pos rollout: ', roll_target_pos

                if not self.use_net:
                    ball_coord = self.model.data.qpos[:2].squeeze()
                    self.pred_pos[smp, itr, t] = self.mujoco_to_imagespace(ball_coord, numpix=480)
                    self.rec_target_pos[smp, itr, t] = self.mujoco_to_imagespace(roll_target_pos, numpix=480)

                if self.policyparams['low_level_ctrl'] == None:
                    force = currentaction
                else:
                    qpos = self.model.data.qpos[:2].squeeze()
                    qvel = self.model.data.qvel[:2].squeeze()
                    force = rollout_ctrl.act(qpos, qvel, None, t, roll_target_pos)

                for _ in range(self.agentparams['substeps']):
                    self.model.data.ctrl = force
                    self.model.step()  # simulate the model in mujoco

                if self.verbose:
                    self.viewer.loop_once()

                    self.small_viewer.loop_once()
                    img_string, width, height = self.small_viewer.get_image()
                    img = np.fromstring(img_string, dtype='uint8').reshape(
                        (height, width, 3))[::-1, :, :]
                    self.gtruth_images[t][smp] = img
                    # self.check_conversion()

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
        import pdb;
        pdb.set_trace()

    def act(self, x_full, xdot_full, full_images, t, init_model= None):
        """
        Return a random action for a state.
        Args:
            x_full, xdot_full history of states.
            ref_point: a reference point on the object which shall be moved to a goal
            dref_point: speed of reference point
            t: Time step
            init_model: mujoco model to initialize from
        """
        self.init_model = init_model
        desig_pos = init_model.data.site_xpos[0, :2]
        self.desig_pix[t, :] = self.mujoco_to_imagespace(desig_pos)

        if t == 0:
            action = np.zeros(2)
            self.target = copy.deepcopy(self.init_model.data.qpos[:2].squeeze())
        else:
            last_images = full_images[t-1:t+1]
            last_states = np.concatenate((x_full,xdot_full), axis = 1)[t-1: t+1]
            last_desig_pixels = self.desig_pix[t-1: t+1]
            last_action = self.action_list[-1]

            if self.use_first_plan:
                print 'using actions of first plan, no replanning!!'
                if t == 1:
                    self.perform_CEM(last_images, last_states, last_desig_pixels, last_action)
                    action = self.bestaction
                else:
                    # only showing last iteration
                    self.pred_pos = self.pred_pos[:,-1].reshape((self.M, 1, self.repeat*self.horizon, 2))
                    self.rec_target_pos = self.rec_target_pos[:, -1].reshape((self.M, 1, self.repeat * self.horizon, 2))
                    self.bestindices_of_iter = self.bestindices_of_iter[-1, :].reshape((1, self.K))
                action = self.bestaction_withrepeat[t - 1]

            else:
                self.perform_CEM(last_images, last_states, last_desig_pixels, last_action)
                action = self.bestaction[0]

            self.setup_mujoco()
            # print 'current distance :', self.eval_action()


        self.action_list.append(action)
        print 'timestep: ', t, ' taking action: ', action

        if self.policyparams['low_level_ctrl'] == None:
            force = action
        else:
            if (t-1) % self.repeat == 0:
                self.target += action

            force = self.low_level_ctrl.act(x_full[t], xdot_full[t], None, t, self.target)

        return force, self.pred_pos, self.bestindices_of_iter, self.rec_target_pos