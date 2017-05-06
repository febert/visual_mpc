""" This file defines the linear Gaussian policy class. """
import numpy as np

from lsdc.algorithm.policy.policy import Policy
import mujoco_py
from mujoco_py.mjtypes import *
from lsdc.agent.config import AGENT_MUJOCO
import copy
import time
import imp
import cPickle
from video_prediction.utils_vpred.create_gif import *
from datetime import datetime

from PIL import Image
import pdb


class CEM_controller(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, predictor = None):
        Policy.__init__(self)
        self.agentparams = copy.deepcopy(AGENT_MUJOCO)
        self.agentparams.update(ag_params)
        self.policyparams = policyparams

        self.t = None

        if self.policyparams['low_level_ctrl']:
            self.low_level_ctrl = policyparams['low_level_ctrl']['type'](None, policyparams['low_level_ctrl'])

        self.model = mujoco_py.MjModel(self.agentparams['filename'])

        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False

        if 'use_first_plan' in self.policyparams:
            self.use_first_plan = self.policyparams['use_first_plan']
        else: self.use_first_plan = True

        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

        self.use_net = self.policyparams['usenet']
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
        else:
            self.M = self.policyparams['num_samples']
            self.pred_len = self.nactions * self.repeat

        self.K = 10  # only consider K best samples for refitting

        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]
        self.gtruth_states = np.zeros((self.nactions * self.repeat, self.M, 4))

        # the full horizon is actions*repeat
        # self.action_cost_mult = 0.00005
        self.action_cost_mult = 0

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


        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.target = np.zeros(2)

        self.mean =None
        self.sigma =None

    def reinitialize(self):
        self.use_net = self.policyparams['usenet']
        self.action_list = []
        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]
        self.initial_std = self.policyparams['initial_std']

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

        self.mean = np.zeros(self.adim * self.nactions)
        self.sigma = np.diag(np.ones(self.adim * self.nactions) * self.initial_std ** 2)

        print '------------------------------------------------'
        print 'starting CEM cylce'

        # last_action = np.expand_dims(last_action, axis=0)
        # last_action = np.repeat(last_action, self.netconf['batch_size'], axis=0)
        # last_action = last_action.reshape(self.netconf['batch_size'], 1, self.adim)

        scores = np.empty(self.M, dtype=np.float64)

        for itr in range(self.niter):
            print '------------'
            print 'iteration: ', itr
            t_startiter = datetime.now()

            actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
            actions = actions.reshape(self.M, self.nactions, self.adim)
            # import pdb; pdb.set_trace()

            if self.verbose or not self.use_net:
                term_img = []
                for smp in range(self.M):
                    self.setup_mujoco()
                    term_img.append(self.sim_rollout(actions[smp], smp, itr))

                if 'mujoco_with_rewardnet' in self.policyparams:
                    scores = self.eval_with_rewardnet(term_img)

            actions = np.repeat(actions, self.repeat, axis=1)

            t_start = datetime.now()

            if self.use_net:
                scores = self.video_pred(last_frames, last_states, actions, itr)
                print 'overall time for evaluating actions {}'.format(
                    (datetime.now() - t_start).seconds + (datetime.now() - t_start).microseconds / 1e6)

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

        last_states = np.expand_dims(last_states, axis=0)
        last_states = np.repeat(last_states, self.netconf['batch_size'], axis=0)

        last_frames = np.expand_dims(last_frames, axis=0)
        last_frames = np.repeat(last_frames, self.netconf['batch_size'], axis=0)
        app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length']-
                                    self.netconf['context_frames'], 64, 64, 3))
        last_frames = np.concatenate((last_frames, app_zeros), axis=1)
        last_frames = last_frames.astype(np.float32)/255.

        inf_low_state, gen_images, gen_states = self.predictor( input_images= last_frames,
                                                                input_state=last_states,
                                                                input_actions = actions)
        for tstep in range(len(gen_states)):
            for smp in range(self.M):
                self.pred_pos[smp, itr, tstep+1] = self.mujoco_to_imagespace(
                                            gen_states[tstep][smp, :2], numpix=480)

        #evaluate distances to goalstate
        scores = np.zeros(self.netconf['batch_size'])

        if 'ballinvar' not in self.policyparams:  # the standard

            if 'usepixelerror' in self.policyparams:
                for b in range(self.netconf['batch_size']):
                    scores[b] = np.linalg.norm(
                        (self.goal_image[0][0] - gen_images[-1][b]).flatten())

            elif 'rewardnetconf' in self.policyparams:
                reward_func = self.policyparams['rewardnet_func']
                softmax_out =  reward_func(gen_images[-1], self.goal_image[0,0])
                # compute expected number time-steps
                if 'rewardmodel_sequence_length' in self.policyparams:
                    rewmodel_s_length = self.policyparams['rewardmodel_sequence_length']
                else:
                    rewmodel_s_length = 15
                scores = np.sum(softmax_out * np.arange(rewmodel_s_length-1), axis=1)
            else:
                    for b in range(self.netconf['batch_size']):
                        scores[b] = np.linalg.norm(self.goal_state.flatten()
                                                        - inf_low_state[-1][b].flatten())

        else:
            selected_scores = np.zeros(self.netconf['batch_size'], dtype= np.int)
            for b in range(self.netconf['batch_size']):
                scores_diffballpos = []
                for ballpos in range(self.netconf['batch_size']):
                    scores_diffballpos.append(
                         np.linalg.norm(self.goal_state[ballpos] - inf_low_state[-1][b])**2)

                selected_scores[b] = np.argmin(scores_diffballpos)

                scores[b] = np.min(scores_diffballpos)

        # compare prediciton with simulation
        if self.verbose: #and itr == self.policyparams['iterations']-1:
            # print 'creating visuals for best sampled actions at last iteration...'

            file_path = self.netconf['current_dir'] + '/verbose'

            bestindices = scores.argsort()[:self.K]
            bestscores = [scores[ind] for ind in bestindices]

            def best(inputlist):
                outputlist = [np.zeros_like(a)[:self.K] for a in inputlist]

                for ind in range(self.K):
                    for tstep in range(len(inputlist)):
                        outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
                return outputlist

            self.gtruth_images = [img.astype(np.float) / 255. for img in self.gtruth_images]  #[1:]
            cPickle.dump(best(gen_images), open(file_path + '/gen_image_seq.pkl', 'wb'))
            cPickle.dump(best(self.gtruth_images), open(file_path + '/ground_truth.pkl', 'wb'))
            print 'written files to:' + file_path
            comp_video(file_path, gif_name='check_eval_t{}'.format(self.t))

            f = open(file_path + '/actions_last_iter_t{}'.format(self.t), 'w')
            sorted = scores.argsort()
            for i in range(actions.shape[0]):
                f.write('index: {0}, score: {1}, rank: {2}'.format(i, scores[i],
                                                                   np.where(sorted == i)[0][0]))
                f.write('action {}\n'.format(actions[i]))


            # for i in range(self.K):
            #     bestind = bestindices[i]
            #     goalim  = self.goal_image[selected_scores[bestind]]
            #     Image.fromarray((goalim * 255.).astype(np.uint8)).show()


        bestindex = scores.argsort()[0]
        if 'store_video_prediction' in self.agentparams and\
                itr == (self.policyparams['iterations']-1):
            self.terminal_pred = gen_images[-1][bestindex]

        if itr == (self.policyparams['iterations']-2):
            self.verbose = True

        if 'store_whole_pred' in self.agentparams and \
            itr == (self.policyparams['iterations'] - 1):
            self.best_gen_images = [img[bestindex] for img in gen_images]
            self.best_gtruth_images = [img[bestindex] for img in self.gtruth_images]
            self.best_actions = actions[bestindex]
            self.verbose = False

        return scores


    def sim_rollout(self, actions, smp, itr):

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
                    self.pred_pos[smp, itr, t] = self.mujoco_to_imagespace(ball_coord, numpix=480)
                    if self.policyparams['low_level_ctrl']:
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

                if self.verbose or 'mujoco_with_rewardnet' in self.policyparams:
                    self.viewer.loop_once()

                    self.small_viewer.loop_once()
                    img_string, width, height = self.small_viewer.get_image()
                    img = np.fromstring(img_string, dtype='uint8').reshape(
                        (height, width, 3))[::-1, :, :]
                    self.gtruth_images[t][smp] = img
                    self.gtruth_states[t][smp] = np.concatenate([self.model.data.qpos[:2].squeeze(),
                                                                self.model.data.qvel[:2].squeeze()], axis=0)

        return img

    def eval_with_rewardnet(self, term):
        term = np.stack(term, axis=0)
        reward_func = self.policyparams['rewardnet_func']

        term = term.astype(np.float32)/255.

        softmax_out = reward_func(term, self.goal_image)
        # compute expected number time-steps
        if 'rewardmodel_sequence_length' in self.policyparams:
            rewmodel_s_length = self.policyparams['rewardmodel_sequence_length']
        else:
            rewmodel_s_length = 15
        scores = np.sum(softmax_out * np.arange(rewmodel_s_length - 1), axis=1)

        bestindices = scores.argsort()[:self.K]

        if self.verbose: #and itr == self.policyparams['iterations']-1:
            # print 'creating visuals for best sampled actions at last iteration...'
            file_path = '/home/frederik/Documents/lsdc/experiments/cem_exp/benchmarks_goalimage/mujoco_rewardnet/verbose'
            bestindices = scores.argsort()[:self.K]
            bestscores = [scores[ind] for ind in bestindices]

            def best(inputlist):
                outputlist = [np.zeros_like(a)[:self.K] for a in inputlist]

                for ind in range(self.K):
                    for tstep in range(len(inputlist)):
                        outputlist[tstep][ind] = inputlist[tstep][bestindices[ind]]
                return outputlist
            gtruth_images = best(self.gtruth_images)
            imlist = assemble_gif([gtruth_images], convert_from_float= False)
            npy_to_gif(imlist, file_path + '/check_eval_t{}'.format(self.t))

            Image.fromarray((self.goal_image*255).astype(np.uint8)).show()
            for i in range(self.K):
                Image.fromarray((term[bestindices[i]]*255).astype(np.uint8)).show()
                print 'bestscore ', bestscores[i]
                print 'softmax out', softmax_out[bestindices[i]]

            pdb.set_trace()

        return scores

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


    def act(self, traj, t, init_model= None):
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


        if t == 0:
            action = np.zeros(2)
            self.target = copy.deepcopy(self.init_model.data.qpos[:2].squeeze())
            self.get_goalimg()

            if 'use_lt_state' in self.policyparams:
                self.goal_state = self.inf_goal_state()

        else:

            last_images = traj._sample_images[t-1:t+1]
            last_states = traj.X_Xdot_full[t-1: t+1]
            last_action = self.action_list[-1]

            if self.use_first_plan:
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


    def get_goalimg(self):
        dict = cPickle.load(open(self.policyparams['use_goalimage'], "rb"))
        goal_image = dict['goal_image']
        # Image.fromarray(goal_image).show()
        self.goal_image = goal_image.astype(np.float32) / 255.

    def inf_goal_state(self):

        dict= cPickle.load(open(self.policyparams['use_goalimage'], "rb"))
        goal_image = dict['goal_image']
        goal_low_dim_st = dict['goal_ballpos']

        if 'ballinvar' not in self.policyparams:
            # Image.fromarray(goal_image).show()

            last_states = np.expand_dims(goal_low_dim_st, axis=0)
            last_states = np.repeat(last_states, 2, axis=0)  # copy over timesteps

            last_states = np.expand_dims(last_states, axis=0)
            last_states = np.repeat(last_states, self.netconf['batch_size'], axis=0) #copy over batch

            goal_image = np.expand_dims(goal_image, axis=0)
            goal_image = np.repeat(goal_image, 2, axis=0)   # copy over timesteps
            goal_image = np.expand_dims(goal_image, axis=0)
            goal_image = np.repeat(goal_image, self.netconf['batch_size'], axis=0) #copy over batch
            app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length'] -
                                        self.netconf['context_frames'], 64, 64, 3))
            goal_image = np.concatenate((goal_image, app_zeros), axis=1)
            goal_image = goal_image.astype(np.float32) / 255.

        else:

            last_states = np.expand_dims(goal_low_dim_st, axis=1)
            last_states = np.repeat(last_states, 2, axis=1)  # copy over timesteps

            goal_image = np.expand_dims(goal_image, axis=1)
            goal_image = np.repeat(goal_image, 2, axis=1)  # copy over timesteps

            app_zeros = np.zeros(shape=(self.netconf['batch_size'], self.netconf['sequence_length'] -
                                        self.netconf['context_frames'], 64, 64, 3))
            goal_image = np.concatenate((goal_image, app_zeros), axis=1)
            goal_image = goal_image.astype(np.float32) / 255.

        self.goal_image = goal_image

        actions = np.zeros([self.netconf['batch_size'], self.netconf['sequence_length'], 2])


        inf_low_state, gen_images, gen_sates = self.predictor(  input_images= goal_image,
                                                                input_state=last_states,
                                                                input_actions = actions)

        #TODO: Look at the predictions at the goalstate!!!!!!!!!

        # taking the inferred latent state of the last time step

        if 'no_pix_distrib' not in self.netconf:
            if 'ballinvar' not in self.policyparams:
                # taking any of the identical example in the batch
                goal_state = inf_low_state[-1][0]
            else:
                if 'nonrec' in self.netconf:
                    goal_state = inf_low_state[2]
                else:
                    goal_state = inf_low_state[-1]

        else:
            goal_state = None

        return  goal_state