from .general_agent import GeneralAgent
import pdb
import pickle as pkl
import numpy as np
import cv2

class BenchmarkAgent(GeneralAgent):
    def __init__(self, hyperparams):
        self._start_goal_confs = hyperparams['start_goal_confs']
        self.ncam = hyperparams['env'][1]['ncam']
        super().__init__(hyperparams)

    def _setup_world(self, itr):
        self._reset_state = self._load_raw_data(itr)
        super()._setup_world(itr)

    def _required_rollout_metadata(self, agent_data, traj_ok):
        super()._required_rollout_metadata(agent_data, traj_ok)
        agent_data['stats'] = self.env.eval()

    def _init(self):
        self.env.set_goal_obj_pose(self._goal_obj_pose)
        super()._init()

    def _load_raw_data(self, itr):
        """
        doing the reverse of save_raw_data
        :param itr:
        :return:
        """
        ngroup = 1000
        igrp = itr // ngroup
        group_folder = '{}/traj_group{}'.format(self._start_goal_confs, igrp)
        traj_folder = group_folder + '/traj{}'.format(itr)

        print('reading from: ', traj_folder)
        num_images = 2

        obs_dict = {}
        goal_images = np.zeros([num_images, self.ncam, self._hyperparams['image_height'], self._hyperparams['image_width'], 3])
        for t in range(num_images):  #TODO detect number of images automatically in folder
            for i in range(self.ncam):
                goal_images[t, i] = cv2.imread('{}/images{}/im_{}.png'.format(traj_folder, i, t))
        self._goal_image = goal_images[-1]
        #TODO: make compatible with fullimage tracknig
        with open('{}/agent_data.pkl'.format(traj_folder), 'rb') as file:
            agent_data = pkl.load(file)
        with open('{}/obs_dict.pkl'.format(traj_folder), 'rb') as file:
            obs_dict.update(pkl.load(file))
        reset_state = agent_data['reset_state']

        self._goal_obj_pose = obs_dict['object_qpos'][-1]

        return reset_state

