#!/usr/bin/env python
import os
import rospy
import argparse
import imp
import cPickle as pkl
from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
import shutil
import cv2
from python_visual_mpc.visual_mpc_core.agent.utils.traj_saver import record_worker
from multiprocessing import Manager, Process


supported_robots = {'sudri', 'vestri'}


class RobotEnvironment:
    def __init__(self, robot_name, conf, resume = False, ngpu = 1, gpu_id = 0):
        self._hyperparams = conf
        self.agentparams, self.policyparams, self.envparams = conf['agent'], conf['policy'], conf['agent']['env'][1]

        if robot_name not in supported_robots:
            msg = "ROBOT {} IS NOT A SUPPORTED ROBOT (".format(robot_name)
            for k in supported_robots:
                msg = msg + "{} ".format(k)
            msg = msg + ")"
            raise NotImplementedError(msg)

        self.envparams['robot_name'] = robot_name

        if 'benchmark_exp' in self.agentparams:
            self.is_bench = True
            self.task_mode = '{}/test'.format(robot_name)
        else:
            self.is_bench = False
            self.task_mode = '{}/train'.format(robot_name)

        if 'register_gtruth' in self.policyparams:
            assert 'register_gtruth' not in self.agentparams, "SHOULD BE IN POLICY PARAMS"
            self.agentparams['register_gtruth'] = self.policyparams['register_gtruth']
        self._ngpu = ngpu
        self._gpu_id = gpu_id

        #since the agent interacts with Sawyer, agent creation handles recorder/controller setup
        self.agent = self.agentparams['type'](self.agentparams)

        self._netconf = {}
        self._predictor = None
        self._goal_image_warper = None
        self._gdnconf = {}
        self.init_policy()

        robot_dir = self.agentparams['data_save_dir'] + '/{}'.format(robot_name)
        if not os.path.exists(robot_dir):
            os.makedirs(robot_dir)

        self._ck_path = self.agentparams['data_save_dir'] + '/{}/checkpoint.pkl'.format(robot_name)
        if resume and os.path.exists(self._ck_path):
            with open(self._ck_path, 'rb') as f:
                self._ck_dict = pkl.load(f)

            self._hyperparams['start_index'] = self._ck_dict['ntraj']
        else:
            self._ck_dict = {'ntraj' : 0, 'broken_traj' : []}



    def init_policy(self):

        if 'use_server' not in self.policyparams and 'netconf' in self.policyparams and self._predictor is None:
            self._netconf = imp.load_source('params', self.policyparams['netconf']).configuration
            self._predictor = self._netconf['setup_predictor']({}, self._netconf, self._gpu_id, self._ngpu)

        if 'use_server' not in self.policyparams and 'gdnconf' in self.policyparams and self._goal_image_warper is None:
            self._gdnconf = imp.load_source('params', self.policyparams['gdnconf']).configuration
            self._goal_image_warper = setup_gdn(self._gdnconf, self._gpu_id)

        if self._goal_image_warper is not None and self._predictor is not None:
            self.policy = self.policyparams['type'](None, self.agentparams, self.policyparams, self._predictor, self._goal_image_warper)
        elif self._predictor is not None:
            self.policy = self.policyparams['type'](None, self.agentparams, self.policyparams, self._predictor)
        else:
            self.policy = self.policyparams['type'](None, self.agentparams, self.policyparams)

    def run(self):
        if not self.is_bench:
            for i in xrange(self._hyperparams['start_index'], self._hyperparams['end_index']):
                if i % self._hyperparams.get('nshuffle', 200) == 0 and i > 0:
                    print("You have 30 seconds to shuffle objects....")
                    rospy.sleep(30)
                self.take_sample(i)
        else:
            itr = 0
            while True:
                self.take_sample(itr)
                itr += 1

    def take_sample(self, sample_index):
        print("Collecting sample {}".format(sample_index))

        self.init_policy()
        agent_data, obs_dict, policy_out = self.agent.sample(self.policy, sample_index)

        if self._hyperparams['save_data']:
            self.save_data(sample_index, agent_data, obs_dict, policy_out)

        self._ck_dict['ntraj'] += 1

        ck_file = open(self._ck_path, 'wb')
        pkl.dump(self._ck_dict, ck_file)
        ck_file.close()

        print("CHECKPOINTED")

    def save_data(self, itr, agent_data, obs_dict, policy_outputs):
        self._save_raw_images(itr, agent_data, obs_dict, policy_outputs)

    def _save_raw_images(self, itr, agent_data, obs_dict, policy_outputs):
        if 'RESULT_DIR' in os.environ:
            data_save_dir = os.environ['RESULT_DIR'] + '/data'
        else: data_save_dir = self.agentparams['data_save_dir']
        data_save_dir += '/' + self.task_mode

        ngroup = self._hyperparams['ngroup']
        igrp = itr // ngroup
        group_folder = data_save_dir + '/traj_group{}'.format(igrp)
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        traj_folder = group_folder + '/traj{}'.format(itr)
        if os.path.exists(traj_folder):
            shutil.rmtree(traj_folder)

        os.makedirs(traj_folder)
        if 'images' in obs_dict:
            images = obs_dict.pop('images')
            T, n_cams = images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/images{}'.format(i))
            for t in range(T):
                for i in range(n_cams):
                    cv2.imwrite('{}/images{}/im_{}.jpg'.format(traj_folder, i, t), images[t, i, :, :, ::-1])
        with open('{}/agent_data.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(agent_data, file)
        with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(obs_dict, file)
        with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(policy_outputs, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('robot_name', type=str, help="name of robot we're running on")
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('-r', action='store_true', dest='resume',
                        default=False, help='Set flag if resuming training')
    parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    args = parser.parse_args()

    hyperparams = imp.load_source('hyperparams', args.experiment)
    conf = hyperparams.config

    env = RobotEnvironment(args.robot_name, conf, args.resume, args.ngpu, args.gpu_id)
    env.run()
