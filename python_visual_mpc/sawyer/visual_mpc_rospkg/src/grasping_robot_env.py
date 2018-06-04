#!/usr/bin/env python
import numpy as np
import rospy
import argparse
import imp

class RobotEnvironment:
    def __init__(self, conf):
        self._hyperparams = conf
        self.agentparams, self.policyparams = conf['agent'], conf['policy']

        #since the agent interacts with Sawyer, agent creation handles recorder/controller setup
        self.agent = self.agentparams['type'](self.agentparams)

        self.init_policy()

    def init_policy(self):
        self.policy = self.policyparams['type'](self.agentparams, self.policyparams)

    def run(self):
        for i in xrange(self._hyperparams['start_index'], self._hyperparams['end_index']):
            self.take_sample(i)

    def take_sample(self, itr):
        print("Collecting sample {}".format(itr))

        self.init_policy()
        traj, traj_ok = self.agent.sample(self.policy, itr)

        if traj is not None and traj_ok:
            group = itr // self._hyperparams['ngroup']
            traj_num = itr % self._hyperparams['ngroup']
            traj.save(self.agentparams['data_save_dir'] + '/traj_group{}/traj{}'.format(group, traj_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str, help='experiment name')
    args = parser.parse_args()

    hyperparams = imp.load_source('hyperparams', args.experiment)
    conf = hyperparams.config

    env = RobotEnvironment(conf)
    env.run()