#!/usr/bin/env python
import os
import rospy
import argparse
import imp
import cPickle as pkl
class RobotEnvironment:
    def __init__(self, conf, resume = False):
        self._hyperparams = conf
        self.agentparams, self.policyparams = conf['agent'], conf['policy']

        #since the agent interacts with Sawyer, agent creation handles recorder/controller setup
        self.agent = self.agentparams['type'](self.agentparams)

        self.init_policy()

        self._ck_path = self.agentparams['data_save_dir'] + '/checkpoint.pkl'
        if resume and os.path.exists(self._ck_path):
            with open(self._ck_path, 'rb') as f:
                self._ck_dict = pkl.load(f)

            self._hyperparams['start_index'] = self._ck_dict['ntraj']
        else:
            self._ck_dict = {'ntraj' : 0, 'broken_traj' : []}

    def init_policy(self):
        self.policy = self.policyparams['type'](self.agentparams, self.policyparams)

    def run(self):
        for i in xrange(self._hyperparams['start_index'], self._hyperparams['end_index']):
            if i % self._hyperparams.get('nshuffle', 200) == 0:
                print("You have one minute to shuffle objects....")
                rospy.sleep(60)
            self.take_sample(i)

    def take_sample(self, itr):
        print("Collecting sample {}".format(itr))

        self.init_policy()
        traj, traj_ok = self.agent.sample(self.policy, itr)

        if traj is not None and traj_ok:
            group = itr // self._hyperparams['ngroup']
            traj_num = itr % self._hyperparams['ngroup']
            traj.save(self.agentparams['data_save_dir'] + '/traj_group{}/traj{}'.format(group, traj_num))
        else:
            self._ck_dict['broken_traj'].append(itr)
        self._ck_dict['ntraj'] += 1

        ck_file = open(self._ck_path, 'wb')
        pkl.dump(self._ck_dict, ck_file)
        ck_file.close()

        print("CHECKPOINTED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('-r', action='store_true', dest='resume',
                        default=False, help='Set flag if resuming training')
    args = parser.parse_args()

    hyperparams = imp.load_source('hyperparams', args.experiment)
    conf = hyperparams.config

    env = RobotEnvironment(conf, args.resume)
    env.run()