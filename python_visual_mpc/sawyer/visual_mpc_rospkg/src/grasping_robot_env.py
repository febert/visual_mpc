#!/usr/bin/env python
import os
import rospy
import argparse
import imp
import cPickle as pkl
from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn

class RobotEnvironment:
    def __init__(self, conf, resume = False, ngpu = 1, gpu_id = 0):
        self._hyperparams = conf
        self.agentparams, self.policyparams = conf['agent'], conf['policy']

        if 'benchmark_exp' in self.agentparams:
            self.is_bench = True
        else: self.is_bench = False
        self._ngpu = ngpu
        self._gpu_id = gpu_id

        #since the agent interacts with Sawyer, agent creation handles recorder/controller setup
        self.agent = self.agentparams['type'](self.agentparams)

        self._netconf = {}
        self._predictor = None
        self._goal_image_warper = None
        self._gdnconf = {}
        self.init_policy()

        self._ck_path = self.agentparams['data_save_dir'] + '/checkpoint.pkl'
        if resume and os.path.exists(self._ck_path):
            with open(self._ck_path, 'rb') as f:
                self._ck_dict = pkl.load(f)

            self._hyperparams['start_index'] = self._ck_dict['ntraj']
        else:
            self._ck_dict = {'ntraj' : 0, 'broken_traj' : []}

    def init_policy(self):
        if 'use_server' not in self.policyparams and 'netconf' in self.policyparams:
            self._netconf = imp.load_source('params', self.policyparams['netconf']).configuration
            self._predictor = self._netconf['setup_predictor']({}, self._netconf, self._gpu_id, self._ngpu)

        if 'use_server' not in self.policyparams and 'gdnconf' in self.policyparams:
            self._gdnconf = imp.load_source('params', self.policyparams['gdnconf']).configuration
            self._goal_image_warper = setup_gdn(self._gdnconf, self._gpu_id)

        if self._goal_image_warper is not None and self._predictor is not None:
            self.policy = self.policyparams['type'](None, self.agentparams, self.policyparams, self._predictor, self._goal_image_warper)
        elif self._predictor is not None:
            self.policy = self.policyparams['type'](None, self.agentparams, self.policyparams, self._predictor)
        else:
            self.policy = self.policyparams['type'](None, self.agentparams, self.policyparams)

    def run(self):
        for i in xrange(self._hyperparams['start_index'], self._hyperparams['end_index']):
            if i % self._hyperparams.get('nshuffle', 200) == 0 and i > 0 and 'benchmark_exp' not in self.is_bench:
                print("You have 30 seconds to shuffle objects....")
                rospy.sleep(30)
            self.take_sample(i)

    def take_sample(self, itr):
        print("Collecting sample {}".format(itr))

        self.init_policy()
        traj, traj_ok = self.agent.sample(self.policy, itr)

        if traj is not None and traj_ok:
            group = itr // self._hyperparams['ngroup']
            traj_num = itr % self._hyperparams['ngroup']
            if self.is_bench:
                traj.save(self.agentparams['data_save_dir'] + '/{}/traj_data'.format(self.agentparams['benchmark_exp']))
            else:
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
    parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    args = parser.parse_args()

    hyperparams = imp.load_source('hyperparams', args.experiment)
    conf = hyperparams.config



    env = RobotEnvironment(conf, args.resume, args.ngpu, args.gpu_id)
    env.run()