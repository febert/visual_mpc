#!/usr/bin/env python
import os
import rospy
import argparse
import imp
import cPickle as pkl
from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
import numpy as np

sudri_crop = {'left_cam': {'crop_bot': 70, 'crop_left': 130, 'crop_right': 120},
              'front_cam': {'crop_bot': 70, 'crop_left': 90, 'crop_right': 160}}
sudri_dict = {
    'robot_name': 'sudri',
    'targetpos_clip': [[0.375, -0.22, 0.184, -0.5 * np.pi, 0], [0.825, 0.24, 0.32, 0.5 * np.pi, 0.1]],
    'data_conf': sudri_crop
}

vestri_crop = {'left_cam': {'crop_bot': 70, 'crop_left': 150, 'crop_right': 100},
               'front_cam': {'crop_bot': 70, 'crop_left': 90, 'crop_right': 160}}
vestri_dict = {
    'robot_name': 'vestri',
    'targetpos_clip': [[0.42, -0.24, 0.184, -0.5 * np.pi, 0], [0.87, 0.22, 0.32, 0.5 * np.pi, 0.1]],
    'data_conf': vestri_crop
}
robot_confs = {
    'sudri': sudri_dict,
    'vestri': vestri_dict
}

class RobotEnvironment:
    def __init__(self, robot_name, conf, resume = False, ngpu = 1, gpu_id = 0):
        self._hyperparams = conf
        self.agentparams, self.policyparams = conf['agent'], conf['policy']

        if robot_name not in robot_confs:
            msg = "ROBOT {} IS NOT A SUPPORTED ROBOT ("
            for k in robot_confs.keys():
                msg = msg + "{} ".format(k)
            msg = msg + ")"
            raise NotImplementedError(msg)

        robot_conf = robot_confs[robot_name]
        for k in robot_conf.keys():
            if k not in self.agentparams:
                self.agentparams[k] = robot_conf[k]
            elif self.agentparams[k] != robot_conf[k]:
                """
                A major disadvantage to keeping config consistent across confs
                is we c+an't easily experiment w/ different settings. The benefit
                is all experiments have same default setting. Worth looking into
                this more later. 
                """
                raise AttributeError("ATTRIBUTE {} IS NOW SET BY COMMAND LINE".format(k))

        if 'benchmark_exp' in self.agentparams:
            self.is_bench = True
        else: self.is_bench = False
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

        self._ck_path = self.agentparams['data_save_dir'] + '/checkpoint.pkl'
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