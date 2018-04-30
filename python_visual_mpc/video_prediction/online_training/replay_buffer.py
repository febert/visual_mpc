import numpy as np
import os
import tensorflow as tf
import random
import ray
from collections import namedtuple
from python_visual_mpc.video_prediction.read_tf_records2 import build_tfrecord_input
import pdb

Traj = namedtuple('Traj', 'images X_Xdot_full actions')

class ReplayBuffer(object):
    def __init__(self, maxsize, batch_size, data_collectors, todo_ids):
        self.ring_buffer = []
        self.maxsize = maxsize
        self.batch_size = batch_size
        self.data_collectors = data_collectors
        self.todo_ids = todo_ids

    def push_back(self, traj):
        self.ring_buffer.append(traj)
        if len(self.ring_buffer) > self.maxsize:
            self.ring_buffer.pop(0)

    def get_batch(self):
        images = []
        states = []
        actions = []
        current_size = len(self.ring_buffer)
        for b in range(self.batch_size):
            i = random.randint(0, current_size-1)
            traj = self.ring_buffer[i]
            images.append(traj.images.astype(np.float32)/255.)
            states.append(traj.X_Xdot_full)
            actions.append(traj.actions)
        return np.stack(images,0), np.stack(states,0), np.stack(actions,0)

    def prefil(self, prefil_n, trainvid_conf):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        g_vidpred = tf.Graph()
        sess = tf.Session()
        with sess.as_default():
            with g_vidpred.as_default():
                pdb.set_trace()
                pdb.set_trace()
                dict = build_tfrecord_input(trainvid_conf, training=True)
                tf.train.start_queue_runners(sess)
                sess.run(tf.global_variables_initializer())
                for i_run in range(prefil_n//trainvid_conf['batch_size']):
                    images, actions, endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos']])
                    for b in range(trainvid_conf['batch_size']):
                        t = Traj(images[b], endeff[b], actions[b])
                        self.push_back(t)

        sess.close()
        tf.reset_default_graph()

    def update(self):
        done_id, self.todo_ids = ray.wait(self.todo_ids, timeout=0)
        if done_id is not []:
            for id in done_id:
                traj, info = ray.get(id)
                print("pushing back traj")
                self.push_back(traj)
                # relauch the collector if it hasn't done all its work yet.
                returning_collector = self.data_collectors[info['collector_id']]
                if info['itraj'] < info['maxtraj']:
                    self.todo_ids.append(returning_collector.run_traj.remote())
                else:
                    print('tasks finished, no new task launched')
