from .tracking_model import Tracking_Model
import tensorflow as tf
import pickle
import numpy as np
import collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
from python_visual_mpc.video_prediction.basecls.prediction_train_base import create_one_hot

from python_visual_mpc.flow.descriptor_based_flow.train_descriptor_flow import search_region

class DescpTracking_Model(Tracking_Model):
    def __init__(self,
                conf = None,
                trafo_pix = True,
                load_data = True,
                mode=True):
        Tracking_Model.__init__(self,
                                conf = conf,
                                trafo_pix = trafo_pix,
                                load_data = load_data,
                                mode=mode)

    def visualize(self, sess, images, actions, states):
        if 'adim' in self.conf:
            from python_visual_mpc.video_prediction.read_tf_record_wristrot import \
                build_tfrecord_input as build_tfrecord_fn
        else:
            from python_visual_mpc.video_prediction.read_tf_record_sawyer12 import \
                build_tfrecord_input as build_tfrecord_fn

        images, actions, states = build_tfrecord_fn(self.conf, training=True)
        tf.train.start_queue_runners(sess)

        image_data, action_data, state_data = sess.run([images, actions, states])

        feed_dict = {}
        desig_pos_l = []
        load_desig_pos = False
        if load_desig_pos:
            desig_pos_l = pickle.load(open('utils/desig_pos.pkl', "rb"))
        else:
            for i in range(self.conf['batch_size']):
                c = Getdesig(image_data[i, 0], self.conf, 'b{}'.format(i))
                desig_pos = c.coords.astype(np.int32)
                desig_pos_l.append(desig_pos)
                # print "selected designated position for aux1 [row,col]:", desig_pos_aux1
            pickle.dump(desig_pos_l, open('utils/desig_pos.pkl', 'wb'))

        pix_distrib = np.concatenate(create_one_hot(self.conf, desig_pos_l), axis=0)
        feed_dict[self.pix_distrib_pl] = pix_distrib

        feed_dict[self.states_pl] = state_data
        feed_dict[self.images_pl] = image_data
        feed_dict[self.actions_pl] = action_data

        assert self.conf['schedsamp_k'] == -1

        ground_truth, gen_images, gen_masks, pred_flow, track_flow, gen_distrib, d0, d1 = sess.run([self.images,
                                                                                            self.gen_images,
                                                                                            self.gen_masks,
                                                                                            self.prediction_flow,
                                                                                            self.tracking_flow01,
                                                                                            self.gen_distrib,
                                                                                            self.descp0,
                                                                                            self.descp1,
                                                                                            ],
                                                                                           feed_dict)
        tracked_pos, heat_maps=  self.trace_points(desig_pos_l, d0, d1)
        ground_truth = add_crosshairs(ground_truth, tracked_pos)

        dict = collections.OrderedDict()
        dict['ground_truth'] = ground_truth
        dict['gen_images'] = gen_images
        dict['gen_masks'] = gen_masks
        import re
        itr_vis = re.match('.*?([0-9]+)$', self.conf['visualize']).group(1)
        dict['iternum'] = itr_vis
        dict['prediction_flow'] = pred_flow
        dict['tracking_flow'] = track_flow

        dict['heat_map'] = heat_maps

        if "visualize_tracking" in self.conf:
            dict['gen_distrib'] = gen_distrib

        file_path = self.conf['output_dir']
        pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
        print('written files to:' + file_path)

        v = Visualizer_tkinter(dict, numex=self.conf['batch_size'], append_masks=False, gif_savepath=self.conf['output_dir'], renorm_heatmaps=False)
        v.build_figure()
        return

    def trace_points(self, desig_pos_l, d0, d1):
        tracked_pos = np.zeros([self.conf['batch_size'], self.conf['sequence_length']-1, 2])
        heat_map = [np.zeros([self.conf['batch_size'], 64,64, 1]) for _ in range(self.conf['sequence_length']-1)]
        for b in range(self.conf['batch_size']):
            pos0 = desig_pos_l[b]
            tar_descp = d0[0][b][pos0[0], pos0[1]]
            current_pos = pos0
            for t in range(len(d0)):
                current_pos, hmap = search_region(self.conf, current_pos, d1[t][b], tar_descp)
                heat_map[t][b] = hmap
                tracked_pos[b, t] = current_pos
        return tracked_pos, heat_map


def add_crosshairs(images, pos):

    for b in range(images[0].shape[0]):
        for t in range(len(images)-1):
            im = np.squeeze(images[t+1][b])
            p = pos[b,t].astype(np.int)
            im[p[0]-5:p[0]-2,p[1]] = np.array([0, 1,1])
            im[p[0]+3:p[0]+6, p[1]] = np.array([0, 1, 1])

            im[p[0],p[1]-5:p[1]-2] = np.array([0, 1,1])

            im[p[0], p[1]+3:p[1]+6] = np.array([0, 1, 1])

            im[p[0], p[1]] = np.array([0, 1, 1])

            # plt.imshow(im)
            # plt.show()
            images[t][b] = im

    return images
