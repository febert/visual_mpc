import tensorflow as tf
import numpy as np
import pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from python_visual_mpc.goaldistancenet.gdnet import apply_warp
import collections
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet
from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn

def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

class Temp_DnC_GDnet(GoalDistanceNet):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True,
                 iter_num = None,
                 load_testimages=None
                 ):
        GoalDistanceNet.__init__(self, conf = conf,
                                 build_loss=build_loss,
                                 load_data = load_data,
                                 iter_num = iter_num,
                                 load_testimages=load_testimages
                                 )
        self.build_loss = build_loss
        self.load_data = load_data

    def build_net(self):
        if 'compare_gtruth_flow' in self.conf:
            # model for eval purposes:
            with tf.variable_scope('warpnet'):
                self.gen_I1, self.warp_pts_bwd, self.flow_bwd, _ = self.warp(self.I0, self.I1)
        else:
            self.build_cons_model()

        if self.build_loss:
            if 'sched_layer_train' in self.conf:
                self.sched_layer_train()
            self.combine_losses()

    def merge_t_losses(self):
        "add together losses with same name"
        print('merging same ts')
        merged_losses = {}

        loss_list = []
        # stripping of last tag
        for n in list(self.losses.keys()):
            if '/t' in n:
                n = str.split(n,'/')[:-1]
                n ='/'.join(n)
            loss_list.append(n)
        unique_names_l = set(loss_list)

        # merging losses of different time steps
        for uname in unique_names_l:
            comb_loss_val = []
            for l_ in list(self.losses.keys()):
                if uname in l_:
                    comb_loss_val.append(self.losses[l_])
                    print("merging", l_)
            print('-----')
            comb_loss_val = tf.reduce_mean(tf.stack(comb_loss_val))
            merged_losses[uname] = comb_loss_val

        self.losses = merged_losses

    def sched_layer_train(self):
        thresholds = self.conf['sched_layer_train']
        for l in range(self.n_layer):
            layer_mult = tf.cast(tf.cast(self.iter_num,tf.int32) > tf.constant(int(thresholds[l]), tf.int32), tf.float32)
            for k in list(self.losses.keys()):
                if 'l{}'.format(l) in k:
                    self.losses[k] *= layer_mult
                    print('multiplying {} with layer_mult{}'.format(k, l))

    def combine_losses(self):
        self.merge_t_losses()
        super(Temp_DnC_GDnet, self).combine_losses()

    def build_cons_model(self):
        self.n_layer = int(np.log2(self.seq_len)) + 1
        if 'fwd_bwd' not in self.conf:
            occ_bwd = self.occ_bwd
        used = False
        flow_bwd_lm1 = None

        self.gen_img_ll = [[] for _ in range(self.n_layer)]
        self.flow_bwd_ll = [[] for _ in range(self.n_layer)]
        self.cons_diffs_ll = [[] for _ in range(self.n_layer)]
        self.img_warped_intflow = [[] for _ in range(self.n_layer)]

        im_summ_l = []

        for l in range(self.n_layer):
            tstep = int(np.power(2, l))
            flow_bwd_l = []

            cons_loss_per_layer = 0
            for i, t in enumerate(range(0, self.seq_len - 1, tstep)):

                print('l{}, t{}, warping im{} to im{}'.format(l, t, t, t + tstep))
                I0 = self.images[:, t]
                I1 = self.images[:, t + tstep]

                with tf.variable_scope('warpnet', reuse=used):
                    gen_I1, warp_pts_bwd, flow_bwd, _ = self.warp(I0, I1)

                    self.gen_img_ll[l].append(gen_I1)
                    self.flow_bwd_ll[l].append(flow_bwd)

                self.add_pair_loss(I1, gen_I1, occ_bwd, flow_bwd, suf='/l{}/t{}'.format(l,t))
                used = True

                if flow_bwd_lm1 is not None:
                    cons_loss, cons_diffs, int_flow = self.consistency_loss(i, flow_bwd_lm1, flow_bwd)
                    cons_loss_per_layer += cons_loss
                    self.cons_diffs_ll[l].append(cons_diffs)

                    self.img_warped_intflow[l].append(apply_warp(I0, int_flow))
                flow_bwd_l.append(flow_bwd)

                if i == 0:
                    im_summ_l.append(self.build_image_summary(
                        [I0, I1, gen_I1, length(flow_bwd)],
                        name='warp_im{}_to_im{}'.format(t, t + tstep)))

            self.image_summaries = tf.summary.merge(im_summ_l)

            self.losses['cons_loss/l{}'.format(l)] = cons_loss_per_layer*self.conf['cons_loss']
            flow_bwd_lm1 = flow_bwd_l


    def visualize(self, sess):
        if 'compare_gtruth_flow' in self.conf:  # visualizing single warps from pairs of images
            super(Temp_DnC_GDnet, self).visualize(sess)
        else:
            images, gen_img_ll, flow_bwd_ll, cons_diffs, warped_int_flow = sess.run([self.images, self.gen_img_ll, self.flow_bwd_ll, self.cons_diffs_ll, self.img_warped_intflow], feed_dict = {self.train_cond:1})
    
            dict = collections.OrderedDict()
            dict['images'] = images
            dict['gen_img_ll'] = gen_img_ll
            dict['flow_bwd_ll'] = flow_bwd_ll
            dict['cons_diffs'] = cons_diffs
            dict['warped_int_flow'] = warped_int_flow
    
            name = str.split(self.conf['output_dir'], '/')[-2]
            dict['name'] = name
    
            pickle.dump(dict, open(self.conf['output_dir'] + '/data.pkl', 'wb'))
            make_plots(self.conf, dict=dict)

    def consistency_loss(self, i, flow_bwd_lm1, flow_bwd):
        int_flow = apply_warp(flow_bwd_lm1[i*2],flow_bwd_lm1[i*2+1]) + flow_bwd_lm1[i*2+1]
        # return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(int_flow - flow_bwd), axis=-1)))  # this does not seem to work

        cons_diffs = int_flow - flow_bwd
        return tf.reduce_mean(tf.square(int_flow - flow_bwd)), cons_diffs, int_flow


def make_plots(conf, dict=None, filename = None):
    if dict == None:
        dict = pickle.load(open(filename))

    print('loaded')
    images = dict['images']
    gen_img_ll = dict['gen_img_ll']
    flow_bwd_ll = dict['flow_bwd_ll']
    cons_diffs = dict['cons_diffs']
    warped_int_flow = dict['warped_int_flow']

    num_rows = len(gen_img_ll)*4 +1

    seq_len = images.shape[1]
    num_cols = seq_len

    width_per_ex = 2.5

    standard_size = np.array([width_per_ex * num_cols, num_rows * 1.5])  ### 1.5
    figsize = (standard_size).astype(np.int)

    f, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    bexp = 1
    #plot images:
    for col in range(num_cols):
        row = 0
        h = axarr[row, col].imshow(np.squeeze(images[bexp, col]), interpolation='none')

    n_layers = len(gen_img_ll)
    for l in range(n_layers):
        tstep = int(np.power(2, l))
        for i, t in enumerate(range(0, seq_len - 1, tstep)):
            print('l{}, t{}, showing warp im{} to im{}'.format(l, t, t, t + tstep))

            axarr[l*4+1, t + tstep].imshow(np.squeeze(gen_img_ll[l][i][bexp]), interpolation='none')
            sq_len = np.sqrt(np.sum(np.square(flow_bwd_ll[l][i][bexp]), -1))
            h = axarr[l * 4 + 2, t+tstep].imshow(sq_len, interpolation='none')
            plt.colorbar(h, ax=axarr[l * 4 + 2, t+tstep])

            if l > 0:
                im = warped_int_flow[l][i][bexp]
                h = axarr[l * 4 + 3, t + tstep].imshow(im, interpolation='none')

                sq_len_diff = np.sqrt(np.sum(np.square(cons_diffs[l][i][bexp]), -1))
                h = axarr[l * 4 + 4, t + tstep].imshow(sq_len_diff, interpolation='none')
                plt.colorbar(h, ax=axarr[l * 4 + 4, t+tstep])

    # plt.axis('off')
    f.subplots_adjust(wspace=0, hspace=0.0)
    # f.subplots_adjust(wspace=0, hspace=0.3)

    # f.subplots_adjust(vspace=0.1)
    plt.show()
    # plt.savefig(conf['output_dir']+'/consloss{}.png'.format(dict['name']))


if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/tdac_cons1e-4/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')