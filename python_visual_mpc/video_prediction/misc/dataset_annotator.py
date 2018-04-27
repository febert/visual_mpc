import numpy as np
import os
import pdb
import tensorflow as tf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pickle
from python_visual_mpc.video_prediction.read_tf_record_wristrot import build_tfrecord_input
from matplotlib import animation
from PIL import Image

import argparse


class Getdesig(object):
    def __init__(self,images, i_click_max, color='r'):
        fig = plt.figure()
        self.images = images
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        self.color = color

        plt.imshow(images[0])

        self.pos = []

        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_click = 0
        self.i_click_max = i_click_max

        plt.show()

    def onclick(self, event):
        # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       (event.button, event.x, event.y, event.xdata, event.ydata))
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)

        try:
            newpos = np.around(np.array([event.ydata, event.xdata])).astype(np.int64)
            self.ax.scatter(newpos[1], newpos[0], s=20, marker="D", facecolors=self.color, edgecolors=self.color)
            self.pos.append(newpos)
            # print 'marked', newpos
            plt.draw()

            self.i_click += 1

            if self.i_click == self.i_click_max:
                plt.close()
            else:
                plt.imshow(self.images[self.i_click])
                plt.show()
        except:
            print('clicked in the wrong place!')


t = 0
def play_video(conf, images):

    fig = plt.figure()
    im_handle = plt.imshow(images[0], interpolation='none', animated=True)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    tlen = conf['sequence_length']

    Writer = animation.writers['imagemagick_file']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim = animation.FuncAnimation(fig, animate,
                                   fargs=[im_handle, images, tlen],
                                   frames=tlen, interval=200, blit=True)
    plt.show()

def animate(self, *args):
    global t
    # pdb.set_trace()
    im_handle, video,  tlen = args
    im_handle.set_array(video[t])
    # print 'update at t', t
    t += 1
    if t == tlen:
        t = 0
    return [im_handle]


def run_annotation(conf, traj_data):
    n_object_tracks = conf['object_pos']
    images, actions, endeff_pos = traj_data

    for b in range(images.shape[0]):
        print('example {}'.format(b))
        done_example = False
        while not done_example:
            current_vid = images[b]
            # play_video(conf, current_vid)

            print('choose a point on the robot and track it!')

            tr = Getdesig(current_vid,conf['sequence_length'])
            robot_track = np.stack(tr.pos, axis=0)
            del (tr)

            object_track_l = []
            for i in range(n_object_tracks):
                print('choose a point on an object and track it!')
                tr = Getdesig(current_vid, conf['sequence_length'], color='b')
                object_track_l.append(np.stack(tr.pos, axis=0))
                del (tr)

            object_track = np.stack(object_track_l, axis=1)
            # print 'len object_tracks', len(object_tracks)
            # print 'object tracks[0].shape', object_tracks[0].shape

            print('press c to continue, r to repeat')
            done_enter = False
            while not done_enter:
                cmd = input('Input:')
                if cmd == 'c':
                    done_example = True
                    done_enter = True
                elif cmd == 'r':
                    done_enter = True
                    print('repeating current example')
                    pdb.set_trace()
                else:
                    print('wrong key')
                    print('press c to continue, r to repeat')

            filename = conf['pkl_dir'] + '/b{}.pkl'.format(b)
            print('saving to', filename)
            pickle.dump({'robot_pos':robot_track,
                          'object_pos':object_track,
                          'images':images[b],
                          'endeff_pos':endeff_pos[b],
                          'actions':actions[b],
                          },
                         open(filename, 'wb'))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tf_record(conf, filename):
    """
    saves data_files from one sample trajectory into one tf-record file
    """
    filename =os.path.join(conf['data_dest_dir'], filename)
    print(('Writing', filename))

    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for b in range(conf['batch_size']):

        dict = pickle.load(open(conf['pkl_dir'] + '/b{}.pkl'.format(b), "rb"))
        actions = dict['actions']
        endeff_pos = dict['endeff_pos']
        object_pos = dict['object_pos']
        robot_pos = dict['robot_pos']
        images = dict['images']

        sequence_length = conf['sequence_length']

        for tstep in range(sequence_length):

            feature[str(tstep) + '/action']= _float_feature(actions[tstep].tolist())
            feature[str(tstep) + '/endeffector_pos'] = _float_feature(endeff_pos[tstep].tolist())
            feature[str(tstep) + '/robot_pos'] = _int64_feature(robot_pos[tstep].tolist())
            object_pos_t = object_pos[tstep].flatten()
            feature[str(tstep) + '/object_pos'] = _int64_feature(object_pos_t.tolist())

            image_raw = (images[tstep]*255.).astype(np.uint8).tostring()  # for camera 0, i.e. main
            feature[str(tstep) + '/image_view0/encoded'] = _bytes_feature(image_raw)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Run annotation')
    parser.add_argument('testdatafile', type=str, help='test data tf records file')
    parser.add_argument('destdir', type=str, help='tfrecords destination dir')

    args = parser.parse_args()
    sourcefilename = args.testdatafile
    conf = {}

    import python_visual_mpc
    # data_source_dir = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2]) + '/pushing_data/wristrot_test_newobj/test'
    # conf['data_dir'] = data_source_dir  # 'directory containing data_files.'
    # data_dest_dir = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2]) + '/pushing_data/wristrot_test_newobj/test_annotations'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dest_dir'] = args.destdir
    # folder to
    conf['train_val_split'] = 1.
    conf['sequence_length'] = 15 #48  # 'sequence length, including context frames.'
    conf['batch_size'] = 128  ## **must** correspond to the number of trajectories in each tf records file!!
    conf['visualize'] = True # in order not to do train val splitting
    conf['img_height'] = 64
    conf['sdim'] = 4
    conf['adim'] = 5
    conf['skip_frame'] = 1

    conf['robot_pos'] = 1
    conf['object_pos'] = 2  #number of tracked positions on objects

    print('-------------------------------------------------------------------')
    print('verify current settings!! ')
    for key in list(conf.keys()):
        print(key, ': ', conf[key])
    print('-------------------------------------------------------------------')

    print('testing the reader')

    image_batch, action_batch, endeff_pos_batch = build_tfrecord_input(conf, training=True,
                                                                       input_file=sourcefilename)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    n_batches = 1

    i_traj_start = 0
    i_traj_end = conf['batch_size'] -1
    for b in range(n_batches):
        print('using batch', b)
        [images, actions, endeff_pos] = sess.run([image_batch, action_batch, endeff_pos_batch])

        ckpt_path = '/'.join(str.split(sourcefilename, '/')[:-2]) + '/annotation_ckpts'
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        sourcename = str.split(sourcefilename, '/')[-1]
        conf['pkl_dir'] = os.path.join(ckpt_path, sourcename)
        if not os.path.exists(conf['pkl_dir']):
            os.mkdir(conf['pkl_dir'])

        traj_data = [images, actions, endeff_pos]
        run_annotation(conf, traj_data)

        file_name = 'traj_{}_to_{}.tfrecords'.format(i_traj_start, i_traj_end)
        save_tf_record(conf, file_name)

        i_traj_start += conf['batch_size']
        i_traj_end += i_traj_start + conf['batch_size']-1

        pdb.set_trace()
        print('finished batch, press c + enter to continue')


if __name__ == '__main__':
    main()