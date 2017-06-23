import numpy as np
import cPickle
import os
from video_prediction.sawyer.read_tf_record_sawyer import build_tfrecord_input
import tensorflow as tf
import matplotlib.pyplot as plt


def create_one_hot(conf, desig_pix):
    one_hot = np.zeros((1, 64, 64, 1), dtype=np.float32)
    # switch on pixels
    one_hot[0, desig_pix[0], desig_pix[1]] = 1.
    one_hot = np.repeat(one_hot, conf['context_frames'], axis=0)
    app_zeros = np.zeros((conf['sequence_length'] - conf['context_frames'], 64, 64, 1), dtype=np.float32)
    one_hot = np.concatenate([one_hot, app_zeros], axis=0)

    return one_hot


class Getdesig(object):
    def __init__(self,img,filepath):
        self.filepath = filepath
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        self.coords = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        self.coords = np.array([event.ydata, event.xdata]).astype(np.int32)
        self.ax.scatter(self.coords[1], self.coords[0], s=60, facecolors='none', edgecolors='b')
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.draw()
        plt.savefig(self.filepath)

def create_canoncical_examples(file_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/pushing_data/softmotion30/test'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split'] = 0.95
    conf['sequence_length'] = 15  # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['batch_size'] = 20
    conf['visualize'] = True
    conf['single_view'] = ''
    conf['context_frames'] = 2

    image_aux_batch, action_batch, endeff_pos_batch = build_tfrecord_input(conf, training=False)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    images, actions, endeff = sess.run([image_aux_batch, action_batch, endeff_pos_batch])
    num_ex = 10

    for ex in range(num_ex):
        c = Getdesig(images[ex, 0], file_path +'/desig_pix_img/img{}'.format(ex))
        desig_pix = c.coords

        init_pix_distrib = (create_one_hot(conf, desig_pix))

        dict = {}
        dict['desig_pix'] = desig_pix
        dict['init_pix_distrib'] =init_pix_distrib
        dict['images'] = images[ex]
        dict['actions'] = actions[ex]
        dict['endeff'] = endeff[ex]

        cPickle.dump(dict, open(file_path +'/pkl/example{}.pkl'.format(ex), 'wb'))

if __name__ == '__main__':
    file_path = '/home/frederik/Documents/catkin_ws/src/lsdc/pushing_data/canonical_examples'
    create_canoncical_examples(file_path)