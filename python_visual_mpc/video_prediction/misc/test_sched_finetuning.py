import tensorflow as tf
from copy import deepcopy
import os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from python_visual_mpc.video_prediction.read_tf_record_wristrot import \
                        build_tfrecord_input as build_tfrecord_fn


from python_visual_mpc.video_prediction.basecls.prediction_model_basecls import mix_datasets


def main():
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))

    DATA_BASE_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/pushing_data'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 14 #48      # 'sequence length, including context frames.'
    conf['batch_size']= 10
    conf['visualize']= True
    conf['context_frames'] = 2

    conf['img_height'] = 54
    conf['img_width'] = 64
    conf['sdim'] = 4
    conf['adim'] = 5

    conf['color_augmentation']=""
    conf['scheduled_finetuning']=[DATA_BASE_DIR + '/weiss_gripper_20k/train', DATA_BASE_DIR + '/online_data1/train']

    conf['color_augmentation'] = ''

    # mixing ratio num(dataset1)/num(dataset2) in batch
    dataset_01ratio = tf.placeholder(tf.float32, shape=[], name="dataset_01ratio")
    d0_conf = deepcopy(conf)  # the larger source dataset
    d0_conf['data_dir'] = conf['scheduled_finetuning'][0]
    d0_train_images, d0_train_actions, d0_train_states = build_tfrecord_fn(d0_conf, training=True)

    d1_conf = deepcopy(conf)
    d1_conf['data_dir'] = conf['scheduled_finetuning'][1]
    d1_train_images, d1_train_actions, d1_train_states = build_tfrecord_fn(d1_conf, training=True)

    train_images, train_actions, train_states = mix_datasets([d0_train_images, d0_train_actions, d0_train_states],
                                                             [d1_train_images, d1_train_actions, d1_train_states],
                                                             conf['batch_size'],
                                                             dataset_01ratio)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video

    for i_run in range(1):
        print('run number ', i_run)

        images, actions, endeff = sess.run([train_images, train_actions, train_states],{dataset_01ratio:0.5})

        file_path = '/'.join(str.split(conf['scheduled_finetuning'][0], '/')[:-1] + ['mixed'])
        comp_single_video(file_path, images, num_exp=conf['batch_size'])

        # show some frames
        for b in range(conf['batch_size']):
            print('b',b)
            print('actions')
            print(actions[b])

            print('endeff')
            print(endeff[b])

            # print 'video mean brightness', np.mean(images[b])
            # if np.mean(images[b]) < 0.25:
            #     print b
            #     plt.imshow(images[b, 0])
            #     plt.show()


if __name__ == '__main__':
    main()