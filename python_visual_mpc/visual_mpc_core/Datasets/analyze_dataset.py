from python_visual_mpc.visual_mpc_core.Datasets.base_dataset import BaseVideoDataset
import argparse
import moviepy.editor as mpy
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize good and bad trajectories from tfrecord dataset')
    parser.add_argument('base_path', type=str, help="path to folder containing good/bad Dataset folders")
    parser.add_argument('--view', type=int, default=0, help="0 for view0 1 for view1")
    args = parser.parse_args()

    batch_size = 5
    good_dataset, bad_dataset = BaseVideoDataset('{}/good'.format(args.base_path), batch_size), BaseVideoDataset('{}/bad'.format(args.base_path), batch_size)
    good_images, bad_images = good_dataset['images'], bad_dataset['images']
    good_states, good_actions = good_dataset['state'], good_dataset['actions']

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    good_images, bad_images, good_states, good_actions = sess.run([good_images, bad_images, good_states, good_actions])
    T = good_images.shape[1]
    for i in range(batch_size):
        clip = mpy.ImageSequenceClip([good_images[i, t, args.view, :, :] for t in range(T)], fps = 5)
        clip.write_gif('good_{}.gif'.format(i))

        clip = mpy.ImageSequenceClip([bad_images[i, t, args.view, :, :] for t in range(T)], fps = 5)
        clip.write_gif('bad_{}.gif'.format(i))

    for i in range(batch_size):
        print('actions')
        print(good_actions[0])
        print('states')
        print(good_states[0])
