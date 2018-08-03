from python_visual_mpc.visual_mpc_core.Datasets.base_dataset import BaseVideoDataset
import argparse
import moviepy.editor as mpy
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize good and bad trajectories from tfrecord dataset')
    parser.add_argument('base_path', type=str, help="path to folder containing good/bad Dataset folders")
    args = parser.parse_args()

    batch_size = 30
    good_dataset, bad_dataset = BaseVideoDataset('{}/good'.format(args.base_path), batch_size), BaseVideoDataset('{}/bad'.format(args.base_path), batch_size)
    good_images, bad_images = good_dataset['images'], bad_dataset['images']

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    good_images, bad_images = sess.run([good_images, bad_images])
    T = good_images.shape[1]
    for i in range(batch_size):
        clip = mpy.ImageSequenceClip([good_images[i, t, 0, :, :] for t in range(T)], fps = 5)
        clip.write_gif('good_{}.gif'.format(i))

        clip = mpy.ImageSequenceClip([bad_images[i, t, 0, :, :] for t in range(T)], fps = 5)
        clip.write_gif('bad_{}.gif'.format(i))
        
