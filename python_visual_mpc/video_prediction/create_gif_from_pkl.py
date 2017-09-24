import numpy as np
import moviepy.editor as mpy
import cPickle as pickle
import os


def npy_to_gif(im_list, filename):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print 'creating directory: ', save_dir
        os.mkdir(save_dir)

    images = [(255 * i).astype(np.uint8) for i in im_list]
    clip = mpy.ImageSequenceClip(images, fps=4)
    clip.write_gif(filename + '.gif')
    return

def main():
    from tensorflow.python.platform import flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('pkl_path', './', 'path to pkl file')
    flags.DEFINE_string('out_path', './', 'path to save gif files')

    gen_images = pickle.load(open(FLAGS.pkl_path, 'rb'))['gen_images']

    for i in range(gen_images[0].shape[0]):
        frames = [x[i] for x in gen_images]
        npy_to_gif(frames, FLAGS.out_path +'frames' + str(i))


if __name__ == '__main__':
    main()