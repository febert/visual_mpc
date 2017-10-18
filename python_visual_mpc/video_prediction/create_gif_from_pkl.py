import numpy as np
import moviepy.editor as mpy
import cPickle as pickle
import os
import matplotlib.pyplot as plt

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

    data_dict = pickle.load(open(FLAGS.pkl_path, 'rb'))
    gen_images = data_dict['gen_images']

    for i in range(gen_images[0].shape[0]):
        frames = [x[i] for x in gen_images]
        npy_to_gif(frames, FLAGS.out_path +'batch' + str(i))

    # if 'gen_distrib' in data_dict:
    #     gen_distrib = data_dict['gen_distrib']
    #
    #     for i in range(gen_distrib[0].shape[0]):
    #         frames = [x[i] for x in gen_distrib]
    #         npy_to_gif(frames, FLAGS.out_path + 'distrib_batch' + str(i))

if __name__ == '__main__':
    main()