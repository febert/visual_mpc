import moviepy.editor as mpy
import os
from PIL import Image
import numpy as np

def npy_to_gif(im_list, filename):

    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print 'creating directory: ', save_dir
        os.mkdir(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=4)
    clip.write_gif(filename + '.gif')
    return


def get_images(folder):
    imlist = []
    for i in range(20):
        im = Image.open(folder + '/startimg__t{}.png'.format(i))
        imlist.append(np.asarray(im))

    return imlist

if __name__ == '__main__':
    folder = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/manual_correction/exp/small_object/videos'
    imlist  = get_images(folder)

    npy_to_gif(imlist, folder + "/manual_desig_pix")