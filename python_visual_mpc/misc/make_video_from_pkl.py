import glob
import imageio
from PIL import Image
import numpy as np

import cPickle

def save_video_mp4(filename, frames):
    writer = imageio.get_writer(filename + '.mp4', fps=10)
    for i, frame in enumerate(frames):
        print 'frame',i
        writer.append_data(frame)
    writer.close()

def main():

    # file = '/home/frederik/Documents/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop_1stimg_bckgd/exp/unseen_clutter/verbose/gen_image_t3.pkl'

    for cem_dir in range(5):
        filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks_sawyer/wristrot/videos/record_cem{}'.format(cem_dir)
        print 'processing dir ', filedir
        dict = cPickle.load(open(filedir+'/pred.pkl', "rb"))

        dest_dir = filedir

        t = 3

        for b in range(10):
            # for itr in range(3):
            itr = 2
            gen_images = dict['gen_images_t{}_iter{}'.format(t,itr)]
            gen_images = [im[b] for im in gen_images]
            save_video_mp4(dest_dir + '/gen_images_b{}_t{}_iter{}'.format(b, t,itr), gen_images)

if __name__ == '__main__':
    main()

