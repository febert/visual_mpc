import glob
import imageio
from PIL import Image
import numpy as np

import cPickle

def save_video_mp4(filename, frames):
    writer = imageio.get_writer(filename + '.mp4', fps=10)
    for frame in frames:
        writer.append_data(frame)
    writer.close()



def main():

    file = '/home/frederik/Documents/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop_1stimg_bckgd/exp/unseen_clutter/verbose/gen_image_t3.pkl'
    gen_images = cPickle.load(open(file, "rb"))

    gen_images = [im[0] for im in gen_images]
    save_video_mp4('vid_from_pkl', gen_images)

if __name__ == '__main__':
    main()

