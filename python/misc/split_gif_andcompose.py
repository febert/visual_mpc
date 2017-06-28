import os
from PIL import Image
import numpy as np

from scipy import misc



def extractFrames(inGif, outFolder):

    frame = Image.open(inGif)
    nframes = 0
    while frame:

        frame.save( 'splitted/im%i.png' % (nframes) , 'PNG')
        nframes += 1
        try:
            frame.seek(nframes)
        except EOFError:
            break;

    return nframes

def getFrames(file):
    nframes = extractFrames(file, 'splitted')

    imlist = []
    for i in range(nframes):
        imlist.append(np.asarray(misc.imread('splitted/im%i.png' % (i))))

    return imlist


def make_gen_pix():

    # file = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/cdna_multobj_1stimg/success/vid_cem_control_t7.gif'
    file ='/home/frederik/Documents/catkin_ws/src/lsdc/tensorflow_data/sawyer/1stimg_bckgd_cdna/modeldata/vid_rndaction_var10_64002_diffmotions_b0_l15.gif'

    gen_pix = getFrames(file)

    gen_pix_list = []
    for f in gen_pix:
        im = Image.fromarray(f)
        im = np.asarray(im)
        pic_row = 1
        im = im[:64*3,pic_row*64:(pic_row+1)*64]
        # im = Image.fromarray(im).show()
        gen_pix_list.append(im)

    fullimg = np.concatenate(gen_pix_list, axis=1)
    # imgpath = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/cdna_multobj_1stimg/success/gen_pix_overtime.png'
    imgpath = '/home/frederik/Documents/lsdc/tensorflow_data/sawyer/1stimg_bckgd_cdna/modeldata/gen_pixb0_overtime.png'
    print 'save to', imgpath
    Image.fromarray(fullimg).save(imgpath)


def make_highres():
    # file = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/cdna_multobj_1stimg/success/highres_traj0.gif'
    file = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop_1stimg_bckgd/videos/highres_traj0.gif'

    highres = getFrames(file)

    downsampled = []
    for f in highres:
        im = Image.fromarray(f)
        im.thumbnail([140, 140], Image.ANTIALIAS)

        im = np.asarray(im)
        im = im[0:90, 20:125]
        # im = Image.fromarray(im).show()
        downsampled.append(im)

    fullimg = np.concatenate(downsampled,axis=1)
    # imgpath = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/cdna_multobj_1stimg/success/highres_overtime.png'
    imgpath = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop_1stimg_bckgd/videos/highres_overtime.png'
    print 'save to', imgpath
    Image.fromarray(fullimg).save(imgpath)


if __name__ == '__main__':
    make_gen_pix()
    # make_highres()












