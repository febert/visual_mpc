import os
from PIL import Image
import numpy as np

from scipy import misc
import copy
import imageio

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
    pic_col = 2

    for f in gen_pix:
        im = Image.fromarray(f)
        im = np.asarray(im)

        im = im[:64*3,pic_col*64:(pic_col+1)*64]
        # im = Image.fromarray(im).show()
        gen_pix_list.append(im)

    fullimg = np.concatenate(gen_pix_list, axis=1)
    # imgpath = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/cdna_multobj_1stimg/success/gen_pix_overtime.png'
    imgpath = '/home/frederik/Documents/lsdc/tensorflow_data/sawyer/1stimg_bckgd_cdna/modeldata/gen_pixb0_overtime_col{}.png'.format(pic_col)
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


def put_genpix_in_frame():
    file = '/home/guser/catkin_ws/src/lsdc/tensorflow_data/sawyer/dna_correct_nummask/vid_rndaction_var10_66002_diffmotions_b0_l30.gif'
    frames_dna = getFrames(file)

    file = '/home/guser/catkin_ws/src/lsdc/tensorflow_data/sawyer/1stimg_bckgd_cdna/vid_rndaction_var10_64002_diffmotions_b0_l30.gif'
    frames_cdna = getFrames(file)


    t = 1
    dest_path = '/home/guser/frederik/doc_video'

    frame = Image.open(dest_path + '/frame_comp_oadna.png', mode='r')

    writer = imageio.get_writer(dest_path + '/genpix_withframe.mp4', fps=3)

    pic_path = dest_path + "/animated"
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)


    for i, img_dna, img_cdna in zip(range(len(frames_dna)), frames_dna, frames_cdna):
        newimg = copy.deepcopy(np.asarray(frame)[:, :, :3])

        img_dna, size_insert = resize(img_dna)
        # Image.fromarray(img_dna)
        startr = 230
        startc = 650
        newimg[startr:startr + size_insert[0], startc: startc + size_insert[1]] = img_dna

        img_cdna, size_insert = resize(img_cdna)
        # Image.fromarray(img_cdna)
        startr = 540
        startc = 650
        newimg[startr:startr + size_insert[0], startc: startc + size_insert[1]] = img_cdna

        writer.append_data(newimg)
        Image.fromarray(newimg).save(pic_path + '/img{}.png'.format(i))

    writer.close()


def resize(img):
    img = img[:,64:64*2]
    origsize = img.shape
    img = Image.fromarray(img)
    img = img.resize((origsize[1] * 2, origsize[0] * 2), Image.ANTIALIAS)
    img = np.asarray(img)
    size_insert = img.shape
    return img, size_insert


if __name__ == '__main__':
    make_gen_pix()
    # make_highres()

    # put_genpix_in_frame()












