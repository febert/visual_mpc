import cPickle
import numpy
from video_prediction.utils_vpred.create_gif import *


def add_crosshairs(distrib, pix_list):
    """
    add crosshairs to video
    :param distrib:
    :param pix_list: list of x, y coords
    :return:
    """
    batch = pix_list[0].shape[0]
    for b in range(batch):
        for i in range(len(pix_list)):
            x, y = pix_list[i][b]
            distrib[i][b, x] = 0
            distrib[i][b, :, y] = 0

    return distrib


def comp_pix_distrib(file_path, name= None, masks = False, examples = 8):
    pix_distrib = cPickle.load(open(file_path + '/gen_distrib.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_images.pkl', "rb"))
    gtruth_images = cPickle.load(open(file_path + '/gtruth_images.pkl', "rb"))
    maxcoord = cPickle.load(open(file_path + '/maxcoord.pkl', "rb"))

    print 'finished loading'

    pix_distrib = make_color_scheme(pix_distrib)
    pix_distrib = add_crosshairs(pix_distrib, maxcoord)

    videolist = [gtruth_images, gen_images, pix_distrib]

    suffix = ''
    if masks:
        gen_masks = cPickle.load(open(file_path + '/gen_masks.pkl', "rb"))
        mask_videolist = []
        nummasks = len(gen_masks[0])
        tsteps = len(gen_masks)
        for m in range(nummasks):
            mask_video = []
            for t in range(tsteps):
                 mask_video.append(np.repeat(gen_masks[t][m], 3, axis=3))

            mask_videolist.append(mask_video)
        videolist += mask_videolist
        suffix = "_masks"

    fused_gif = assemble_gif(videolist, num_exp= examples)
    if not name:
        npy_to_gif(fused_gif, file_path + '/gen_images_pix_distrib'+ suffix)
    else:
        npy_to_gif(fused_gif, file_path + '/' + name + suffix)


if __name__ == '__main__':
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/retina/static/modeldata'
    comp_pix_distrib(file_path)