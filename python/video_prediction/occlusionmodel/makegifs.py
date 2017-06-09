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


def comp_gif(conf, file_path, name= None, examples = 10):
    dict_ = cPickle.load(open(file_path + '/dict_.pkl', "rb"))

    ground_truth =dict_['ground_truth']
    gen_images = dict_['gen_images']
    object_masks = dict_['object_masks']
    background_masks = dict_['background_masks']
    generation_masks = dict_['generation_masks']
    trafos = dict_['trafos']

    print 'finished loading ...'

    # generation_masks = prepare_masks(generation_masks)

    # object_mask_sum = np.sum(np.stack(object_masks, axis=0), axis=0)

    #copy object masks over timesteps:
    object_masks = [object_masks for _ in object_masks]
    object_masks = prepare_masks(object_masks)

    object_masks = prepare_masks(object_masks)


    videolist = [ground_truth, gen_images, object_masks, object_masks, background_masks, generation_masks]
    suffix = ''

    fused_gif = assemble_gif(videolist, num_exp= examples)
    if not name:
        npy_to_gif(fused_gif, file_path + '/gen_images_pix_distrib'+ suffix)
    else:
        npy_to_gif(fused_gif, file_path + '/' + name + suffix)


def prepare_masks(masks):
    tsteps = len(masks)
    nmasks = len(masks[0])
    list_of_maskvideos = []

    for m in range(nmasks):  # for timesteps
        mask_video = []
        for t in range(tsteps):
            # single_mask_batch = np.repeat(masks[t][m], 3, axis=3 )
            single_mask_batch = masks[t][m]
            mask_video.append(single_mask_batch)
        list_of_maskvideos.append(mask_video)

    return list_of_maskvideos


def pad_pos(conf, vid, pos, origsize = 64):

    batch = vid[0].shape[0]
    padded_vid = [np.zeros([batch, origsize, origsize, 3]) for _ in range(len(vid))]

    retina_size = conf['retina_size']
    halfret = retina_size /2

    for b in range(batch):
        for t in range(len(vid)):
            rstart = pos[t][b,0] - halfret
            rend = pos[t][b,0] + halfret + 1
            cstart = pos[t][b,1] - halfret
            cend = pos[t][b,1] + halfret + 1
            padded_vid[t][b, rstart:rend, cstart:cend] = vid[t][b]

    return padded_vid

if __name__ == '__main__':
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/occulsionmodel/firsttest'
    hyperparams = imp.load_source('hyperparams', file_path +'/conf.py')

    conf = hyperparams.configuration
    conf['visualize'] = conf['output_dir'] + '/model28002'


    comp_gif(conf, file_path + '/modeldata')