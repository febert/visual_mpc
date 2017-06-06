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


def comp_pix_distrib(conf, file_path, name= None, masks = False, examples = 10):
    dict_ = cPickle.load(open(file_path + '/dict_.pkl', "rb"))

    gen_images =dict_['gen_images']
    if 'true_ret' in dict_:
        true_ret = dict_['true_ret']
    pred_ret = dict_['pred_ret']
    images = dict_['images']
    gen_distrib = dict_['gen_distrib']
    retpos = dict_['retpos']


    print 'finished loading ...'

    gen_distrib = make_color_scheme(gen_distrib)
    # gen_pix_distrib = add_crosshairs(gen_pix_distrib, maxcoord)

    if 'true_ret' in dict_:
        true_ret = pad_pos(conf, true_ret, retpos)
    pred_ret = pad_pos(conf, pred_ret, retpos)

    if not isinstance(images, list):
        images = np.split(images, images.shape[1], axis=1)
        images = [np.squeeze(g) for g in images]

    if 'true_ret' in dict_:
        videolist = [images, gen_images, true_ret, pred_ret, gen_distrib]
    else:
        videolist = [images, gen_images, pred_ret, gen_distrib]

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
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/costmask/moving_retina'
    hyperparams = imp.load_source('hyperparams', file_path +'/conf.py')

    conf = hyperparams.configuration
    conf['visualize'] = conf['output_dir'] + '/model48002'


    comp_pix_distrib(conf, file_path + '/modeldata')