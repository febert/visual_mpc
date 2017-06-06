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


def comp_pix_distrib(conf, file_path, masks = False, examples = 10):
    dict_ = cPickle.load(open(file_path + '/dict_.pkl', "rb"))

    gen_retinas =dict_['gen_retinas']
    gtruth_retinas = dict_['gtruth_retinas']
    val_highres = dict_['val_highres_images']
    gen_pix_distrib = dict_['gen_pix_distrib']
    maxcoord = dict_['maxcoord']
    retina_pos = dict_['retina_pos']

    print 'finished loading'

    gen_pix_distrib = make_color_scheme(gen_pix_distrib)
    gen_pix_distrib = add_crosshairs(gen_pix_distrib, maxcoord)

    gtruth_retinas = pad_pos(gtruth_retinas, retina_pos)
    gen_retinas = pad_pos(gen_retinas, retina_pos)
    gen_pix_distrib = pad_pos(gen_pix_distrib, retina_pos)

    if not isinstance(val_highres, list):
        val_highres = np.split(val_highres, val_highres.shape[1], axis=1)
        val_highres = [np.squeeze(g) for g in val_highres]

    videolist = [val_highres, gtruth_retinas, gen_retinas, gen_pix_distrib]

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

    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    name = file_path + '/vid_' + conf['experiment_name'] + '_' + str(itr_vis) + suffix

    fused_gif = assemble_gif(videolist, num_exp= examples)
    npy_to_gif(fused_gif, name)


def pad_pos(vid, pos):

    batch = vid[0].shape[0]
    padded_vid = [np.zeros([batch, 80, 80, 3]) for _ in range(len(vid))]
    for b in range(batch):
        for t in range(len(vid)):
            rstart = pos[t][b,0] - 16
            rend = pos[t][b,0] + 16
            cstart = pos[t][b,1] - 16
            cend = pos[t][b,1] + 16
            padded_vid[t][b, rstart:rend, cstart:cend] = vid[t][b]

    return padded_vid

if __name__ == '__main__':
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/retina/fromstatic/modeldata'
    comp_pix_distrib(file_path)