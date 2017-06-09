import cPickle
import numpy
from video_prediction.utils_vpred.create_gif import *
from PIL import Image

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
    image_parts = dict_['image_parts']
    background_masks = dict_['background_masks']
    # generation_masks = dict_['generation_masks']
    trafos = dict_['trafos']

    print 'finished loading ...'

    img = create_images(object_masks, image_parts, examples)
    img = Image.fromarray(img)
    img.save(file_path +'/objectparts_masks.png')

    if not isinstance(ground_truth, list):
        ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
        ground_truth = [np.squeeze(g) for g in ground_truth]


    background_masks = [np.expand_dims(m, 0) for m in background_masks]
    [background_masks] = prepare_masks(background_masks)

    videolist = [ground_truth, gen_images,background_masks]
    suffix = ''

    fused_gif = assemble_gif(videolist, num_exp= examples)
    if not name:
        npy_to_gif(fused_gif, file_path + '/gen_images_pix_distrib'+ suffix)
    else:
        npy_to_gif(fused_gif, file_path + '/' + name + suffix)


def create_images(object_masks, image_parts, nexp):
    object_masks = [np.repeat(m, 3, axis=-1) for m in object_masks]
    rows = []

    num_objects = len(object_masks)
    for ob in range(num_objects):
        maskrow = []
        objectrow = []
        for ex in range(nexp):
            objectrow.append(image_parts[ob][ex])
            maskrow.append(object_masks[ob][ex])
        rows.append(np.concatenate(objectrow, axis=1))
        rows.append(np.concatenate(maskrow, axis=1))

    combined = (np.concatenate(rows, axis=0)*255.).astype(np.uint8)
    return combined

def prepare_masks(masks, copy_last_dim= True):
    tsteps = len(masks)
    nmasks = len(masks[0])
    list_of_maskvideos = []

    for m in range(nmasks):  # for timesteps
        mask_video = []
        for t in range(tsteps):
            if copy_last_dim:
                single_mask_batch = np.repeat(masks[t][m], 3, axis=3 )
            else:
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
    conf['visualize'] = conf['output_dir'] + '/model10002'


    comp_gif(conf, file_path + '/modeldata')