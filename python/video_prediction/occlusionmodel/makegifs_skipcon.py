import cPickle
import numpy
from video_prediction.utils_vpred.create_gif import *
from PIL import Image
from matplotlib import pyplot as plt


def comp_gif(conf, file_path, name= "", examples = 10, show_parts=False):
    dict_ = cPickle.load(open(file_path + '/dict_.pkl', "rb"))
    print 'finished loading ...'

    suffix = ''
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)

    gen_images = dict_['gen_images']

    total_length = len(gen_images)


    videolist  =[]

    if 'ground_truth' in dict_:
        ground_truth = dict_['ground_truth']
        if not isinstance(ground_truth, list):
            ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
            ground_truth = [np.squeeze(g) for g in ground_truth]
        ground_truth = ground_truth[1:]

        videolist = [ground_truth]
        print 'adding ground_truth'


    videolist.append(gen_images)
    print 'adding gen_images'

    moved_images = dict_['moved_imagesl']
    moved_images = prepare_video(moved_images, copy_last_dim=False)
    videolist += moved_images

    comp_masks_l = dict_['comp_masks_l']
    comp_masks_l = prepare_video(comp_masks_l, copy_last_dim=True)
    videolist += comp_masks_l

    accum_Images_l = dict_['accum_Images_l']
    accum_Images_l = prepare_video(accum_Images_l, copy_last_dim=False)
    videolist += accum_Images_l


    accum_masks_l = dict_['accum_masks_l']
    accum_l = len(accum_masks_l)
    app_zeros = []
    for i in range(total_length - accum_l):
        app_zeros.append([np.zeros((32,64,64,1)) for _ in range(len(accum_masks_l[0]))])
    accum_masks_l += app_zeros

    accum_masks_l = prepare_video(accum_masks_l, copy_last_dim=True)
    videolist += accum_masks_l

    fused_gif = assemble_gif(videolist, num_exp= examples)



    npy_to_gif(fused_gif, file_path + '/' +name +'vid_'+itr_vis+ suffix)

    # npy_to_gif(fused_gif[:10], file_path + '/context' + name + 'vid_' + itr_vis + suffix)
    # npy_to_gif(fused_gif[10:], file_path + '/pred' + name + 'vid_' + itr_vis + suffix)



    pdb.set_trace()

def create_images(object_masks, nexp, background=None, background_mask=None):
    object_masks = [np.repeat(m, 3, axis=-1) for m in object_masks]

    rows = []
    if background != None:
        background_mask = np.repeat(background_mask, 3, axis=-1)
        background = np.split(background, background.shape[0], 0)
        background = [np.squeeze(b) for b in background]
        background_mask = np.split(background_mask, background_mask.shape[0], 0)
        background_mask = [np.squeeze(b) for b in background_mask]
        rows.append(np.concatenate(background, 1))
        rows.append(np.concatenate(background_mask, 1))

    num_objects = len(object_masks)
    for ob in range(num_objects):
        maskrow = []
        for ex in range(nexp):
            maskrow.append(object_masks[ob][ex])
        rows.append(np.concatenate(maskrow, axis=1))

    combined = (np.concatenate(rows, axis=0)*255.).astype(np.uint8)
    return combined

def plot_psum_overtime(gen_distrib, n_exp, name):
    plt.figure(figsize=(25, 2),dpi=80)

    for ex in range(n_exp):
        psum = []
        plt.subplot(1,n_exp, ex+1)
        for t in range(len(gen_distrib)):
            psum.append(np.sum(gen_distrib[t][ex]))

        psum = np.array(psum)
        plt.plot(range(len(gen_distrib)), psum)
        plt.ylim([0,2.5])

    # plt.show()
    plt.savefig(name +"_psum.png")
    plt.close('all')


def prepare_video(masks, copy_last_dim):
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

if __name__ == '__main__':
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/occulsionmodel/skipcon_window/1st_test'
    hyperparams = imp.load_source('hyperparams', file_path +'/conf.py')

    conf = hyperparams.configuration
    conf['visualize'] = conf['output_dir'] + '/model54002'


    comp_gif(conf, file_path + '/modeldata', show_parts=True)