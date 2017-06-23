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


def comp_gif(conf, file_path, name= "", examples = 10, show_parts=False):
    dict_ = cPickle.load(open(file_path + '/dict_.pkl', "rb"))

    gen_images = dict_['gen_images']

    trafos = dict_['trafos']
    comp_factors = dict_['comp_factors']
    comp_factors = [np.stack(c) for c in comp_factors]

    print 'finished loading ...'

    suffix = ''
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)

    if 'object_masks' in dict_:
        object_masks = dict_['object_masks']
        img = create_images(object_masks, examples)
        img = Image.fromarray(img)
        img.save(file_path +'/objectparts_masks{}.png'.format(itr_vis))

    videolist  =[]

    if 'ground_truth' in dict_:
        ground_truth = dict_['ground_truth']
        if not isinstance(ground_truth, list):
            ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
            ground_truth = [np.squeeze(g) for g in ground_truth]
        ground_truth = ground_truth[1:]

        videolist = [ground_truth]

    videolist.append(gen_images)

    if 'gen_pix_distrib' in dict_:
        gen_pix_distrib = dict_['gen_pix_distrib']
        plot_psum_overtime(gen_pix_distrib, examples,file_path+"/"+ name)
        videolist.append(make_color_scheme(gen_pix_distrib))

    if 'moved_pix_distrib' in dict_:
        moved_pix_distrib = dict_['moved_pix_distrib']
        # moved_pix_distrib_ = []
        # for t in range(len(moved_pix_distrib)):
        #     moved_pix_t = [m[:examples] for m in moved_pix_distrib[t]]
        #     moved_pix_distrib_.append(moved_pix_t)
        moved_pix_distrib = prepare_video(moved_pix_distrib, copy_last_dim=True)
        # moved_pix_distrib = [make_color_scheme(m) for m in moved_pix_distrib]
        videolist += moved_pix_distrib


    moved_images = dict_['moved_images']
    moved_images = prepare_video(moved_images, copy_last_dim=False)
    videolist += moved_images

    if 'gen_masks' in dict_:
        if dict_['gen_masks'] != []:
            gen_masks = dict_['gen_masks']
            videolist += prepare_video(gen_masks, copy_last_dim=True)

    if dict_['moved_masks'] != []:
        moved_masks = dict_['moved_masks']
        videolist += prepare_video(moved_masks, copy_last_dim=True)

    if show_parts:
        moved_parts = dict_['moved_parts']
        videolist += prepare_video(moved_parts, copy_last_dim=False)

    if 'dynamic_first_step_mask' in conf:
        first_step_masks = dict_['first_step_masks']
        videolist += prepare_video(first_step_masks, copy_last_dim=True)



    fused_gif = assemble_gif(videolist, num_exp= examples)
    npy_to_gif(fused_gif, file_path + '/' +name +'vid_'+itr_vis+ suffix)

def create_images(object_masks, nexp):
    object_masks = [np.repeat(m, 3, axis=-1) for m in object_masks]
    rows = []

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
    file_path = '/home/frederik/Documents/lsdc/tensorflow_data/occulsionmodel/CDNA_quad_sawyer_fullactions'
    hyperparams = imp.load_source('hyperparams', file_path +'/conf.py')

    conf = hyperparams.configuration
    conf['visualize'] = conf['output_dir'] + '/model78002'


    comp_gif(conf, file_path + '/modeldata', show_parts=True)