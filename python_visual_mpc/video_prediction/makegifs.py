import pickle
import numpy

from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
from PIL import Image
import colorsys


def comp_gif(conf, file_path, name= "", examples = 4, append_masks=False, suffix = ''):
    dict_ = pickle.load(open(file_path + '/pred.pkl','rb'))

    videolist = []

    if 'ground_truth' in dict_:
        ground_truth = dict_['ground_truth']
        if not isinstance(ground_truth, list):
            ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
            ground_truth = [np.squeeze(g) for g in ground_truth]

            ground_truth = ground_truth[1:]

        videolist.append(ground_truth)

    gen_images = dict_['gen_images']
    videolist.append(gen_images)

    colored_masks = False  # takes a long time to compute
    if append_masks:
        gen_masks = dict_['gen_masks']

        if colored_masks:
            gen_masks = convert_to_videolist(gen_masks, repeat_last_dim=False)
            gen_masks = [make_color_scheme(v) for v in gen_masks]
        else:
            gen_masks = convert_to_videolist(gen_masks, repeat_last_dim=True)
        videolist += gen_masks

    if 'gen_distrib' in dict_:
        gen_pix_distrib = dict_['gen_distrib']
        gen_pix_distrib = make_color_scheme(gen_pix_distrib)

        videolist.append(gen_pix_distrib)

    if 'flow_vectors' in dict_:
        videolist.append(visualize_flow(dict_))

    fused_gif = assemble_gif(videolist, num_exp= examples)
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    if not name:
        npy_to_gif(fused_gif, file_path + '/vid{}'.format(itr_vis) + suffix)
    else:
        npy_to_gif(fused_gif, file_path + '/' + name + suffix)


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

def convert_to_videolist(input, repeat_last_dim):
    tsteps = len(input)
    nmasks = len(input[0])

    list_of_videos = []

    for m in range(nmasks):  # for timesteps
        video = []
        for t in range(tsteps):
            if repeat_last_dim:
                single_mask_batch = np.repeat(input[t][m], 3, axis=3)
            else:
                single_mask_batch = input[t][m]
            video.append(single_mask_batch)
        list_of_videos.append(video)

    return list_of_videos


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

def visualize_flow(dict_):
    flow_vecs = dict_['flow_vectors']
    bsize = flow_vecs[0].shape[0]
    T = len(flow_vecs)

    magnitudes = [np.linalg.norm(f, axis=3) for f in flow_vecs]
    max_magnitude = np.max(magnitudes)
    norm_magnitudes = [m / max_magnitude for m in magnitudes]

    magnitudes = [np.expand_dims(m, axis=3) for m in magnitudes]

    #pixelflow vectors normalized for unit length
    norm_flow = [np.divide(f, m) for f, m in zip(flow_vecs, magnitudes)]
    flow_angle = [np.arctan2(p[:, :, :, 0], p[:, :, :, 1]) for p in norm_flow]
    color_flow = [np.zeros((bsize, 64, 64, 3)) for _ in range(T)]

    for t in range(T):
        for b in range(bsize):
            for r in range(64):
                for c in range(64):
                    color_flow[t][b, r, c] = colorsys.hsv_to_rgb((flow_angle[t][b, r, c] +np.pi) / 2 / np.pi,
                                                              norm_magnitudes[t][b, r, c],
                                                              1.)

    return color_flow


if __name__ == '__main__':
    file_path = '/home/febert/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/weissgripper_basecls_20k'
    hyperparams = imp.load_source('hyperparams', file_path +'/conf.py')

    conf = hyperparams.configuration
    conf['visualize'] = conf['output_dir'] + '/model92002'

    comp_gif(conf, file_path + '/modeldata', append_masks=True)
    # comp_gif(conf, file_path + '/modeldata', append_masks=True, suffix='diffmotions')