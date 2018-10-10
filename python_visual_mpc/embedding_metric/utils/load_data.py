import os
from PIL import Image
import numpy as np
import glob
import cv2

def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_folders(dir):
    files = glob.glob(dir + '/*')
    folders = []
    for f in files:
        if os.path.isdir(f):
            folders.append(f)
    return sorted_alphanumeric(folders)

def load_benchmark_data(conf):

    bench_dir = conf['bench_dir']
    view = conf['view']
    if isinstance(bench_dir, list):
        folders = []
        for source_dir in bench_dir:
            folders += get_folders(source_dir)
    else:
        folders = get_folders(bench_dir)

    num_ex = conf['batch_size']
    folders = folders[:num_ex]

    image_batch = []
    desig_pix_t0_l = []
    goal_pix_l = []
    goal_image_l = []
    true_desig_l = []

    for folder in folders:
        name = str.split(folder, '/')[-1]
        print('example: ', name)
        exp_dir = folder + '/images{}'.format(view)

        imlist = []
        for t in range(30):
            imname = exp_dir + '/im_{}.jpg'.format(t)
            im = np.array(Image.open(imname))
            orig_imshape = im.shape
            im = cv2.resize(im, (conf['orig_size'][1], conf['orig_size'][0]), interpolation=cv2.INTER_AREA)
            imlist.append(im)

        images = np.stack(imlist).astype(np.float32) / 255.
        image_batch.append(images)

        image_size_ratio = conf['orig_size'][0]/orig_imshape[0]

        true_desig = np.load(folder + '/points.npy')
        true_desig = (true_desig[view]*image_size_ratio).astype(np.int)
        true_desig_l.append(true_desig)

        goal_image_l.append(images[-1])
        desig_pix_t0_l.append(true_desig[0])
        goal_pix_l.append(true_desig[-1])


    image_batch = np.stack(image_batch)
    desig_pix_t0 = np.stack(desig_pix_t0_l)
    goal_pix = np.stack(goal_pix_l)
    true_desig = np.stack(true_desig_l, axis=0)
    goal_images = np.stack(goal_image_l, axis=0)

    return image_batch, desig_pix_t0, goal_pix, true_desig, goal_images