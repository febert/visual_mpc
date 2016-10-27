import glob
import moviepy.editor as mpy
import numpy as np

from PIL import Image

filenames = glob.glob('new1_9*.npy')
np.random.seed(15)
np.random.shuffle(filenames)

def gif_to_npy(filename):
    print filename
    gif = Image.open(filename)
    npy = np.zeros([8, 64, 64, 3], 'uint8')
    for i in range(8):
        gif.seek(i)
        npy[i] = np.array(gif.convert('RGB'))
    return npy

def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)
    clip.write_gif(filename)

num_rows = 5
num_cols = 5
index = 0
for col in range(num_cols):
    for row in range(num_rows):
        # frames are T x H x W x 3, we will concat along H and then along W
        if row == 0:
            current_col = gif_to_npy(filenames[index])
            gt_name = 'old/gt_1_9_' + filenames[index][4:-4] + '.npy'
            gt_col = np.uint8(255*np.load(gt_name))
        else:
            current_col = np.concatenate([current_col, gif_to_npy(filenames[index])], 1)
            gt_name = 'old/gt_1_9_' + filenames[index][4:-4] + '.npy'
            gt_col = np.concatenate([gt_col, np.uint8(255*np.load(gt_name))], 1)
        index += 1
    if col == 0:
        current_image = current_col
        gt_image = gt_col
    else:
        current_image = np.concatenate([current_image, current_col], 2)
        gt_image = np.concatenate([gt_image, gt_col], 2)


gt_image = gt_image[1:9]
npy_to_gif(current_image, '/home/cfinn/test.gif')
npy_to_gif(gt_image, '/home/cfinn/testgt.gif'))