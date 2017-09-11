import imageio
from PIL import Image
import numpy as np

import glob

image_folder = 'testdata/img'

writer = imageio.get_writer('testdata/testvideo.mp4', fps=10)

for i_save in range(96):
    [imfile] = glob.glob(image_folder + "/main_full_cropped_im{}_*.jpg".format(str(i_save).zfill(2)))
    im = np.asarray(Image.open(imfile))
    writer.append_data(im)

writer.close()