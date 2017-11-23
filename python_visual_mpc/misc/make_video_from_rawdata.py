import glob
import imageio
from PIL import Image
import numpy as np

def save_video_mp4(filename, frames):
    writer = imageio.get_writer(filename + '.mp4', fps=10)
    for frame in frames:
        writer.append_data(frame)
    writer.close()



def main():

    imlist = []
    for t in range(30):
        image_folder = '/mnt/sda1/sawyerdata/wristrot_test_seenobj/main/traj_group0/traj28/images'
        [imfile] = glob.glob(image_folder + "/main_full_cropped_im{}_*.jpg".format(str(t).zfill(2)))

        image = np.asarray(Image.open(imfile))

        imlist.append(image)

    save_video_mp4('rawdata_video', imlist)

if __name__ == '__main__':
    main()

