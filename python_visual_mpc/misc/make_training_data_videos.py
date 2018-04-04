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


    for g in range(2,11):
        imlist = []
        for t in range(30):
            print(t)
            itr = g*1000
            image_folder = '/mnt/sda1/sawyerdata/wrist_rot/main/traj_group{}/traj{}/images'.format(g, itr)
            '/images/main_full_cropped_im{}_*'.format(str(t).zfill(2))
            [imfile] = glob.glob(image_folder + "/main_full_cropped_im{}_*.jpg".format(str(t).zfill(2)))

            image = np.asarray(Image.open(imfile))

            imlist.append(image)

        destination = '/home/frederik/Documents/documentation/doc_video/video_clips/training_data'
        save_video_mp4(destination  +'/traj_{}_gr{}'.format(itr, g), imlist)

if __name__ == '__main__':
    main()

