import glob
import moviepy.editor as mpy
import numpy as np
import cPickle
import os
import pdb
import imp
from PIL import Image
import re
import matplotlib.pyplot as plt


def npy_to_gif(im_list, filename):

    # save_dir = '/'.join(str.split(filename, '/')[:-1])
    #
    # if not os.path.exists(save_dir):
    #     print 'creating directory: ', save_dir
    #     os.mkdir(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=4)
    clip.write_gif(filename + '.gif')
    return


def comp_video(traj_list, file_path):
    vid_batch = []
    for cam in range(len(traj_list[0].cameranames)):


        img_list = []
        for t in range(traj_list[0].T):
            img_ex = []
            for tr in traj_list:
                img_ex.append(tr.images[t, cam])
            img_batch = np.stack(img_ex,axis=0)
            img_list.append(img_batch)

        vid_batch.append(img_list)


        # dimg_list = []
        # for t in range(traj_list[0].T):
        #     dimg_ex = []
        #     for tr in traj_list:
        #         dimg = tr.dimages[t, cam]
        #         dimg = np.expand_dims(dimg, axis=2)
        #         dimg = np.repeat(dimg, 3, axis= 2)
        #         dimg_ex.append(dimg)
        #     dimg_batch = np.stack(dimg_ex, axis=0)
        #     dimg_list.append(dimg_batch)
        #
        # vid_batch.append(dimg_list)

    fused_gif = assemble_gif(vid_batch)
    npy_to_gif(fused_gif, file_path)


def assemble_gif(video_batch):
    """
    accepts a list of different video batches
    each video batch is a list of [batchsize, 64, 64, 3] with length timesteps
    :param image_rows:
    :return:
    """


    num_exp = video_batch[0][0].shape[0]
    vid_length = min([len(vid) for vid in video_batch])
    print 'video length:', vid_length
    for i in range(len(video_batch)):
        video_batch[i] = [np.expand_dims(videoframe, axis=0) for videoframe in video_batch[i]]

    # import pdb; pdb.set_trace()
    for i in range(len(video_batch)):
        video_batch[i] = np.concatenate(video_batch[i], axis= 0)

    #videobatch is a list of [timelength, batchsize, 64, 64, 3]

    fullframe_list = []
    for t in range(vid_length):
        column_list = []
        for exp in range(num_exp):
            column_images = [video[t,exp] for video in video_batch]
            column_images = np.concatenate(column_images, axis=0)  #make column
            column_list.append(column_images)

        full_frame = np.concatenate(column_list, axis= 1)
        # full_frame = np.uint8(255 * full_frame)

        fullframe_list.append(full_frame)


    return fullframe_list



# if __name__ == '__main__':

