import glob
import moviepy.editor as mpy
import numpy as np


import os


from PIL import Image

# filenames = glob.glob('new1_9*.npy')
# np.random.seed(15)
# np.random.shuffle(filenames)

# def gif_to_npy(filename):
#     print filename
#     gif = Image.open(filename)
#     npy = np.zeros([8, 64, 64, 3], 'uint8')
#     for i in range(8):
#         gif.seek(i)
#         npy[i] = np.array(gif.convert('RGB'))
#     return npy

def npy_to_gif(im_list, filename):

    # import pdb; pdb.set_trace()

    clip = mpy.ImageSequenceClip(im_list, fps=10)
    clip.write_gif(filename)

if __name__ == '__main__':
    splitted = str.split(os.path.dirname(__file__), '/')
    file_path = '/'.join(splitted[:-3] + ['tensorflow_data/gifs'])


    import cPickle

    print 'reading files from:', file_path

    ground_truth = cPickle.load(open(file_path + '/ground_truth.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_image_seq.pkl', "rb"))


    collected_pairs = []
    num_exp =8
    for j in range(num_exp):

        ground_truth_list = list(np.uint8(255*ground_truth[j]))

        gen_image_list =[]
        for i in range(len(gen_images)):
            gen_image_list.append(np.uint8(255*gen_images[i][j]))

        print len(gen_image_list)
        print len(ground_truth_list)
        print gen_image_list[0].shape
        print ground_truth_list[0].shape

        column_list = [np.concatenate((truth, gen), axis=0)
                       for truth, gen in zip(ground_truth_list, gen_image_list)]

        collected_pairs.append(column_list)

    fused_gif = []
    for i in range(len(collected_pairs[0])):
        frame_list = [collected_pairs[j][i] for j in range(num_exp)]
        fused_gif.append(np.concatenate( tuple(frame_list), axis= 1))

    npy_to_gif(fused_gif, file_path + '/traj_no{0}.gif'.format(j))

    # npy_to_gif(ground_truth_list, file_path + '/groundtruth{0}.gif'.format(j))
    # npy_to_gif(gen_image_list, file_path + '/gen_images{0}.gif'.format(j))


