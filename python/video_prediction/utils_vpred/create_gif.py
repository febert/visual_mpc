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

    ground_truth = cPickle.load(open(file_path + '/ground_truth.pkl', "rb"))
    gen_images = cPickle.load(open(file_path + '/gen_image_seq.pkl', "rb"))

    for j in range(4):

        ground_truth_list = list(np.uint8(255*ground_truth[j]))

        # from PIL import Image
        # for i in range( len(ground_truth)):
        #     img = Image.fromarray(ground_truth[i])
        #     img.show()

        gen_image_list =[]
        for i in range(len(gen_images)):
            gen_image_list.append(np.uint8(255*gen_images[i][j]))

        # for i in range(len(gen_image_list)):
        #     img = Image.fromarray(gen_image_list[i])
        #     img.show()


        print len(gen_image_list)
        print len(ground_truth_list)
        print gen_image_list[0].shape
        print ground_truth_list[0].shape

        npy_to_gif(ground_truth_list, file_path + '/groundtruth{0}.gif'.format(j))
        npy_to_gif(gen_image_list, file_path + '/gen_images{0}.gif'.format(j))


