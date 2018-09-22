import os
from PIL import Image
import numpy as np
from scipy import misc


def extractFrames(inGif, outFolder):

    frame = Image.open(inGif)
    nframes = 0
    while frame:

        frame.save( 'splitted/im%i.png' % (nframes) , 'PNG')
        nframes += 1
        try:
            frame.seek( nframes )
        except EOFError:
            break;

    return nframes

def getFrames(ind):
    imlist = []
    for i in ind:
        imlist.append(np.asarray(misc.imread('splitted/im%i.png' % (i))))
    return imlist


if __name__ == '__main__':
    file = '/home/frederik/Documents/tactile_mpc_paper/image_data/gelsight/exp_mpc-2018-09-14:12:12:15/trajectories/gelsight/exp/full_benchmark/traj_data/record12/video12.gif'
    nframes = extractFrames(file, 'splitted')



