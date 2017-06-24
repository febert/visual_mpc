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
    nframes = extractFrames('traj0_gr0_withpixdistrib.gif', 'splitted')


    ind = range(4,int(nframes/2)+10, 5*2)
    print ind
    last_iter = getFrames(ind)
    last_iter = np.concatenate(last_iter, axis=1)
    Image.fromarray(last_iter).save('last_iter_{0}to{1}.png'.format(ind[0], ind[-1]), 'PNG')

    ind = range(ind[-1], nframes, 5 * 2)
    print ind
    last_iter = getFrames(ind)
    last_iter = np.concatenate(last_iter, axis=1)
    Image.fromarray(last_iter).save('last_iter_{0}to{1}.png'.format(ind[0], ind[-1]), 'PNG')

    ind = range(5,10)
    print ind
    last_iter = getFrames(ind)
    last_iter = np.concatenate(last_iter, axis=1)
    Image.fromarray(last_iter).save('show_iter.png', 'PNG')


    # show_iter = imlist[5:10]
    # show_iter = np.concatenate(show_iter, axis=1)
    # Image.fromarray(last_iter).save('show_iter', 'PNG')



