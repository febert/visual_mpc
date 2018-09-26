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
        print('reading splitted/im%i.png' % (i))
        imlist.append(np.asarray(misc.imread('splitted/im%i.png' % (i))))
    return imlist

def crop(imlist, rowstart, rowwidth, colstart, colwidth):

    outlist = []
    for im in imlist:
        outlist.append(im[rowstart:rowstart+rowwidth, colstart:colstart+colwidth])
    return outlist


def disassemble_planning():
    file = '/home/frederik/Documents/tactile_mpc_paper/joystick_qulitative/joystick/interesting_traj/record3/plan/direct_t1iter2.gif'
    nframes = extractFrames(file, 'splitted')

    last_iter = getFrames(range(13))
    last_iter = crop(last_iter, 48, 48, 200, 64)
    last_iter = make_diff(last_iter)

    last_iter = np.concatenate(last_iter, axis=1)
    Image.fromarray(last_iter).save('pred1.png', 'PNG')


    last_iter = getFrames([0])
    last_iter = crop(last_iter, 0, 48, 200, 64)
    last_iter = np.concatenate(last_iter, axis=1)
    Image.fromarray(last_iter).save('goal.png', 'PNG')

    file = '/home/frederik/Documents/tactile_mpc_paper/joystick_qulitative/joystick/interesting_traj/record3/video3.gif'
    nframes = extractFrames(file, 'splitted')
    ind = range(1,12,2)
    print(ind)
    last_iter = getFrames(ind)
    last_iter = np.concatenate(last_iter, axis=1)
    Image.fromarray(last_iter).save('real.png', 'PNG')


def make_diff(list):
    out = []
    for i in range(1,len(list)):
        sq_diff = np.square(list[i]-list[0]).astype(np.float32)
        out.append((sq_diff/np.max(sq_diff)*255.).astype(np.uint8))

    return out

def arange_pred():

    num_ex = 4

    rows = []
    for ex in range(num_ex):
        file = '/home/frederik/Documents/tactile_mpc_paper/extracted_images/pred_gtruth_compare/gtruth.gif'
        rows.append(get_row(ex, file))

        file = '/home/frederik/Documents/tactile_mpc_paper/extracted_images/pred_gtruth_compare/genimage.gif'
        rows.append(get_row(ex, file))

        if ex < num_ex -1:
            rows.append(np.ones([10, rows[0].shape[1], 3], dtype=np.uint8)*255)

    image = np.concatenate(rows, axis=0)
    Image.fromarray(image).save('comppred.png', 'PNG')

def get_row(ex, file):
    extractFrames(file, 'splitted')
    frames = getFrames(range(0,15,2))
    frames = [f[...,:3]  for f in frames]
    last_iter = crop(frames, 0, 48, ex * 64,  64)
    return np.concatenate(last_iter, axis=1)

if __name__ == '__main__':
    disassemble_planning()
    # arange_pred()
