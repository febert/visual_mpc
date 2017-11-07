import numpy as np
from matplotlib import animation
import matplotlib.gridspec as gridspec

import imageio
import os
import pdb
import cPickle

import Tkinter as Tk

from Tkinter import Button, Frame, Canvas, Scrollbar
import Tkconstants

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import colorsys

from python_visual_mpc.video_prediction.misc.makegifs2 import assemble_gif, npy_to_gif
import scipy.misc
frame = None
canvas = None

def plot_psum_overtime(gen_distrib, n_exp, filename):
    plt.figure(figsize=(25, 2),dpi=80)

    for ex in range(n_exp):
        psum = []
        plt.subplot(1,n_exp, ex+1)
        for t in range(len(gen_distrib)):
            psum.append(np.sum(gen_distrib[t][ex]))

        psum = np.array(psum)
        plt.plot(range(len(gen_distrib)), psum)
        plt.ylim([0,2.5])

    # plt.show()
    plt.savefig(filename + "/psum.png")
    plt.close('all')


def plot_normed_at_desig_pos(gen_distrib, filename, desig_pos):
    plt.figure(figsize=(5, 5),dpi=80)
    b_exp = 1
    p = []

    for t in range(len(gen_distrib)):
        p.append(gen_distrib[t][b_exp][desig_pos[0], desig_pos[1]]/ np.sum(gen_distrib[t][b_exp]))

    psum = np.array(p)
    plt.plot(range(len(gen_distrib)), psum, 'g-o',)

    plt.xticks(range(0, len(gen_distrib),3))


    # plt.rcParams.update({'font.size': 62})
    a = plt.ylabel('probability of designated pixel')
    a.set_fontsize(14)

    b = plt.xlabel('time steps')
    b.set_fontsize(14)

    # plt.ylim([0,2.5])

    # plt.show()
    plt.savefig(filename + "/p_at_desig.png")
    plt.close('all')


def visualize_flow(flow_vecs):
    bsize = flow_vecs[0].shape[0]
    T = len(flow_vecs)

    magnitudes = [np.linalg.norm(f, axis=3) + 1e-5 for f in flow_vecs]
    max_magnitude = np.max(magnitudes)
    norm_magnitudes = [m / max_magnitude for m in magnitudes]

    magnitudes = [np.expand_dims(m, axis=3) for m in magnitudes]

    #pixelflow vectors normalized for unit length
    norm_flow = [np.divide(f, m) for f, m in zip(flow_vecs, magnitudes)]
    flow_angle = [np.arctan2(p[:, :, :, 0], p[:, :, :, 1]) for p in norm_flow]
    color_flow = [np.zeros((bsize, 64, 64, 3)) for _ in range(T)]

    for t in range(T):
        for b in range(bsize):
            for r in range(64):
                for c in range(64):
                    color_flow[t][b, r, c] = colorsys.hsv_to_rgb((flow_angle[t][b, r, c] +np.pi) / 2 / np.pi,
                                                              norm_magnitudes[t][b, r, c],
                                                              1.)

    return color_flow


t = 0
class Visualizer_tkinter(object):
    def __init__(self, dict_ = None, append_masks = True, filepath=None, numex = 4, suf= "", col_titles = None, renorm_heatmaps=True):
        """
        :param dict_: dictionary containing image tensors
        :param append_masks: whether to visualize the masks
        :param gif_savepath: the path to save the gif
        :param numex: how many examples of the batch to visualize
        :param suf: append a suffix to the gif name
        :param col_titles: a list of titles for each column

        The dictionary contains keys-values pairs of {"video_name":"image_tensor_list"}
        where "video_name" is used as the caption in the visualization
        where "image_tensor_list" is a list with np.arrays (batchsize, 64,64,n_channel) for each time step.

        If n_channel is 1 a heatmap will be shown. Use renorm_heatmaps=True to normalize the heatmaps
        at every time step (this is necessary when the range of values changes significantly over time).

        If the key contains the string "flow" a color-coded flow field will be shown.

        if the key contains the string "masks" the image_tensor_list needs to be of the following form:
        [mask_list_0, ..., mask_list_Tmax]
        where mask_list_t = [mask_0, ..., mask_N]
        where mask_i.shape = [batch_size, 64,64,1]
        """
        self.gif_savepath = filepath
        if dict_ == None:
            dict_ = cPickle.load(open(filepath + '/pred.pkl', "rb"))

        if 'iternum' in dict_:
            self.iternum = dict_['iternum']
            del dict_['iternum']
        else: self.iternum = 0

        if 'gen_images' in dict_:
            gen_images = dict_['gen_images']
            if gen_images[0].shape[0] < numex:
                raise ValueError("batchsize too small for providing desired number of exmaples!")

        self.numex = numex
        self.video_list = []

        for key in dict_.keys():
            print 'processing key {}'.format(key)
            data = dict_[key]

            if key ==  'ground_truth':  # special treatement for gtruth
                ground_truth = dict_['ground_truth']
                if not isinstance(ground_truth, list):
                    ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
                    if ground_truth[0].shape[0] == 1:
                        ground_truth = [g.reshape((1,64,64,3)) for g in ground_truth]
                    else:
                        ground_truth = [np.squeeze(g) for g in ground_truth]
                ground_truth = ground_truth[1:]

                if 'overlay_'+key in dict_:
                    overlay_points = dict_['overlay_'+key]
                    self.video_list.append((ground_truth, 'Ground Truth', overlay_points))
                else:
                    self.video_list.append((ground_truth, 'Ground Truth'))

            elif type(data[0]) is list or '_l' in key:    # for lists of videos
                print "the key \"{}\" contains {} videos".format(key, len(data[0]))
                if key == 'gen_masks' and not append_masks:
                    print 'skipping masks!'
                    continue
                vid_list = convert_to_videolist(data, repeat_last_dim=False)

                for i, m in enumerate(vid_list):
                    self.video_list.append((m, '{} {}'.format(key, i)))

            elif 'flow' in key:
                print 'visualizing key {} with colorflow'.format(key)
                self.video_list.append((visualize_flow(data), key))

            elif 'actions' in key:
                self.visualize_states_actions(dict_['states'], dict_['actions'])

            elif key == 'gen_distrib':  # if gen_distrib plot psum overtime!
                self.video_list.append((data, key))
                plot_psum_overtime(data, numex, filepath)
                desig_pos = dict_['desig_pos']
                plot_normed_at_desig_pos(data, filepath, desig_pos)

            else:
                if isinstance(data, list):
                    if data[0].shape[-3:] == (64,64,3) or data[0].shape[-3:] == (64,64,1):
                        self.video_list.append((data, key))
                else:
                    print 'ignoring key ',key

        self.renormalize_heatmaps = renorm_heatmaps
        print 'renormalizing heatmaps: ', self.renormalize_heatmaps


        self.t = 0

        self.suf = suf
        self.append_masks = append_masks
        self.num_rows = len(self.video_list)

        self.col_titles = col_titles

    def make_gifs(self, separate_vid = False):
        self.new_video_list = []
        print 'making gif with tags'
        for vid, name in self.video_list:
            print 'name: ', name
            if name ==  'gen_images':
                gen_images = vid

            if name == 'gen_distrib':
                vid = color_code_distrib(vid, self.numex, renormalize=True)
                vid = compute_overlay(gen_images, vid, self.numex)

            vid = upsample_nearest(vid, self.numex)
            self.new_video_list.append(vid)

        if separate_vid:
            vid_path = self.gif_savepath + '/sep_videos'
            if not os.path.exists(vid_path):
                os.mkdir(vid_path)
            for b in range(self.numex):
                frames = assemble_gif(self.new_video_list, convert_from_float=False, only_ind=b)
                save_video_mp4(vid_path + '/example{}'.format(b), frames)
        else:
            framelist = assemble_gif(self.new_video_list, convert_from_float=False)
            npy_to_gif(framelist, self.gif_savepath +'/direct{}{}'.format(self.iternum,self.suf))

    def visualize_states_actions(self, states, actions):

        plt.figure(figsize=(25, 2), dpi=80)

        for ex in range(self.numex):
            plt.subplot(1, self.numex, ex + 1)
            plt.axis('equal')

            move = actions[ex,:,:2]
            updown = actions[ex,:,2]
            rot = actions[ex, :,3]
            open = actions[ex ,:4]

            state_xy = states[ex, :,:2]
            alpha = states[ex,:, 3]

            action_startpoints = state_xy
            action_endpoints = state_xy + move

            plt.plot(state_xy[:,0], state_xy[:,1], '-o')
            plt.ylim([-0.17, 0.17])
            plt.xlim([0.46, 0.83])

            for t in range(states.shape[1]):

                x = [action_startpoints[t,0], action_endpoints[t, 0]]
                y = [action_startpoints[t,1], action_endpoints[t, 1]]
                if t % 2 == 0:
                    plt.plot(x, y, '--r')
                else:
                    plt.plot(x, y, '--y')

        # plt.show()
        plt.savefig(self.gif_savepath + "/actions_vis.png")
        plt.close('all')

    def make_image_strip(self, i_ex, tstart=5, tend=15):
        """
        :param i_ex:  the index of the example to flatten to the image strip
        :param tstart:
        :param tend:
        :return:
        """

        cols = tend - tstart +1

        width_per_ex = 1.

        standard_size = np.array([width_per_ex * cols, self.num_rows * 1.0])  ### 1.5
        # standard_size = np.array([6, 24])
        figsize = (standard_size * 1.0).astype(np.int)
        fig = plt.figure(num=1, figsize=figsize)


        outer_grid = gridspec.GridSpec(self.num_rows, 1)

        drow = 1. / self.num_rows

        self.im_handle_list = []

        axes_list = []

        for row in range(self.num_rows):
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, cols,
                                                          subplot_spec=outer_grid[row], wspace=0.0, hspace=0.0)
            image_row = self.video_list[row][0]

            im_handle_row = []
            col = 0
            for t in range(tstart, tend):

                ax = plt.Subplot(fig, inner_grid[col])
                ax.set_xticks([])
                ax.set_yticks([])
                axes_list.append(fig.add_subplot(ax))

                if image_row[0][i_ex].shape[-1] == 1:

                    im_handle = axes_list[-1].imshow(np.squeeze(image_row[t][i_ex]),
                                                     zorder=0, cmap=plt.get_cmap('jet'),
                                                     interpolation='none',
                                                     animated=True)
                else:
                    im_handle = axes_list[-1].imshow(image_row[t][i_ex], interpolation='none',
                                                     animated=True)

                im_handle_row.append(im_handle)

                col += 1
            self.im_handle_list.append(im_handle_row)

            # plt.figtext(.5, 1 - (row * drow * 0.990) - 0.01, self.video_list[row][1], va="center", ha="center",
            #             size=8)

        plt.savefig(self.gif_savepath + '/iter{}ex{}_overtime.png'.format(self.iternum, i_ex))


    def build_figure(self):

        # plot each markevery case for linear x and y scales
        root = Tk.Tk()
        root.rowconfigure(1, weight=1)
        root.columnconfigure(1, weight=1)

        frame = Frame(root)
        frame.grid(column=1, row=1, sticky=Tkconstants.NSEW)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(1, weight=1)

        if self.numex == 1:
            width_per_ex = 1.5
        else:
            width_per_ex = 0.9
        standard_size = np.array([width_per_ex*self.numex, self.num_rows * 1.0])  ### 1.5
        # standard_size = np.array([6, 24])
        figsize = (standard_size * 1.0).astype(np.int)
        fig = plt.figure(num=1, figsize=figsize)

        self.addScrollingFigure(fig, frame)

        buttonFrame = Frame(root)
        buttonFrame.grid(row=1, column=2, sticky=Tkconstants.NS)
        biggerButton = Button(buttonFrame, text="larger",
                              command=lambda: self.changeSize(fig, 1.5))
        biggerButton.grid(column=1, row=1)
        smallerButton = Button(buttonFrame, text="smaller",
                               command=lambda: self.changeSize(fig, .5))
        smallerButton.grid(column=1, row=2)

        axes_list = []
        l = []

        for vid in self.video_list:
            l.append(len(vid[0]))
        tlen = np.min(np.array(l))
        print 'minimum video length',tlen

        outer_grid = gridspec.GridSpec(self.num_rows, 1)

        drow = 1./self.num_rows

        self.im_handle_list = []
        self.plot_handle_list = []
        for row in range(self.num_rows):
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, self.numex,
                                                          subplot_spec=outer_grid[row], wspace=0.0, hspace=0.0)
            image_row = self.video_list[row][0]

            im_handle_row = []
            plot_handle_row = []
            for col in range(self.numex):
                ax = plt.Subplot(fig, inner_grid[col])
                ax.set_xticks([])
                ax.set_yticks([])
                axes_list.append(fig.add_subplot(ax))
                if row==0 and self.col_titles != None:
                    axes_list[-1].set_title(self.col_titles[col])

                if image_row[0][col].shape[-1] == 1:

                    im_handle = axes_list[-1].imshow(np.squeeze(image_row[0][col]),   # first timestep
                                                     zorder=0, cmap=plt.get_cmap('jet'),
                                                     interpolation='none',
                                                     animated=True)
                else:
                    im_handle = axes_list[-1].imshow(image_row[0][col], interpolation='none',
                                                     animated=True)

                if len(self.video_list[row]) == 3:
                    #overlay with markers:
                    coords = self.video_list[row][2][t][col]
                    plothandle = axes_list[-1].scatter(coords[1], coords[0], marker= "d", s=70, edgecolors='r', facecolors="None")
                    axes_list[-1].set_xlim(0, 63)
                    axes_list[-1].set_ylim(63, 0)
                    plot_handle_row.append(plothandle)
                else:
                    plot_handle_row.append(None)

                im_handle_row.append(im_handle)
            self.im_handle_list.append(im_handle_row)
            self.plot_handle_list.append(plot_handle_row)

            plt.figtext(.5, 1-(row*drow*1.)-0.001, self.video_list[row][1], va="center", ha="center", size=8)

        plt.axis('off')
        fig.tight_layout()

        # Set up formatting for the movie files
        Writer = animation.writers['imagemagick_file']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, self.animate,
                                       fargs= [self.im_handle_list, self.plot_handle_list, self.video_list, self.numex, self.num_rows, tlen],
                                       frames=tlen, interval=200, blit=True)

        if self.append_masks:
            self.suf = '_masks'+self.suf
        if self.gif_savepath != None:
            filepath = self.gif_savepath + '/animation{}{}.gif'.format(self.iternum,self.suf)
            print 'saving gif under: ', filepath
            anim.save(filepath, writer='imagemagick')
        root.mainloop()

    def changeSize(self, figure, factor):
        global canvas, mplCanvas, interior, interior_id, frame, cwid
        oldSize = figure.get_size_inches()
        print("old size is", oldSize)
        figure.set_size_inches([factor * s for s in oldSize])
        wi, hi = [i * figure.dpi for i in figure.get_size_inches()]
        print("new size is", figure.get_size_inches())
        print("new size pixels: ", wi, hi)
        mplCanvas.config(width=wi, height=hi)
        printBboxes("A")
        # mplCanvas.grid(sticky=Tkconstants.NSEW)
        canvas.itemconfigure(cwid, width=wi, height=hi)
        printBboxes("B")
        canvas.config(scrollregion=canvas.bbox(Tkconstants.ALL), width=200, height=200)
        figure.canvas.draw()
        printBboxes("C")
        print()

    def addScrollingFigure(self, figure, frame):
        global canvas, mplCanvas, interior, interior_id, cwid
        # set up a canvas with scrollbars
        canvas = Canvas(frame)
        canvas.grid(row=1, column=1, sticky=Tkconstants.NSEW)

        xScrollbar = Scrollbar(frame, orient=Tkconstants.HORIZONTAL)
        yScrollbar = Scrollbar(frame)

        xScrollbar.grid(row=2, column=1, sticky=Tkconstants.EW)
        yScrollbar.grid(row=1, column=2, sticky=Tkconstants.NS)

        canvas.config(xscrollcommand=xScrollbar.set)
        xScrollbar.config(command=canvas.xview)
        canvas.config(yscrollcommand=yScrollbar.set)
        yScrollbar.config(command=canvas.yview)

        # plug in the figure
        figAgg = FigureCanvasTkAgg(figure, canvas)
        mplCanvas = figAgg.get_tk_widget()
        # mplCanvas.grid(sticky=Tkconstants.NSEW)

        # and connect figure with scrolling region
        cwid = canvas.create_window(0, 0, window=mplCanvas, anchor=Tkconstants.NW)
        printBboxes("Init")
        canvas.config(scrollregion=canvas.bbox(Tkconstants.ALL), width=200, height=200)

    def animate(self, *args):
        global t
        _, im_handle_list, plot_handle_list, video_list, num_ex, num_rows, tlen = args

        artistlist = []
        for row in range(num_rows):
            image_row = video_list[row][0]                       #0 stands for images

            for col in range(num_ex):

                if image_row[0][col].shape[-1] == 1: # if visualizing with single-channel heatmap
                    im = np.squeeze(image_row[t][col])
                    if self.renormalize_heatmaps:
                        im = im/(np.max(im)+1e-5)
                    im_handle_list[row][col].set_array(im)
                else:
                    im_handle_list[row][col].set_array(image_row[t][col])

                if len(video_list[row]) == 3:
                    overlay_row = video_list[row][2]                       #2 stands for overlay
                    plot_handle_list[row][col].set_array(overlay_row[t][col]) #2 stands for overlay
                    # print "set array to", overlay_row[t][col]
                    artistlist.append(plot_handle_list[row][col])

            artistlist += im_handle_list[row]
        # print 'update at t', t
        t += 1

        if t == tlen:
            t = 0

        return artistlist

def printBboxes(label=""):
  global canvas, mplCanvas, interior, interior_id, cwid
  print("  "+label,
    "canvas.bbox:", canvas.bbox(Tkconstants.ALL),
    "mplCanvas.bbox:", mplCanvas.bbox(Tkconstants.ALL))


def convert_to_videolist(input, repeat_last_dim):
    tsteps = len(input)
    nmasks = len(input[0])

    list_of_videos = []

    for m in range(nmasks):  # for timesteps
        video = []
        for t in range(tsteps):
            if repeat_last_dim:
                single_mask_batch = np.repeat(input[t][m], 3, axis=3)
            else:
                single_mask_batch = input[t][m]
            video.append(single_mask_batch)
        list_of_videos.append(video)

    return list_of_videos

def upsample_nearest(imlist, numex, size = (256,256)):
    out = []
    for im in imlist:
        out_t = []
        for b in range(numex):
            out_t.append(scipy.misc.imresize(im[b], size, 'nearest'))
        out_t = np.stack(out_t, 0)
        out.append(out_t)
    return out

def color_code_distrib(distrib_list, num_ex, renormalize=False):
    print 'renormalizing heatmaps: ', renormalize
    out_distrib = []
    for distrib in distrib_list:
        out_t = []

        for b in range(num_ex):
            cmap = plt.cm.get_cmap('jet')
            if renormalize:
                distrib[b] /= (np.max(distrib[b])+1e-6)
            colored_distrib = cmap(np.squeeze(distrib[b]))[:, :, :3]
            out_t.append(colored_distrib)

            # plt.imshow(np.squeeze(distrib[b]))
            # plt.show()

        out_t = np.stack(out_t, 0)
        out_distrib.append(out_t)

    return out_distrib

def compute_overlay(images, color_coded_distrib, numex):
    alpha = .6
    output_list = []

    for im, distrib in zip(images, color_coded_distrib):
        out_t = []
        for b in range(numex):

            im_ex = im[b].astype(np.float32)
            distrib_ex = distrib[b].astype(np.float32)

            fused = distrib_ex * alpha + (1 - alpha) * im_ex
            out_t.append(fused)
        out_t = np.stack(out_t, 0)
        output_list.append(out_t)
    return output_list


def save_video_mp4(filename, frames):
    writer = imageio.get_writer(filename + '.mp4', fps=10)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    # import moviepy.editor as mpy
    # clip = mpy.ImageSequenceClip(frames, fps=4)
    # clip.write_gif(filename + '.gif')

if __name__ == '__main__':
    # file_path = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/data_amount_study/5percent_of_data/modeldata'
    file_path = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/alexmodel_finalpaper/improved_cdna_wristrot_k17d1_generatescratchimage_bs16/modeldata'
    # file_path = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/alexmodel_finalpaper/improved_cdna_wristrot_k17d1_generatescratchimage_adv_bs16/modeldata'

    v  = Visualizer_tkinter(append_masks=True, filepath=file_path, numex=10, renorm_heatmaps=False)
    # v.build_figure()
    v.make_gifs(separate_vid=True)
    # v.make_image_strip(i_ex=3)