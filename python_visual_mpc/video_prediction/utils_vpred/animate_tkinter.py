import numpy as np
from matplotlib import animation
import matplotlib.gridspec as gridspec

import pdb
import cPickle

import Tkinter as Tk

from Tkinter import Button, Frame, Canvas, Scrollbar
import Tkconstants

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

frame = None
canvas = None

t = 0
class Visualizer_tkinter(object):
    # def __init__(self, conf, file_path, name= "", examples = 4, show_parts=False):
    def __init__(self, append_masks = True):

        dict_ = cPickle.load(open(file_path + '/pred.pkl', "rb"))
        gen_images = dict_['gen_images']

        self.num_ex = 4
        self.video_list = []

        if 'ground_truth' in dict_:
            ground_truth = dict_['ground_truth']
            if not isinstance(ground_truth, list):
                ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
                ground_truth = [np.squeeze(g) for g in ground_truth]
            ground_truth = ground_truth[1:]

            self.video_list.append((ground_truth, 'Ground Truth'))

        self.video_list.append((gen_images, 'Gen Images'))

        if 'gen_distrib' in dict_:
            gen_pix_distrib = dict_['gen_distrib']
            self.video_list.append((gen_pix_distrib, 'Gen distrib'))

        if append_masks:
            gen_masks = dict_['gen_masks']
            gen_masks = convert_to_videolist(gen_masks, repeat_last_dim=False)

            for i,m in enumerate(gen_masks):
                self.video_list.append((m,'mask {}'.format(i)))

        # if 'flow_vectors' in dict_:
        #     self.videolist.append(visualize_flow(dict_))

        self.build_figure()
        self.t = 0


    def build_figure(self):


        # plot each markevery case for linear x and y scales
        root = Tk.Tk()
        root.rowconfigure(1, weight=1)
        root.columnconfigure(1, weight=1)

        frame = Frame(root)
        frame.grid(column=1, row=1, sticky=Tkconstants.NSEW)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(1, weight=1)

        standard_size = np.array([6, 24])
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

        self.num_rows = len(self.video_list)

        l = []

        for vid in self.video_list:
            l.append(len(vid[0]))
        tlen = np.min(np.array(l))
        print 'minimum video length',tlen

        outer_grid = gridspec.GridSpec(self.num_rows, 1)

        drow = 1./self.num_rows

        self.im_handle_list = []
        for row in range(self.num_rows):
            # outer_ax = fig.add_subplot(outer_grid[row])
            # if self.row_list[row][1] != '':
            #     outer_ax.set_title(self.row_list[1])

            inner_grid = gridspec.GridSpecFromSubplotSpec(1, self.num_ex,
                              subplot_spec=outer_grid[row], wspace=0.0, hspace=0.0)

            image_row = self.video_list[row][0]

            im_handle_row = []
            for col in range(self.num_ex):
                ax = plt.Subplot(fig, inner_grid[col])
                ax.set_xticks([])
                ax.set_yticks([])
                axes_list.append(fig.add_subplot(ax))
                # if row==0:
                #     axes_list[-1].set_title('example {}'.format(col))

                if image_row[0][col].shape[-1] == 1:
                    im_handle = axes_list[-1].imshow(np.squeeze(image_row[0][col]),
                                                     zorder=0, cmap=plt.get_cmap('jet'),
                                                     interpolation='none',
                                                     animated=True)
                else:
                    im_handle = axes_list[-1].imshow(image_row[0][col], interpolation='none',
                                                     animated=True)

                im_handle_row.append(im_handle)
            self.im_handle_list.append(im_handle_row)

            plt.figtext(.5, 1-(row*drow*0.995)-0.005, self.video_list[row][1], va="center", ha="center", size=15)

        plt.axis('off')
        fig.tight_layout()

        # initialization function: plot the background of each frame

        # Set up formatting for the movie files
        Writer = animation.writers['imagemagick_file']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate,
                                       fargs= [self.im_handle_list, self.video_list, self.num_ex, self.num_rows, tlen],
                                       frames=tlen, interval=100, blit=True)
        # anim.save('basic_animation.gif', writer='imagemagick')
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

def printBboxes(label=""):
  global canvas, mplCanvas, interior, interior_id, cwid
  print("  "+label,
    "canvas.bbox:", canvas.bbox(Tkconstants.ALL),
    "mplCanvas.bbox:", mplCanvas.bbox(Tkconstants.ALL))

def animate(*args):
    global t
    _, im_handle_list, video_list, num_ex, num_rows, tlen = args

    artistlist = []
    for row in range(num_rows):
        image_row = video_list[row][0]
        for col in range(num_ex):
            if image_row[0][col].shape[-1] == 1:
                im_handle_list[row][col].set_array(np.squeeze(image_row[t][col]))
            else:
                im_handle_list[row][col].set_array(image_row[t][col])
        artistlist += im_handle_list[row]

    print 'update at t', t
    t += 1

    if t == tlen:
        t = 0

    return artistlist

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


if __name__ == '__main__':
    file_path = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/1stimg_bckgd_cdna/modeldata'
    v  = Visualizer_tkinter()