import numpy as np
import os
import pdb

import matplotlib.pyplot as plt
import cPickle

from PIL import Image

class Getdesig(object):
    def __init__(self,img,basedir,img_namesuffix = '',mult = False):
        self.mult = mult
        self.suf = img_namesuffix
        self.basedir = basedir
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        if mult:
            self.desig = np.zeros((2,2))
        else:
            self.desig = None

        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.get_current_fig_manager().window.wm_geometry("+1300+400")
        self.i_click = 0

        print 'mark the correspoinding pixel of the object with the diamond'

        plt.show()

    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)

        if self.mult:
            if self.i_click == 0:

                self.desig[0] = np.array([event.ydata, event.xdata]).astype(np.int32)
                self.ax.scatter(self.desig[0,1], self.desig[0,0], s=100, marker="D", facecolors='r', edgecolors='r')
                print 'marked', self.desig[0]
                plt.draw()

                print 'mark the correspoinding pixel of the object with the circle/ball'

            if self.i_click == 1:
                self.desig[1] = np.array([event.ydata, event.xdata]).astype(np.int32)
                self.ax.scatter(self.desig[1, 1], self.desig[1, 0], s=100, marker="o", facecolors='r', edgecolors='r')
                print 'marked', self.desig[1]
                plt.draw()

            if self.i_click == 2:
                plt.savefig(self.basedir + '/marked_img' + self.suf)
                plt.close()
        else:
            if self.i_click == 0:

                self.desig = np.array([event.ydata, event.xdata]).astype(np.int32)
                self.ax.scatter(self.desig[1], self.desig[0], s=100, marker="D", facecolors='r', edgecolors='r')
                print 'marked', self.desig
                plt.draw()
            else:
                plt.savefig(self.basedir +'/marked_img'+self.suf)
                plt.close()

        self.i_click += 1



def run_exp_eval(novel, seen, obj_type, mult, DATA_DIR):
    print '-----------------------'
    print 'start evaluation'
    print 'dir: ', DATA_DIR
    print 'use mult objectives: ', mult
    print 'object type', obj_type
    print 'seen objects', seen
    print 'novel objects', novel

    score_dirs = []

    score_moved = []
    score_stat = []

    imp_moved = []
    imp_stat = []


    for path, subdirs, files in os.walk(DATA_DIR):
        # print path
        # print subdirs

        overwrite_scores = True

        done = False
        if 'desig_goal_pix_traj0.pkl' in files:

            while not done:

                print 'run number ',len(score_moved)

                exp_name = '/'.join(str.split(path, '/')[-3:-1])

                if mult:
                    checklast = 3
                else:
                    checklast = 2

                if obj_type =='novel':
                    if exp_name[-checklast:] not in novel:
                        done = True
                        continue

                elif obj_type =='seen':
                    if exp_name[-checklast:] not in seen:
                        done = True
                        continue
                else:
                    raise ValueError("invalid obj_type!")

                print 'using exp', exp_name

                scorefile = path + '/score.txt'

                if os.path.isfile(scorefile) and not overwrite_scores:
                    print 'scorefile {} already exists!'.format(scorefile)
                else:
                    # try:
                    print 'current directory:',path
                    start_img = Image.open(path + '/startimg__traj0.png')
                    fig1 = plt.figure()
                    ax = plt.imshow(start_img)
                    # ax.set_xlim(0, 63)
                    # ax.set_ylim(63, 0)
                    plt.get_current_fig_manager().window.wm_geometry("+400+400")
                    plt.draw()

                    final_img = Image.open(path + '/finalimage0.png')
                    c = Getdesig(final_img, path, mult= mult)
                    desig_pix = c.desig
                    dict = cPickle.load(open(path + '/desig_goal_pix_traj0.pkl', 'rb'))
                    goal_pix = dict['goal_pix']
                    old_desig_pix = dict['desig_pix']

                    plt.close(fig1)
                    plt.close('all')

                    if mult:
                        score_moved.append(np.linalg.norm((goal_pix[0] - desig_pix[0]).astype(np.float32)))
                        print 'score moved for {}:'.format(path)
                        print score_moved[-1]
                        imp_moved.append(
                            np.linalg.norm((goal_pix[0] - old_desig_pix[0]).astype(np.float32)) - score_moved[-1])
                        print 'impmoved', imp_moved

                        score_stat.append(np.linalg.norm((goal_pix[1] - desig_pix[1]).astype(np.float32)))
                        print 'score stationary for {}:'.format(path)
                        print score_stat[-1]
                        imp_stat.append(
                            np.linalg.norm((goal_pix[1] - old_desig_pix[1]).astype(np.float32)) - score_stat[-1])
                        print 'impstat',imp_stat


                        # print 'experiment name:', exp_name
                        score_dirs.append((exp_name))

                        with open(path + '/score.txt', 'w') as f:
                            f.write('distance moved:{} \n'.format(score_moved[-1]))
                            f.write('distance stationary:{} \n'.format(score_stat[-1]))
                            f.write('combined {} \n'.format(score_moved[-1] + score_stat[-1]))
                    else:
                        if len(goal_pix.shape) == 2:

                            score_moved.append(np.linalg.norm((goal_pix[0] - desig_pix).astype(np.float32)))
                            print 'score moved for {}:'.format(path)
                            print score_moved[-1]
                            imp_moved.append(
                                np.linalg.norm((goal_pix[0] - old_desig_pix[0]).astype(np.float32)) - score_moved[-1])
                            print 'impmoved', imp_moved[-1]

                        else:
                            score_moved.append(np.linalg.norm((goal_pix - desig_pix).astype(np.float32)))
                            print 'score moved for {}:'.format(path)
                            print score_moved[-1]
                            imp_moved.append(
                                np.linalg.norm((goal_pix - old_desig_pix).astype(np.float32)) - score_moved[-1])
                            print 'impmoved', imp_moved[-1]

                        exp_name = '/'.join(str.split(path, '/')[-3:-1])
                        print 'experiment name:', exp_name
                        score_dirs.append((exp_name))

                        # print 'goalpix', goal_pix
                        # print 'desigpix', desig_pix

                        with open(path + '/score.txt', 'w') as f:
                            f.write('distance from designated pixel to goal:{}'.format(score_moved[-1]))

                char = raw_input('press c to continue or r to repeat')

                if char == 'c':
                    print 'continuing...'
                    done = True
                elif char == 'r':
                    print 'repeating this example!'
                    score_dirs.pop()
                    score_moved.pop()
                    score_stat.pop()
                    imp_moved.pop()
                    imp_stat.pop()
                    done = False

    if mult:
        score_moved = np.array(score_moved)
        score_stat = np.array(score_stat)
        avg_distance_moved = np.average(score_moved)
        avg_distance_stat = np.average(score_stat)

        std_distance_moved = np.std(score_moved)
        std_distance_stat = np.std(score_stat)

        imp_moved = np.array(imp_moved)
        imp_stat = np.array(imp_stat)
        avg_imp_moved = np.average(imp_moved)
        avg_imp_stat = np.average(imp_stat)
        std_imp_moved = np.std(imp_moved)
        std_imp_stat = np.std(score_stat)

        summed = score_moved + score_stat
        avg_distance_summed = np.average(summed)
        std_distance_summed = np.std(summed)

        summed_imp = imp_moved + imp_stat
        avg_imp_summed = np.average(summed_imp)
        std_imp_summed = np.std(summed_imp)

        with open(DATA_DIR + '/scores_summary{}.txt'.format(obj_type), 'w') as f:

            f.write('average distance moved {}, std:{} \n'.format(avg_distance_moved, std_distance_moved))
            f.write('average distance stationary {}, std:{}  \n'.format(avg_distance_stat, std_distance_stat))

            f.write('average improvement moved {}, std:{}  \n'.format(avg_imp_moved, std_imp_moved))
            f.write('average improvement stat {}, std:{}  \n'.format(avg_imp_stat, std_imp_stat))

            f.write('summed average distance{}, std:{}  \n'.format(avg_distance_summed, std_distance_summed))
            f.write('summed average improvement{}, std:{}  \n'.format(avg_imp_summed, std_imp_summed))

            for iexp in range(len(score_dirs)):
                f.write('exp:{0}, score moved:{1}, score station{2} dir:{3} \n'.format(iexp, score_moved[iexp], score_stat[iexp], score_dirs[iexp]))
    else:
        score_moved = np.array(score_moved)
        avg_distance_moved = np.average(score_moved)
        std_distance_moved = np.std(score_moved)

        imp_moved = np.array(imp_moved)
        avg_imp_moved = np.average(imp_moved)
        std_imp_moved = np.std(imp_moved)

        with open(DATA_DIR + '/scores_summary{}.txt'.format(obj_type), 'w') as f:

            f.write('average distance moved {}, std:{} \n'.format(avg_distance_moved, std_distance_moved))
            f.write('average improvement moved {}, std:{}  \n'.format(avg_imp_moved, std_imp_moved))

            for iexp in range(len(score_dirs)):
                f.write('exp:{0}, score:{1}, imp{2}, dir:{3} \n'.format(iexp, score_moved[iexp], imp_moved[iexp], score_dirs[iexp]))




if __name__ == '__main__':
    current_dir = __file__


    basedir = '/'.join(str.split(current_dir, '/')[:-2])

    mult = True
    seen = {'2,5'}
    novel = {'6,7'}

    DATA_DIR = basedir + '/cdna_multobj_1stimg/multobj_confs'
    obj_type = 'seen'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)
    obj_type = 'novel'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)

    #
    DATA_DIR = basedir + '/dna_multobj/multobj_confs'
    obj_type = 'seen'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)
    obj_type = 'novel'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)

    mult = False

    novel = {'o5', 'o6', 'o7'}
    seen = {'o2'}
    obj_type = 'seen'

    DATA_DIR = basedir + '/predprop/longdistance'
    obj_type = 'seen'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)
    obj_type = 'novel'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)


    DATA_DIR = basedir + '/random_baseline/longdistance'

    obj_type = 'seen'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)
    obj_type = 'novel'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)

    DATA_DIR = basedir + '/predprop_1stimg_bckgd/longdistance'
    obj_type = 'seen'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)
    obj_type = 'novel'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)


    DATA_DIR = basedir + '/singlepoint_eval/longdistance'
    obj_type = 'seen'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)
    obj_type = 'novel'
    run_exp_eval(novel, seen, obj_type, mult, DATA_DIR)


