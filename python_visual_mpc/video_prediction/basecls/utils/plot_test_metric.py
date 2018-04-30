import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pickle



def add_plot(file, label, color):
    dict = pickle.load(open(file, "rb"))

    pos0_exp_dist_l = dict['pos0_exp_dist_l']
    pos0_exp_dist = np.concatenate(pos0_exp_dist_l, axis=0)

    pos1_exp_dist_l = dict['pos1_exp_dist_l']
    pos1_exp_dist = np.concatenate(pos1_exp_dist_l, axis=0)

    mean_pos_exp = np.mean(np.concatenate([pos0_exp_dist, pos1_exp_dist], axis=0), axis=0)
    std_pos_exp = np.std(np.concatenate([pos0_exp_dist, pos1_exp_dist], axis=0), axis=0) / np.sqrt(128.)


    plt.errorbar(np.arange(mean_pos_exp.shape[0]), mean_pos_exp, yerr=std_pos_exp, fmt='o-', color=color,
                 label=label, linewidth=2)

    plt.legend(loc=0)


def plot(file):


    gray = '#4D4D4D'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    red = '#F15854'

    f = plt.figure(figsize=(6, 5))

    file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/alexmodel_finalpaper/improved_cdna_wristrot_k17d1_generatescratchimage_bs16/modeldata/metric_values.pkl'
    add_plot(file, 'Ours (L2)', red)

    file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/alexmodel_finalpaper/improved_cdna_wristrot_k17d1_generatescratchimage_adv_bs16/modeldata/metric_values.pkl'
    add_plot(file, 'Ours', blue)

    file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/alexmodel_finalpaper/orig_cdna_wristrot_k17_generatescratchimage_bs16/modeldata/metric_values.pkl'
    add_plot(file, 'CDNA (Finn et al., 2016)', orange)

    file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/sawyer/alexmodel_finalpaper/orig_sna_cdna_wristrot_k17_generatescratchimage_bs16/modeldata/metric_values.pkl'
    add_plot(file, 'SNA (Ebert et al., 2017)', pink)

    plt.xlabel('predicted times steps')
    plt.ylabel('distance between predicted and true object positions +-std.err')
    # plt.ylim((0,.54))
    # plt.xlim((1,15))

    plt.tight_layout()
    plt.savefig('./distance_metric.pdf')
    # plt.show()

if __name__ == '__main__':



    plot(file)





