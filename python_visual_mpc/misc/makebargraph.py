import numpy as np
from matplotlib import pyplot as plt


def long_dist_task():

    # order
    skip_OADNA = False

    dist_means_novel = [30.2, 25.5, 17.7, 18.3]
    if skip_OADNA:
        dist_means_novel = dist_means_novel[:-1]
    std_dev_means_novel = [3.6, 11.0, 9.7, 9.5]

    if skip_OADNA:
        std_dev_means_novel = std_dev_means_novel[:-1]

    dist_means_seen = [29.3,26.2, 8.93, 12.8]
    if skip_OADNA:
        dist_means_seen = dist_means_seen[:-1]
    std_dev_means_seen = [3.2, 12.7, 7.9, 6.6]
    if skip_OADNA:
        std_dev_means_seen = std_dev_means_seen[:-1]

    yerr_novel = [std_dev_means_novel, std_dev_means_novel]  # Standard deviation Data
    yerr_seen = [std_dev_means_seen, std_dev_means_seen]  # Standard deviation Data

    ind = np.arange(len(dist_means_novel))
    width = 0.35
    # colours = ['red', 'blue', 'green', 'yellow']

    plt.title('Long-distance pushing benchmark')
    plt.bar(ind, dist_means_novel, width, color='b', align='center', yerr=yerr_novel, ecolor='k',
            label='object not seen during training',alpha = .6)

    plt.bar(ind - width, dist_means_seen, width, color='g', align='center', yerr=yerr_seen, ecolor='k',alpha = .6,
            label='objects seen during training')

    plt.ylabel('distance to goal in pixels')

    if skip_OADNA:
        plt.xticks(ind - width / 2, ('random actions','DNA [Finn et al]','DNA [Finn et al] \nwith exp. dist.\n(ours)',
                                     ))
    else:
        plt.xticks(ind - width / 2, ('random actions', 'DNA [Finn et al]', 'DNA [Finn et al] \nwith exp. dist.\n(ours)',
                                     'OA-DNA \n with exp. dist.\n(ours)'))

    plt.legend(loc="upper left")
    # plt.show()
    if skip_OADNA:
        suf = 'only_dna'
    else: suf = ""
    plt.savefig('/home/frederik/Documents/documentation/doc_video/results_longdistance{}.png'.format(suf), dpi=400)


# def long_dist_task_imp():
#
#
#     # order
#     imp_means_novel = [8.1, 7.5, 1.2, -1.8]  # Mean Data
#     std_dev_imp_novel = [11.2, 10.5, 12.9, 3.6]
#
#     imp_means_seen = [16.2, 12.6, 4.9, -0.2]  # Mean Data
#     std_dev_imp_seen = [10.3, 9.6, 14.4, 0.5]
#     yerr_novel = [std_dev_imp_novel, std_dev_imp_novel]  # Standard deviation Data
#     yerr_seen = [std_dev_imp_seen, std_dev_imp_seen]  # Standard deviation Data
#
#     ind = np.arange(len(imp_means_novel))
#     width = 0.35
#     # colours = ['red', 'blue', 'green', 'yellow']
#
#     plt.title('Long-distance pushing benchmark')
#     plt.bar(ind, imp_means_novel, width, color='b', align='center', yerr=yerr_novel, ecolor='k', label='novel objects', alpha = .6)
#     plt.bar(ind-width, imp_means_seen, width,color='g', align='center', yerr=yerr_seen, ecolor='k', label='seen objects', alpha = .6)
#
#     plt.ylabel('Improvement of distance to goal in pixels')
#     plt.xticks(ind -width/2, ('DNA [Finn et al] \nwith exp. dist.\n(ours)', 'OA-DNA \n with exp. dist.\n(ours)', 'DNA [Finn et al] \n old dist. metr.', 'random actions'))
#
#     plt.legend(loc="upper right")
#     # plt.show()
#     plt.savefig('/home/guser/frederik/doc_video/results_longdistance_imp.png')

def mult_task():

    # order moved
    dist_means_novel_move = [11.9, 2.8]  # Mean Data
    std_dev_means_novel_move = [2.8, 2.35]

    dist_means_seen_move = [15.18, 8.1]  # Mean Data
    std_dev_means_seen_move = [3.9, 3.8]

    # stationary object
    dist_means_novel_stat = [0.97, 1.3]  # Mean Data
    std_dev_means_novel_stat = [0.3, 0.6]

    dist_means_seen_stat = [0.8, 1.1]  # Mean Data
    std_dev_means_seen_stat = [0.7, 0.7]
    yerr_novel_move = [std_dev_means_novel_move, std_dev_means_novel_move]  # Standard deviation Data
    yerr_seen_move = [std_dev_means_seen_move, std_dev_means_seen_move]  # Standard deviation Data
    yerr_novel_stat = [std_dev_means_novel_stat, std_dev_means_novel_stat]  # Standard deviation Data
    yerr_seen_stat = [std_dev_means_seen_stat, std_dev_means_seen_stat]  # Standard deviation Data

    ind = np.arange(2)
    # ind = np.arange(2) * 2
    width = 0.35
    # colours = ['red', 'blue', 'green', 'yellow']

    plt.title('Multi-objective pushing benchmark - Average Distance')
    plt.bar(ind - width, dist_means_novel_move, width, color='b', align='center', yerr=yerr_novel_move,
            ecolor='k', label='objects not seen during training', alpha = .6)
    plt.bar(ind, dist_means_seen_move, width, color='g', align='center', yerr=yerr_seen_move,
            ecolor='k', label='objects seen during training', alpha = .6)
    # plt.bar(ind-width*1, dist_means_novel_stat, width,color='r', align='center', yerr=yerr_novel_stat, ecolor='k', label='novel objects stationary',alpha = .6)
    # plt.bar(ind - width*0, dist_means_seen_stat, width, color='k', align='center', yerr=yerr_seen_stat, ecolor='k', label='seen objects stationary',alpha = .6)

    plt.ylabel('distance to goal in pixels')
    plt.xticks(ind - width / 2, ('DNA [Finn et al] \nwith expected distance\n(ours)', 'OA-DNA \n with expected distance\n(ours)'))

    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig('/home/frederik/Documents/documentation/doc_video/results_mult_obj_nostatic.png', dpi = 400)


# def mult_task_imp():
#
#     # order moved
#     dist_imp_novel_move = [0.8, 11.0]  # Mean Data
#     std_dev_imp_novel_move = [1.0, 3.3]
#
#     dist_imp_seen_move = [2.75,7.4]  # Mean Data
#     std_dev_imp_seen_move = [3.9,3.1]
#
#     # stationary object
#     dist_imp_novel_stat = [-0.8,-1.1]  # Mean Data
#     std_dev_imp_novel_stat = [0.7,0.7]
#
#     dist_imp_seen_stat = [-0.83, -1.12]  # Mean Data
#     std_dev_imp_seen_stat = [0.7,0.7]
#     yerr_novel_move = [std_dev_imp_novel_move, std_dev_imp_novel_move]  # Standard deviation Data
#     yerr_seen_move = [std_dev_imp_seen_move, std_dev_imp_seen_move]  # Standard deviation Data
#     yerr_novel_stat = [std_dev_imp_novel_stat, std_dev_imp_novel_stat]  # Standard deviation Data
#     yerr_seen_stat = [std_dev_imp_seen_stat, std_dev_imp_seen_stat]  # Standard deviation Data
#
#     ind = np.arange(2)*2
#     width = 0.35
#     # colours = ['red', 'blue', 'green', 'yellow']
#
#     plt.title('Multi-objective pushing benchmark - Improvement')
#     plt.bar(ind - width*3, dist_imp_novel_move, width, color='b', align='center', yerr=yerr_novel_move, ecolor='k', label='novel objects moved', alpha = .6)
#     plt.bar(ind-width*2, dist_imp_seen_move, width, color='g', align='center', yerr=yerr_seen_move, ecolor='k', label='seen objects moved', alpha = .6)
#     plt.bar(ind-width*1, dist_imp_novel_stat, width,color='r', align='center', yerr=yerr_novel_stat, ecolor='k', label='novel objects stationary',alpha = .6)
#     plt.bar(ind - width*0, dist_imp_seen_stat, width, color='k', align='center', yerr=yerr_seen_stat, ecolor='k', label='seen objects stationary',alpha = .6)
#
#     plt.ylabel('improvement on distance to goal in pixels')
#     plt.xticks(ind - width / 2, ('DNA [Finn et al] \nwith exp. dist.\n(ours)', 'OA-DNA \n with exp. dist.\n(ours)'))
#
#     plt.legend(loc="upper left")
#     # plt.show()
#     plt.savefig('/home/guser/frederik/doc_video/results_mult_obj_imp.png')


fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.17)

# long_dist_task()
# long_dist_task_imp()
mult_task()
# mult_task_imp()