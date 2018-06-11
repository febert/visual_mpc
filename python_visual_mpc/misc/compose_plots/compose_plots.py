import pickle as pkl
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs_single


def compose(dir):


    nstep = 10

    curr_images = []
    warped_image_start = []
    warped_image_goal = []

    for t step in range(nstep):


        tdict = pkl.load(open(dir + 'verbose/plan/pred_t{}iter2.pkl'.format(t), 'rb'))

        curr_image = tdict['curr_img_cam0']
        curr_image = add_crosshairs_single(curr_images, )

        warped_image_start = tdict['warp_start_cam0']
        warped_image_goal = tdict['warp_goal_cam0']



        warped_img_goal_cam = draw_text_onimage('%.2f' % reg_tradeoff[icam, 1],warped_image_goal[icam].squeeze())










if __name__ == '__main__':
    dir = '/mnt/sda1/visual_mpc/experiments/cem_exp/benchmarks_sawyer/weissgripper_regstartgoal_tradeoff/exp/succ3_pkl'
    compose(dir)
